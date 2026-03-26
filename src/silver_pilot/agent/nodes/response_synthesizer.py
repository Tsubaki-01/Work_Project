"""
模块名称：response_synthesizer
功能描述：回复综合节点，汇总本轮所有子 Agent 的 AIMessage 输出，
         通过 LLM 综合为一条面向老年用户的连贯最终回复。

设计思路：
    - 从 state["messages"] 中提取最近一条 HumanMessage 之后的所有 AIMessage
    - 将它们交给 LLM 综合为一条简洁、温暖、连贯的回答
    - 单个 Agent 仅有一条回复时，跳过 LLM 调用，直接透传（节省延迟）

图拓扑位置：
    supervisor(done) → response_synthesizer → output_guard → memory_writer → END
"""

from pathlib import Path

from langchain_core.messages import AIMessage

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm
from ..state import AgentState
from .helpers import extract_ai_messages_after_last_human, message_to_text, messages_to_text

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "response_synthesizer")

# ================= 默认配置 =================
SYNTHESIZER_MODEL: str = getattr(config, "RESPONSE_SYNTHESIZER_MODEL", "qwen-flash")
SYNTHESIZER_TEMPERATURE: float = getattr(config, "RESPONSE_SYNTHESIZER_TEMPERATURE", 0.3)
SYNTHESIZER_MAX_TOKENS: int = getattr(config, "RESPONSE_SYNTHESIZER_MAX_TOKENS", 800)
PROMPT_TEMPLATE: str = "agent/response_synthesize"

# ────────────────────────────────────────────────────────────
# Response Synthesizer 节点
# ────────────────────────────────────────────────────────────


def response_synthesizer_node(state: AgentState) -> dict:
    """
    回复综合节点：将本轮多个子 Agent 的输出合并为一条连贯回复。

    逻辑：
        1. 提取最后一条 HumanMessage 之后的所有 AIMessage
        2. 如果只有 1 条 → 直接透传（省去 LLM 开销）
        3. 如果有多条 → 调用 LLM 综合为一条回复
        4. 将结果写入 sub_response（供 output_guard 消费）

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 sub_response 的状态更新
    """
    messages = state.get("messages", [])
    user_query, ai_messages = extract_ai_messages_after_last_human(messages)

    logger.info(
        f"Response Synthesizer 开始 | 用户查询长度={len(user_query)} | AI 回复数={len(ai_messages)}"
    )

    # 无内容：兜底
    if not ai_messages:
        logger.warning("无 AI 回复可综合，返回空")
        return {"sub_response": state.get("sub_response", [])}

    # 单条回复：直接透传
    if len(ai_messages) == 1:
        logger.info("仅一条 AI 回复，跳过 LLM 综合，直接透传")
        return {"sub_response": [message_to_text(ai_messages[0])]}

    # 多条回复：LLM 综合
    synthesized = _synthesize_responses(user_query, ai_messages)

    logger.info(f"Response Synthesizer 完成 | 综合回复长度={len(synthesized)}")
    return {"sub_response": [synthesized]}


def _synthesize_responses(user_query: str, ai_messages: list[AIMessage]) -> str:
    """
    调用 LLM 将多条子 Agent 回复综合为一条连贯回复。

    Args:
        user_query: 用户原始查询
        ai_messages: 各子 Agent 的回复列表

    Returns:
        str: 综合后的回复
    """
    # 格式化各段回复
    numbered_parts = []
    for i, msg in enumerate(ai_messages, 1):
        numbered_parts.append(f"回复{i}：{message_to_text(msg)}")
    all_responses = "\n".join(numbered_parts)

    # 构建 Prompt
    messages = prompt_manager.build_prompt(
        PROMPT_TEMPLATE,
        user_query=user_query,
        agent_responses=all_responses,
        response_count=len(ai_messages),
    )

    try:
        result = call_llm(
            SYNTHESIZER_MODEL,
            messages,
            temperature=SYNTHESIZER_TEMPERATURE,
            max_tokens=SYNTHESIZER_MAX_TOKENS,
        )
        if result:
            return result
    except Exception as e:
        logger.error(f"LLM 综合调用失败: {e}，回退到简单拼接")

    # 降级：简单拼接
    return messages_to_text(ai_messages)
