"""
模块名称：supervisor
功能描述：Supervisor 循环编排节点，Agent 系统的"大脑"。负责意图分类、复合意图拆解、
         子 Agent 路由调度，以及循环终止判断。通过 LLM 结构化输出实现意图解析，
         按优先级顺序逐一分发给子 Agent 执行。

核心职责：
    1. 调用 LLM 对用户输入进行意图分类（支持复合意图拆解）
    2. 按优先级排序意图队列，逐一路由到对应子 Agent
    3. 子 Agent 执行完毕后检查队列是否清空
    4. 执行循环终止保护（最大循环次数、超时）
"""

from pathlib import Path

from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm_parse
from ..state import MAX_SUPERVISOR_LOOPS, AgentState
from .helpers import build_profile_summary, extract_latest_query

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "supervisor")

# ================= 默认配置 =================
SUPERVISOR_LLM_MODEL: str = config.SUPERVISOR_MODEL
PROMPT_TEMPLATE: str = "agent/supervisor_classify"


# ── 意图类型 → 节点名映射 ──
INTENT_TO_AGENT: dict[str, str] = {
    "EMERGENCY": "emergency",
    "MEDICAL_QUERY": "medical",
    "DEVICE_CONTROL": "device",
    "CHITCHAT": "chat",
}

# ── 降级默认返回值 ──
_FALLBACK_STATE: dict = {
    "pending_intents": [],
    "current_agent": "chat",
    "risk_level": "low",
    "loop_count": 1,
    "retry_count": 0,
}

# ────────────────────────────────────────────────────────────
# Pydantic 结构化输出 Schema
# ────────────────────────────────────────────────────────────


class IntentItem(BaseModel):
    """单个意图的结构化表示。"""

    type: str = Field(description="意图类型: EMERGENCY / MEDICAL_QUERY / DEVICE_CONTROL / CHITCHAT")
    sub_query: str = Field(description="该意图对应的具体子查询")
    priority: int = Field(description="优先级数值，越小越高")


class SupervisorOutput(BaseModel):
    """Supervisor LLM 的结构化输出。"""

    intents: list[IntentItem] = Field(description="识别到的意图列表")
    risk_level: str = Field(default="low", description="整体风险等级")


# ────────────────────────────────────────────────────────────
# Supervisor 节点
# ────────────────────────────────────────────────────────────


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor 编排节点：意图分类 + 路由决策。

    执行逻辑：
        1. 如果 pending_intents 非空，说明上一个子 Agent 已完成，取下一个意图
        2. 如果 pending_intents 为空且是首次进入，调用 LLM 分类意图
        3. 如果 pending_intents 为空且非首次，说明所有意图已处理完毕

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 pending_intents、current_agent、risk_level、loop_count 的状态更新
    """
    loop_count = state.get("loop_count", 0)
    pending = state.get("pending_intents", [])

    # ── 循环保护 ──
    if loop_count >= MAX_SUPERVISOR_LOOPS:
        logger.warning(f"达到最大循环次数 {MAX_SUPERVISOR_LOOPS}，强制终止")
        return {
            "current_agent": "done",
        }

    # ── 如果有待处理意图，取出下一个 ──
    if pending:
        return _dispatch_next(state)

    # ── 首次进入：调用 LLM 分类意图 ──
    if loop_count == 0:
        return _classify_and_dispatch(state)

    # ── 所有意图已处理完毕 ──
    logger.info("所有意图已处理完毕，进入输出阶段")
    return {
        "current_agent": "done",
        "loop_count": loop_count + 1,
    }


# ────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────


def _dispatch_next(state: AgentState) -> dict:
    """从意图队列中取出下一个并分发，同时将 sub_query 写入 state。"""
    next_intent, *remaining = state.get("pending_intents", [])
    agent = INTENT_TO_AGENT.get(next_intent["type"], "chat")

    logger.info(
        f"分发意图 | type={next_intent['type']} | agent={agent} | "
        f"sub_query={next_intent.get('sub_query', '')[:30]}... | 剩余={len(remaining)}"
    )

    return {
        "pending_intents": remaining,
        "current_agent": agent,
        "current_sub_query": next_intent.get("sub_query", ""),
        "loop_count": state.get("loop_count", 0) + 1,
        "retry_count": 0,
        "total_turns": state.get("total_turns", 0) + 1,
    }


def _classify_and_dispatch(state: AgentState) -> dict:
    """调用 LLM 分类意图，取第一个立即分发，其余入队。"""
    user_query = extract_latest_query(state)
    if not user_query:
        logger.warning("无法提取用户查询，降级为闲聊")
        return {**_FALLBACK_STATE}

    logger.info(f"意图分类开始 | user_query={user_query}")
    # 构建 Prompt 并调用 LLM
    messages = prompt_manager.build_prompt(
        PROMPT_TEMPLATE,
        user_query=user_query,
        user_emotion=state.get("user_emotion", "NEUTRAL"),
        current_image_context=state.get("current_image_context", ""),
        conversation_summary=state.get("conversation_summary", ""),
        user_profile_summary=build_profile_summary(state.get("user_profile", {})),
    )

    parsed = call_llm_parse(SUPERVISOR_LLM_MODEL, messages, SupervisorOutput)
    if parsed is None or not parsed.intents:
        logger.warning("意图分类失败或为空，降级为闲聊")
        return {**_FALLBACK_STATE}

    # 按优先级排序
    sorted_intents = sorted(
        [intent.model_dump() for intent in parsed.intents],
        key=lambda x: x["priority"],
    )

    first, *remaining = sorted_intents
    logger.info(
        f"意图分类完成 | types={[i['type'] for i in sorted_intents]} | risk={parsed.risk_level}"
    )

    return {
        "pending_intents": remaining,
        "current_agent": INTENT_TO_AGENT.get(first["type"], "chat"),
        "current_sub_query": first.get("sub_query", ""),
        "risk_level": parsed.risk_level,
        "loop_count": 1,
        "retry_count": 0,
        "total_turns": state.get("total_turns", 0) + 1,
    }


# ────────────────────────────────────────────────────────────
# 路由函数（供 LangGraph conditional_edges 使用）
# ────────────────────────────────────────────────────────────


def route_by_intent(state: AgentState) -> str:
    """
    条件路由函数：根据 current_agent 决定下一个执行节点。

    Args:
        state: 当前 AgentState

    Returns:
        str: 目标节点名称
    """
    agent = state.get("current_agent", "done")
    logger.debug(f"路由决策 | current_agent={agent}")
    return agent
