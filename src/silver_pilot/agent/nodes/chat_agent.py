"""
模块名称：chat
功能描述：Chat Agent 节点，负责处理日常闲聊和情感陪伴类对话。根据 SenseVoice
         输出的情感标签调整回复策略，以温和、耐心、简洁的"小银"人格回应用户。

情绪应对策略：
    - SURPRISED → 顺应惊讶情绪，共同探索原因
    - NEUTRAL → 常规陪聊，分享积极话题
    - HAPPY → 积极回应，分享喜悦，维持轻松氛围
    - SAD → 先共情倾听，再温和引导
    - DISGUSTED → 表达理解，尊重感受，适时转移话题
    - ANGRY → 耐心倾听，询问是否需要帮助
    - FEARFUL → 安抚情绪，提供安全感
"""

from pathlib import Path

from langchain_core.messages import AIMessage

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm
from ..state import AgentState
from .helpers import extract_latest_query

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "chat_agent")

# ================= 默认配置 =================
CHAT_AGENT_MODEL: str = config.CHAT_AGENT_MODEL
PROMPT_TEMPLATE: str = "agent/chat_generate"

# ================= 情绪 → 兜底回复 =================
EMOTION_FALLBACKS: dict[str, str] = {
    "SURPRISED": "哇，听起来很让人惊讶呢！发生什么事了？",
    "NEUTRAL": "我在呢，有什么想聊的吗？",
    "HAPPY": "看您心情不错呀，有什么开心事儿吗？",
    "SAD": "我能感觉到您心情不太好，我陪着您呢。想聊聊吗？",
    "DISGUSTED": "听起来这确实挺让人反感的，我完全理解。要不我们换个话题聊聊？",
    "ANGRY": "我理解您现在的感受，别着急慢慢说，我在听。",
    "FEARFUL": "别担心，我就在这里陪着您。需要我帮您联系家人吗？",
}

# ────────────────────────────────────────────────────────────
# Chat Agent 节点
# ────────────────────────────────────────────────────────────


def chat_agent_node(state: AgentState) -> dict:
    """
    Chat Agent 节点：情绪感知型闲聊陪伴。

    根据 ``user_emotion`` 动态调整回复策略和语气，
    以"小银"人格生成温暖、简洁的回复。

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 messages 和 sub_response 的状态更新
    """
    user_query = extract_latest_query(state)
    user_emotion = state.get("user_emotion", "NEUTRAL")

    logger.info(f"Chat Agent 开始处理 | emotion={user_emotion} | query={user_query[:30]}...")

    # 构建 Prompt
    messages = prompt_manager.build_prompt(
        PROMPT_TEMPLATE,
        user_query=user_query,
        user_emotion=user_emotion,
        current_image_context=state.get("current_image_context", ""),
        conversation_summary=state.get("conversation_summary", ""),
    )

    # 调用 LLM
    answer = call_llm(
        CHAT_AGENT_MODEL,
        messages,
        temperature=config.CHAT_AGENT_TEMPERATURE,
        max_tokens=config.CHAT_AGENT_MAX_TOKENS,
    )
    if answer is None:
        answer = EMOTION_FALLBACKS.get(user_emotion, EMOTION_FALLBACKS["NEUTRAL"])

    logger.info(f"Chat Agent 完成 | 回复长度={len(answer)}")

    return {
        "messages": [AIMessage(content=answer)],
        "sub_response": state.get("sub_response", []) + [answer],
    }
