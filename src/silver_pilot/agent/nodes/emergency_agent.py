"""
模块名称：emergency
功能描述：Emergency Agent 节点，处理 P0 最高优先级的紧急情况。跳过常规流程，
         立即执行安抚用户、通知紧急联系人、记录事件三个并行任务。

执行流程：
    1. 跳过常规流程，直接进入紧急模式
    2. 生成即时安抚回复
    3. 触发紧急联系人通知（send_alert 工具）
    4. 将事件记录到 AgentState 供 MemoryWriter 持久化
"""

from pathlib import Path

from langchain_core.messages import AIMessage

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm
from ..state import AgentState
from ..tools import ToolExecutor
from .helpers import extract_latest_query

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "emergency_agent")

# ================= 默认配置 =================
EMERGENCY_LLM_MODEL: str = config.EMERGENCY_AGENT_MODEL
PROMPT_TEMPLATE: str = "agent/emergency_respond"

DEFAULT_COMFORT: str = (
    "我在这里陪着您，不要担心。我正在帮您联系家人，请尽量保持冷静。如果情况紧急，请拨打 120。"
)

# ────────────────────────────────────────────────────────────
# Emergency Agent 节点
# ────────────────────────────────────────────────────────────


def emergency_agent_node(state: AgentState) -> dict:
    """
    Emergency Agent 节点：紧急情况即时响应。

    并行执行三项任务：
        1. 生成安抚性回复（即时返回给用户）
        2. 触发紧急联系人通知
        3. 记录紧急事件到 State（供后续持久化）

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 messages、tool_results、safety_flags、risk_level、final_response 的状态更新
    """
    user_query = extract_latest_query(state)
    user_emotion = state.get("user_emotion", "NEUTRAL")
    profile = state.get("user_profile", {})
    contacts = profile.get("emergency_contacts", [])
    contact_names = [c["name"] for c in contacts if isinstance(c, dict) and c.get("name")]

    logger.warning(f"🚨 Emergency Agent 触发 | emotion={user_emotion} | query={user_query[:50]}...")

    # ── 任务 1: 生成即时安抚回复 ──
    comfort_response = _generate_comfort_response(user_query, user_emotion, contact_names)

    # ── 任务 2: 触发紧急联系人通知 ──
    alert_results = _send_emergency_alerts(contacts, user_query)

    # ── 任务 3: 构建紧急事件记录 ──
    safety_flags = state.get("safety_flags", [])
    if "emergency_triggered" not in safety_flags:
        safety_flags = [*safety_flags, "emergency_triggered"]

    # 组装最终回复：安抚 + 通知状态
    response_parts = [comfort_response]
    if alert_results:
        notified_names = [r.get("contact", "联系人") for r in alert_results if r.get("success")]
        if notified_names:
            response_parts.append(
                f"我已经帮您通知了{', '.join(notified_names)}，他们很快会联系您。"
            )

    final_response = "\n".join(response_parts)

    logger.warning(f"Emergency Agent 完成 | 已通知联系人数={len(alert_results)}")

    return {
        "messages": [AIMessage(content=final_response)],
        "tool_results": alert_results,
        "safety_flags": safety_flags,
        "risk_level": "critical",
        "final_response": final_response,
    }


# ────────────────────────────────────────────────────────────
# 安抚回复生成
# ────────────────────────────────────────────────────────────


def _generate_comfort_response(user_query: str, user_emotion: str, contact_names: list[str]) -> str:
    """
    生成即时安抚回复。

    Args:
        user_query: 用户紧急描述
        user_emotion: 情感标签
        contact_names: 紧急联系人列表

    Returns:
        str: 安抚性回复文本
    """
    names_text = "、".join(contact_names) if contact_names else "您的家人"
    messages = prompt_manager.build_prompt(
        PROMPT_TEMPLATE,
        user_query=user_query,
        user_emotion=user_emotion,
        emergency_contacts=names_text,
    )
    return call_llm(EMERGENCY_LLM_MODEL, messages, max_tokens=200) or DEFAULT_COMFORT


# ────────────────────────────────────────────────────────────
# 紧急通知发送
# ────────────────────────────────────────────────────────────


def _send_emergency_alerts(contacts: list[dict], emergency_description: str) -> list[dict]:
    """
    向所有紧急联系人发送通知。

    Args:
        contacts: 紧急联系人列表
        emergency_description: 紧急情况描述

    Returns:
        list[dict]: 各通知的执行结果
    """
    if not contacts:
        logger.warning("用户未设置紧急联系人，跳过通知")
        return []

    executor = ToolExecutor()
    results: list[dict] = []

    for contact in contacts:
        if not isinstance(contact, dict):
            continue

        contact_name = contact.get("name", "")
        contact_phone = contact.get("phone", "")

        if not contact_name and not contact_phone:
            continue

        result = executor.execute(
            "send_alert",
            {
                "contact": contact_phone or contact_name,
                "message": f"紧急通知：您的家人可能需要帮助。情况描述：{emergency_description[:100]}",
                "urgency": "critical",
            },
            user_confirmed=True,  # 紧急情况自动确认，跳过 HITL
        )
        result_dict = result.to_dict()
        result_dict["contact"] = contact_name
        results.append(result_dict)

    logger.info(f"紧急通知发送完成 | 发送数={len(results)}")
    return results
