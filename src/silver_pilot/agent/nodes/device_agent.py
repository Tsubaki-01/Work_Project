"""
模块名称：device
功能描述：Device Agent 节点，负责处理设备控制类指令。将用户的自然语言指令
         解析为结构化工具调用，经风险评级后执行或请求用户确认。

执行流程：
    1. 调用 LLM 将自然语言解析为工具调用 JSON
    2. Risk Evaluator 评估风险等级
    3. LOW → 直接执行 | MEDIUM → 确认后执行 | HIGH → HITL 中断
    4. 执行结果写回 AgentState
"""

import time
from pathlib import Path

from langchain_core.messages import AIMessage
from openai import OpenAI
from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm_parse
from ..state import AgentState
from ..tools import ToolExecutor
from .helpers import build_profile_summary, extract_latest_query

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "device_agent")

# ================= 默认配置 =================
DEVICE_LLM_MODEL: str = config.DEVICE_AGENT_MODEL
PROMPT_TEMPLATE: str = "agent/device_parse"
DEFAULT_TEMPERATURE = config.DEVICE_AGENT_TEMPERATURE

# ────────────────────────────────────────────────────────────
# Pydantic 结构化输出 Schema
# ────────────────────────────────────────────────────────────


class ToolCallItem(BaseModel):
    """单个工具调用的结构化表示。"""

    tool_name: str = Field(description="工具名称")
    arguments: dict = Field(description="工具参数")


class DeviceParseOutput(BaseModel):
    """Device Agent LLM 的结构化输出。"""

    tool_calls: list[ToolCallItem] = Field(description="解析出的工具调用列表")


# ────────────────────────────────────────────────────────────
# 模块级单例
# ────────────────────────────────────────────────────────────

_executor: ToolExecutor | None = None


def _get_executor() -> ToolExecutor:
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor


# ────────────────────────────────────────────────────────────
# Device Agent 节点
# ────────────────────────────────────────────────────────────


def device_agent_node(state: AgentState) -> dict:
    """
    Device Agent 节点：指令解析 → 风险评估 → 工具执行。

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 messages、tool_calls、tool_results、final_response 的状态更新
    """
    user_query = extract_latest_query(state)
    logger.info(f"Device Agent 开始处理 | query={user_query[:50]}...")

    # ── 阶段 1: 解析自然语言指令为工具调用 ──
    parsed_calls = _parse_device_intent(user_query, state)

    if not parsed_calls:
        response_text = (
            "抱歉，我没有理解您的指令。您可以试着说具体一些，比如「明天早上7点提醒我吃药」。"
        )
        return {
            "messages": [AIMessage(content=response_text)],
            "tool_calls": [],
            "tool_results": [],
            "final_response": response_text,
        }

    # ── 阶段 2 & 3: 逐个执行工具调用 ──
    executor = _get_executor()
    all_results: list[dict] = []
    response_parts: list[str] = []

    for call in parsed_calls:
        tool_name = call["tool_name"]
        arguments = call["arguments"]

        result = executor.execute(tool_name, arguments)
        all_results.append(result.to_dict())

        if result.needs_confirmation:
            # 需要用户确认的操作，将确认提示加入回复
            confirm_msg = result.result.get("confirmation_message", "请确认是否执行此操作？")
            response_parts.append(confirm_msg)
        elif result.success:
            # 成功执行，生成友好的回复
            result_msg = result.result.get("message", f"{tool_name} 执行成功")
            response_parts.append(f"✅ {result_msg}")
        else:
            # 执行失败
            response_parts.append(f"❌ {tool_name} 执行失败: {result.error}")

    response_text = "\n".join(response_parts)

    logger.info(
        f"Device Agent 完成 | 工具调用数={len(parsed_calls)} | "
        f"成功数={sum(1 for r in all_results if r['success'])}"
    )

    return {
        "messages": [AIMessage(content=response_text)],
        "tool_calls": [c for c in parsed_calls],
        "tool_results": all_results,
        "final_response": response_text,
    }


# ────────────────────────────────────────────────────────────
# 指令解析
# ────────────────────────────────────────────────────────────


def _parse_device_intent(user_query: str, state: AgentState) -> list[dict]:
    """
    调用 LLM 将自然语言指令解析为结构化工具调用。

    Args:
        user_query: 用户原始指令
        state: AgentState（用于提取用户画像）

    Returns:
        list[dict]: 解析出的工具调用列表，每项包含 tool_name 和 arguments
    """
    messages = prompt_manager.build_prompt(
        PROMPT_TEMPLATE,
        user_query=user_query,
        user_profile_summary=build_profile_summary(state.get("user_profile", {})),
        current_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    try:
        response = call_llm_parse(
            model=DEVICE_LLM_MODEL,
            messages=messages,
            response_format=DeviceParseOutput,
            temperature=DEFAULT_TEMPERATURE,
        )
        if response is None:
            logger.warning("Device 指令解析无结果")
            return []
        result = [
            {"tool_name": tc.tool_name, "arguments": tc.arguments} for tc in response.tool_calls
        ]
        logger.info(f"指令解析完成 | 工具调用数={len(result)}")
        return result

    except Exception as e:
        logger.error(f"Device 指令解析 LLM 调用失败: {e}")
        return []
