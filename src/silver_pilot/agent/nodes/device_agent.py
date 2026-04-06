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
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm_parse
from ..state import AgentState
from ..tools import ToolExecutionResult, ToolExecutor
from .helpers import build_profile_summary, extract_latest_query

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "device_agent")

# ================= 默认配置 =================
DEVICE_LLM_MODEL: str = config.DEVICE_AGENT_MODEL
PROMPT_TEMPLATE: str = "agent/device_parse"
DEFAULT_TEMPERATURE = config.DEVICE_AGENT_TEMPERATURE

# 用户确认的肯定回复（不区分大小写）
CONFIRM_KEYWORDS: set[str] = {"确认", "好的", "是", "yes", "ok", "好", "执行", "可以"}
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


def set_executor(executor: ToolExecutor) -> None:
    """设置 ToolExecutor 单例（由 bootstrap 调用）。"""
    global _executor
    _executor = executor


# ────────────────────────────────────────────────────────────
# Device Agent 节点
# ────────────────────────────────────────────────────────────


def device_agent_node(state: AgentState) -> dict:
    """
    Device Agent 节点：指令解析 → 风险评估 → 工具执行。

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 messages、tool_calls、tool_results、sub_response 的状态更新
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
            "sub_response": [response_text],
        }

    # ── 阶段 2 & 3: 逐个执行工具调用 ──
    if _executor is None:
        logger.warning("Device Agent 未初始化，走降级路径")
        return {
            "messages": [
                AIMessage(
                    content="抱歉，我没有理解您的指令。您可以试着说具体一些，比如「明天早上7点提醒我吃药」。"
                )
            ],
            "tool_calls": [],
            "tool_results": [],
            "sub_response": [
                "抱歉，我没有理解您的指令。您可以试着说具体一些，比如「明天早上7点提醒我吃药」。"
            ],
        }

    all_results: list[dict] = []
    response_parts: list[str] = []

    for call in parsed_calls:
        result = _execute_with_confirmation(_executor, call["tool_name"], call["arguments"])
        all_results.append(result.to_dict())

        if result.success:
            response_parts.append(f"{result.result.get('message', '执行成功')}")
        else:
            response_parts.append(f"{call['tool_name']}: {result.error}")

    response_text = "\n".join(response_parts)

    logger.info(
        f"Device Agent 完成 | 工具调用数={len(parsed_calls)} | "
        f"成功数={sum(1 for r in all_results if r['success'])}"
    )

    return {
        "messages": [AIMessage(content=response_text)],
        "tool_calls": [c for c in parsed_calls],
        "tool_results": all_results,
        "sub_response": [response_text],
    }


# ────────────────────────────────────────────────────────────
# HITL 确认逻辑
# ────────────────────────────────────────────────────────────


def _execute_with_confirmation(
    executor: ToolExecutor, tool_name: str, arguments: dict
) -> ToolExecutionResult:
    """
    执行工具调用，需要确认时通过 interrupt 暂停图。

    流程：
        1. 先尝试执行（user_confirmed=False）
        2. 如果返回 needs_confirmation → 调用 interrupt() 暂停
        3. 图恢复后拿到用户回复 → 判断是否确认
        4. 确认 → 带 user_confirmed=True 重新执行
        5. 拒绝 → 返回取消结果
    """
    # 第一次执行（未确认）
    result = executor.execute(tool_name, arguments, user_confirmed=False)

    if not result.needs_confirmation:
        # 低风险，直接返回执行结果
        return result

    # ── 中/高风险：interrupt 暂停，等待用户确认 ──
    confirmation_message = result.result.get("confirmation_message", "请确认是否执行？")
    logger.info(f"HITL 中断 | tool={tool_name} | risk={result.risk_level}")

    # interrupt() 暂停图执行，将确认信息返回给调用方
    # 调用方用 Command(resume="用户回复") 恢复后，interrupt() 返回 "用户回复"
    user_reply = interrupt(
        {
            "type": "confirmation",
            "message": confirmation_message,
            "tool_name": tool_name,
            "risk_level": result.risk_level,
        }
    )

    # ── 图恢复，处理用户回复 ──
    user_reply_str = str(user_reply).strip().lower()
    confirmed = user_reply_str in CONFIRM_KEYWORDS

    if confirmed:
        logger.info(f"用户确认执行 | tool={tool_name}")
        return executor.execute(tool_name, arguments, user_confirmed=True)
    else:
        logger.info(f"用户拒绝执行 | tool={tool_name} | reply={user_reply}")
        return ToolExecutionResult(
            tool_name=tool_name,
            success=False,
            error="用户取消操作",
            risk_level=result.risk_level,
        )


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
