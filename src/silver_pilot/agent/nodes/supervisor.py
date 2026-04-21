"""
模块名称：supervisor
功能描述：Supervisor 循环编排节点，Agent 系统的"大脑"。负责意图分类、复合意图拆解、
         子 Agent 路由调度，以及循环终止判断。通过 LLM 结构化输出实现意图解析，
         支持“组间并行，组内串行”的分批调度。

核心职责：
    1. 首次进入时调用 LLM 完成意图分类与优先级排序
    2. 识别是否存在意图依赖（显式 depends_on + 关键词兜底）
    3. 无依赖：沿用 Send 并行；有依赖：按 depends_on 拆分 stage 批次
    4. 子 Agent 执行完毕后回到 Supervisor 继续调度，直到 done
"""

import re
from pathlib import Path

from langgraph.types import Send
from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm_parse
from ..state import MAX_SUPERVISOR_LOOPS, AgentState
from .helpers import build_profile_summary, extract_latest_query, get_conversation_context

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

# 依赖关系关键词（中英文）
DEPENDENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(先|首先).{0,20}(再|然后|接着)"),
    re.compile(r"(完成后|之后|以后|成功后)"),
    re.compile(r"(查完|确认完|处理完).{0,12}(再|然后|接着)"),
    re.compile(r"(根据|按).{0,16}(结果|建议|回复|上一步).{0,12}(再|然后)"),
    re.compile(r"(first|then|after|once|before)", re.IGNORECASE),
]

# ────────────────────────────────────────────────────────────
# Pydantic 结构化输出 Schema
# ────────────────────────────────────────────────────────────


class IntentItem(BaseModel):
    """单个意图的结构化表示。"""

    type: str = Field(description="意图类型: EMERGENCY / MEDICAL_QUERY / DEVICE_CONTROL / CHITCHAT")
    sub_query: str = Field(description="该意图对应的具体子查询")
    priority: int = Field(description="优先级数值，越小越高")
    depends_on: list[int] = Field(
        default_factory=list,
        description="可选依赖项，引用依赖意图的 priority 列表",
    )


class SupervisorOutput(BaseModel):
    """Supervisor LLM 的结构化输出。"""

    intents: list[IntentItem] = Field(description="识别到的意图列表")
    risk_level: str = Field(default="low", description="整体风险等级")


# ────────────────────────────────────────────────────────────
# Supervisor 节点
# ────────────────────────────────────────────────────────────


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor 编排节点：首次分类建计划，后续持续调度。

    执行逻辑：
        1. loop_count == 0：调用 LLM 分类并生成首批分发策略
        2. loop_count > 0：从 pending_intents 取下一批继续调度
        3. 无剩余任务：current_agent="done"，进入回复综合

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 dispatch_intents、pending_intents、current_agent、loop_count 的状态更新
    """
    loop_count = state.get("loop_count", 0)

    # ── 循环保护 ──
    if loop_count >= MAX_SUPERVISOR_LOOPS:
        logger.warning(f"达到最大循环次数 {MAX_SUPERVISOR_LOOPS}，强制终止")
        return {
            "dispatch_intents": [],
            "current_agent": "done",
            "current_sub_query": "",
        }

    next_loop = loop_count + 1

    # 首次进入：分类并下发首批任务
    if loop_count == 0:
        return _classify_and_dispatch(state, next_loop=next_loop)

    # 子 Agent 回流后：继续调度剩余任务
    return _dispatch_next_batch(state, next_loop=next_loop)


# ────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────


def _classify_and_dispatch(state: AgentState, *, next_loop: int) -> dict:
    """首次调用 LLM 分类，并生成首批调度策略。"""
    user_query = extract_latest_query(state)
    total_turns = state.get("total_turns", 0) + 1

    if not user_query:
        logger.warning("无法提取用户查询，降级为闲聊")
        return _build_fallback_chat_state(
            user_query="",
            next_loop=next_loop,
            total_turns=total_turns,
        )

    logger.info(f"意图分类开始 | user_query={user_query}")
    # 构建 Prompt 并调用 LLM
    messages = prompt_manager.build_prompt(
        PROMPT_TEMPLATE,
        user_query=user_query,
        current_audio_context=state.get("current_audio_context", ""),
        user_emotion=state.get("user_emotion", "NEUTRAL"),
        current_image_context=state.get("current_image_context", ""),
        conversation_summary=get_conversation_context(state.get("messages", [])),
        user_profile_summary=build_profile_summary(state.get("user_profile", {})),
    )

    parsed = call_llm_parse(SUPERVISOR_LLM_MODEL, messages, SupervisorOutput)
    if parsed is None or not parsed.intents:
        logger.warning("意图分类失败或为空，降级为闲聊")
        return _build_fallback_chat_state(
            user_query=user_query,
            next_loop=next_loop,
            total_turns=total_turns,
        )

    # 按优先级排序
    sorted_intents = _normalize_and_sort_intents(parsed.intents, user_query)

    # Emergency 短路：仅处理紧急意图
    emergency_intent = next((i for i in sorted_intents if i.get("type") == "EMERGENCY"), None)
    if emergency_intent:
        logger.warning("检测到 EMERGENCY，短路到 emergency_agent")
        return {
            "pending_intents": [],
            "dispatch_intents": [emergency_intent],
            "completed_intent_priorities": [],
            "current_agent": "emergency",
            "current_sub_query": emergency_intent.get("sub_query", user_query),
            "risk_level": "critical",
            "loop_count": next_loop,
            "retry_count": 0,
            "total_turns": total_turns,
        }

    if len(sorted_intents) == 1:
        only = sorted_intents[0]
        only_agent = INTENT_TO_AGENT.get(only["type"], "chat")
        logger.info(f"单意图路由 | type={only['type']} | agent={only_agent}")
        return {
            "pending_intents": [],
            "dispatch_intents": [only],
            "completed_intent_priorities": [],
            "current_agent": only_agent,
            "current_sub_query": only.get("sub_query", ""),
            "risk_level": parsed.risk_level,
            "loop_count": next_loop,
            "retry_count": 0,
            "total_turns": total_turns,
        }

    if _should_use_dependency_dispatch(user_query, sorted_intents):
        sorted_intents = _ensure_dependency_edges(user_query, sorted_intents)
        logger.info(f"检测到依赖关系，启用 stage 调度 | count={len(sorted_intents)}")
        ready_batch, remain = _select_ready_batch(sorted_intents, completed_priorities=set())
        batch_agent, batch_sub_query = _resolve_batch_dispatch(ready_batch)
        return {
            "pending_intents": remain,
            "dispatch_intents": ready_batch,
            "completed_intent_priorities": [],
            "current_agent": batch_agent,
            "current_sub_query": batch_sub_query,
            "risk_level": parsed.risk_level,
            "loop_count": next_loop,
            "retry_count": 0,
            "total_turns": total_turns,
        }

    logger.info(f"多意图并行路由 | count={len(sorted_intents)}")
    logger.info(
        f"意图分类完成 | types={[i['type'] for i in sorted_intents]} | risk={parsed.risk_level}"
    )

    return {
        "pending_intents": [],
        "dispatch_intents": sorted_intents,
        "completed_intent_priorities": [],
        "current_agent": "parallel",
        "current_sub_query": "",
        "risk_level": parsed.risk_level,
        "loop_count": next_loop,
        "retry_count": 0,
        "total_turns": total_turns,
    }


def _dispatch_next_batch(state: AgentState, *, next_loop: int) -> dict:
    """子 Agent 回流后，按依赖约束推进下一 stage。"""
    completed_set = set(state.get("completed_intent_priorities", []))
    for intent in state.get("dispatch_intents", []):
        priority = intent.get("priority")
        if isinstance(priority, int):
            completed_set.add(priority)

    completed_list = sorted(completed_set)

    pending = list(state.get("pending_intents", []))
    if not pending:
        logger.info("无剩余意图，进入 response_synthesizer")
        return {
            "dispatch_intents": [],
            "completed_intent_priorities": completed_list,
            "current_agent": "done",
            "current_sub_query": "",
            "loop_count": next_loop,
        }

    next_batch, remain_intents = _select_ready_batch(pending, completed_priorities=completed_set)
    next_agent, next_sub_query = _resolve_batch_dispatch(next_batch)

    logger.info(
        f"推进下一批次 | batch_size={len(next_batch)} | "
        f"remaining={len(remain_intents)} | agent={next_agent}"
    )
    return {
        "pending_intents": remain_intents,
        "dispatch_intents": next_batch,
        "completed_intent_priorities": completed_list,
        "current_agent": next_agent,
        "current_sub_query": next_sub_query,
        "loop_count": next_loop,
        "retry_count": 0,
    }


def _build_fallback_chat_state(user_query: str, *, next_loop: int, total_turns: int) -> dict:
    """分类失败时降级到 chat_agent。"""
    fallback_intent = {
        "type": "CHITCHAT",
        "sub_query": user_query,
        "priority": 9,
        "depends_on": [],
    }
    return {
        "pending_intents": [],
        "dispatch_intents": [fallback_intent],
        "completed_intent_priorities": [],
        "current_agent": "chat",
        "current_sub_query": user_query,
        "risk_level": "low",
        "loop_count": next_loop,
        "retry_count": 0,
        "total_turns": total_turns,
    }


def _normalize_and_sort_intents(intents: list[IntentItem], user_query: str) -> list[dict]:
    """规范化 LLM 意图输出并按 priority 排序。"""
    normalized: list[dict] = []

    for intent in intents:
        intent_type = intent.type if intent.type in INTENT_TO_AGENT else "CHITCHAT"
        sub_query = intent.sub_query.strip() if intent.sub_query else ""
        if not sub_query:
            sub_query = user_query

        normalized.append(
            {
                "type": intent_type,
                "sub_query": sub_query,
                "priority": intent.priority,
                "depends_on": list(intent.depends_on),
            }
        )

    normalized.sort(key=lambda x: x.get("priority", 999))
    return normalized


def _should_use_dependency_dispatch(user_query: str, intents: list[dict]) -> bool:
    """判断复合意图是否应启用依赖 stage 调度。"""
    if len(intents) <= 1:
        return False

    if any(intent.get("depends_on") for intent in intents):
        return True

    if _has_dependency_signal(user_query):
        return True

    sub_query_text = "；".join(str(intent.get("sub_query", "")) for intent in intents)
    return _has_dependency_signal(sub_query_text)


def _ensure_dependency_edges(user_query: str, intents: list[dict]) -> list[dict]:
    """依赖调度模式下，必要时补齐依赖边。"""
    if not intents:
        return intents

    if any(intent.get("depends_on") for intent in intents):
        return intents

    if not _has_dependency_signal(user_query):
        return intents

    logger.info("检测到先后关系但 lacks depends_on，按 priority 注入线性依赖")

    patched_intents: list[dict] = []
    previous_priority: int | None = None
    for intent in intents:
        patched = {**intent}
        if previous_priority is None:
            patched["depends_on"] = []
        else:
            patched["depends_on"] = [previous_priority]

        current_priority = patched.get("priority")
        if isinstance(current_priority, int):
            previous_priority = current_priority

        patched_intents.append(patched)

    return patched_intents


def _select_ready_batch(
    intents: list[dict], *, completed_priorities: set[int]
) -> tuple[list[dict], list[dict]]:
    """按 depends_on 选择当前可执行批次。"""
    ready_batch: list[dict] = []
    blocked_batch: list[dict] = []

    for intent in intents:
        deps = intent.get("depends_on", [])
        if not isinstance(deps, list):
            deps = []

        if all(dep in completed_priorities for dep in deps):
            ready_batch.append(intent)
        else:
            blocked_batch.append(intent)

    if ready_batch:
        return ready_batch, blocked_batch

    # 兜底：防止错误依赖（循环/缺失）导致卡死。
    logger.warning("依赖图无可执行节点，降级按优先级取首个意图")
    return [intents[0]], intents[1:]


def _resolve_batch_dispatch(batch: list[dict]) -> tuple[str, str]:
    """将可执行批次映射为 current_agent 和 current_sub_query。"""
    if not batch:
        return "done", ""

    if len(batch) == 1:
        intent = batch[0]
        return INTENT_TO_AGENT.get(intent.get("type", "CHITCHAT"), "chat"), intent.get(
            "sub_query", ""
        )

    return "parallel", ""


def _has_dependency_signal(text: str) -> bool:
    """基于关键词判断文本是否包含先后依赖信号。"""
    if not text:
        return False
    return any(pattern.search(text) for pattern in DEPENDENCY_PATTERNS)


# ────────────────────────────────────────────────────────────
# 路由函数（供 LangGraph conditional_edges 使用）
# ────────────────────────────────────────────────────────────


def route_by_intent(state: AgentState) -> str | list[Send]:
    """
    条件路由函数：根据 current_agent 决定下一个执行节点。

    Args:
        state: 当前 AgentState

    Returns:
        str | list[Send]: 单意图返回节点名；多意图返回 Send 列表并行分发
    """
    agent = state.get("current_agent", "done")
    if agent == "parallel":
        dispatch_intents = state.get("dispatch_intents", []) or state.get("pending_intents", [])

        # 同类型意图合并为一次分发，避免并行写冲突。
        grouped_queries: dict[str, list[str]] = {
            "medical": [],
            "device": [],
            "chat": [],
        }
        for intent in dispatch_intents:
            agent_key = INTENT_TO_AGENT.get(intent.get("type", "CHITCHAT"), "chat")
            if agent_key == "emergency":
                # emergency 已在 supervisor_node 中短路处理，这里忽略
                continue
            grouped_queries.setdefault(agent_key, [])
            grouped_queries[agent_key].append(intent.get("sub_query", ""))

        sends: list[Send] = []
        active_branches: list[str] = []
        for agent_key, queries in grouped_queries.items():
            if not queries:
                continue
            merged_query = "；".join(q for q in queries if q)
            active_branches.append(agent_key)
            sends.append(
                Send(
                    f"{agent_key}_agent",
                    {
                        "current_sub_query": merged_query,
                    },
                )
            )

        logger.info(f"并行分发 | branches={active_branches}")
        return sends if sends else "done"

    logger.debug(f"路由决策 | current_agent={agent}")
    return agent
