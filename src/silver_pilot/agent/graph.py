"""
模块名称：graph
功能描述：LangGraph 状态图拓扑定义与编译。将所有节点和边组装为完整的 Agent 执行图，
         支持 Supervisor 循环编排、条件路由、状态持久化（Checkpointer）等核心能力。

图拓扑结构：
    perception_router → supervisor → {medical, device, chat, emergency}
                               ↘ (Send 并行) ↙
                         {medical, device, chat, emergency} → supervisor
    supervisor(done) → response_synthesizer → output_guard → memory_writer → END

    Supervisor 通过 conditional_edges + Send 实现动态路由：
    - 单意图: route_by_intent 返回 str，路由到对应子 Agent
    - 多意图无依赖: route_by_intent 返回 list[Send]，并行分发
    - 多意图有依赖: 子 Agent 执行后回流 supervisor，按 stage 批次推进
    - 所有任务完成后 current_agent=done，进入 response_synthesizer 聚合
"""

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

from .nodes import (
    chat_agent_node,
    device_agent_node,
    emergency_agent_node,
    medical_agent_node,
    memory_writer_node,
    output_guard_node,
    perception_router_node,
    response_synthesizer_node,
    route_by_intent,
    supervisor_node,
)
from .state import AgentState

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "graph")


def build_agent_graph(checkpointer: object | None = None) -> CompiledStateGraph:
    """
    构建并编译 Agent 状态图。

    Args:
        checkpointer: LangGraph Checkpointer 实例，用于状态持久化。
                      为 None 时使用内存 Checkpointer（开发阶段）。
                      生产环境可传入 SqliteSaver 或 RedisSaver。

    Returns:
        编译后的 LangGraph CompiledStateGraph 实例（可直接调用 invoke / stream）
    """
    logger.info("开始构建 Agent 状态图...")

    graph = StateGraph(AgentState)

    # ── 注册节点 ──
    graph.add_node("perception_router", perception_router_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("medical_agent", medical_agent_node)
    graph.add_node("device_agent", device_agent_node)
    graph.add_node("chat_agent", chat_agent_node)
    graph.add_node("emergency_agent", emergency_agent_node)
    graph.add_node("response_synthesizer", response_synthesizer_node)
    graph.add_node("output_guard", output_guard_node)
    graph.add_node("memory_writer", memory_writer_node)

    # ── 入口 ──
    graph.set_entry_point("perception_router")

    # ── 感知层 → Supervisor ──
    graph.add_edge("perception_router", "supervisor")

    # ── Supervisor 条件路由 ──
    # route_by_intent 根据 state["current_agent"] 返回目标节点名
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "medical": "medical_agent",
            "device": "device_agent",
            "chat": "chat_agent",
            "emergency": "emergency_agent",
            "done": "response_synthesizer",
        },
    )

    # ── 子 Agent 完成后回流 Supervisor，持续调度直到 done ──
    for agent_node in ["medical_agent", "device_agent", "chat_agent", "emergency_agent"]:
        graph.add_edge(agent_node, "supervisor")

    # ── 输出链路 ──
    graph.add_edge("response_synthesizer", "output_guard")
    graph.add_edge("output_guard", "memory_writer")
    graph.add_edge("memory_writer", END)

    # ── 编译 ──
    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.info("使用内存 Checkpointer（开发模式）")

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "Agent 状态图构建完成 | "
        "节点数=9 | "
        "拓扑: perception → supervisor → agents(支持并行) → synthesizer → guard → memory → END"
    )

    return compiled
