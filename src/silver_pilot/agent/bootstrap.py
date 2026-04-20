"""
模块名称：bootstrap
功能描述：Agent 系统的统一启动入口。一次性完成所有耗时组件的初始化和依赖注入，
         确保运行时不再有延迟初始化，且同一组件只存在一个实例。

调用时机：
    - 系统启动时调用一次 ``initialize_agent()``
    - 返回编译好的 graph 实例，后续直接用于 invoke

设计原则：
    - 所有组件在此集中创建，不散落在各模块的 _get_xxx() 里
    - 注入完成后各模块的 _get_xxx() 兜底逻辑不再被触发
    - 如果某个组件初始化失败（如 RAGPipeline 依赖的数据库未启动），
      记录警告但不阻断启动——对应功能会在运行时走降级路径
"""

from langgraph.graph.state import CompiledStateGraph

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

from .graph import build_agent_graph
from .memory.summarizer import ConversationSummarizer
from .memory.user_profile import ProfileManagerProtocol, UserProfileManager
from .nodes.device_agent import set_executor
from .nodes.medical_agent import set_pipeline
from .nodes.memory_writer import set_profile_manager
from .nodes.output_guard import set_summarizer
from .nodes.response_synthesizer import initialize_synthesizer_backend
from .tools.executor import ToolExecutor, set_mcp_client

logger = get_channel_logger(config.LOG_DIR / "agent", "bootstrap")


def initialize_agent(
    *,
    checkpointer: object | None = None,
    skip_rag: bool = False,
    profile_manager: ProfileManagerProtocol | None = None,
) -> CompiledStateGraph:
    """
    Agent 系统统一启动入口。

    Args:
        checkpointer: LangGraph Checkpointer。为 None 时用 MemorySaver（内存）。
        skip_rag: 是否跳过 RAGPipeline 初始化。
        profile_manager: 外部传入的画像管理器实例（需实现 ProfileManagerProtocol）。
                        可以是 UserProfileManager (SQLite) 或 RedisStore。
                        为 None 时内部创建 UserProfileManager (SQLite)。

    Returns:
        编译后的 LangGraph 图实例
    """
    logger.info("=" * 50)
    logger.info("Agent 系统启动")
    logger.info("=" * 50)

    # ── 1. RAGPipeline ──
    if skip_rag:
        logger.warning("[1/3] 跳过 RAGPipeline（skip_rag=True），medical_agent 将走降级路径")
    else:
        logger.info("[1/3] 初始化 RAGPipeline...")
        try:
            from silver_pilot.rag.retriever import PipelineConfig, RAGPipeline

            pipeline = RAGPipeline(PipelineConfig())
            pipeline.initialize()
            set_pipeline(pipeline)
            logger.info("[1/3] RAGPipeline 就绪")
        except Exception as e:
            logger.warning(f"[1/3] RAGPipeline 初始化失败: {e}，medical_agent 将走降级路径")

    # ── 2. RAGPipeline 外的组件 ──
    logger.info("[2/3] 初始化 RAGPipeline 外的组件...")

    # UserProfileManager: 优先使用外部传入（如 RedisStore），否则内部创建 SQLite 版
    if profile_manager is None:
        profile_manager = UserProfileManager()
    set_profile_manager(profile_manager)

    executor = ToolExecutor()
    set_executor(executor)

    # MCPClient：初始化失败时记录警告，天气工具自动降级为模拟路径
    try:
        from silver_pilot.tools.MCP.mcp_client import MCPClient

        mcp_client = MCPClient()
        set_mcp_client(mcp_client)
        logger.info("[2/3] MCPClient 就绪（query_weather / weather_forecast 已接入 MCP）")
    except Exception as e:
        logger.warning(
            f"[2/3] MCPClient 初始化失败: {e}，"
            "天气工具将走模拟降级路径"
        )

    summarizer = ConversationSummarizer()
    set_summarizer(summarizer)

    # Response Synthesizer: 优先本地模型，失败时运行时自动降级到 API。
    try:
        local_ready = initialize_synthesizer_backend()
        if local_ready:
            logger.info("[2/3] Response Synthesizer 本地模型就绪")
        else:
            logger.warning("[2/3] Response Synthesizer 本地模型不可用，将降级到 API")
    except Exception as e:
        logger.warning(
            f"[2/3] Response Synthesizer 初始化探测失败: {e}，将降级到 API"
        )

    logger.info("[2/3] 组件就绪")

    # ── 3. 构建图 ──
    logger.info("[3/3] 构建 LangGraph 图...")
    graph = build_agent_graph(checkpointer=checkpointer)
    logger.info("[3/3] 图构建完成")

    logger.info("=" * 50)
    logger.info("Agent 系统启动完成")
    logger.info("=" * 50)

    return graph
