"""
模块名称：state
功能描述：Agent 系统的核心状态定义。基于 TypedDict 定义 AgentState，作为
         LangGraph 状态图中所有节点共享的数据契约。涵盖对话核心、感知层输出、
         规划层、知识层、记忆层、安全层、执行层及最终输出等全维度字段。
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from silver_pilot.config import config

# ────────────────────────────────────────────────────────────
# 常量定义
# ────────────────────────────────────────────────────────────

MAX_SUPERVISOR_LOOPS: int = config.MAX_SUPERVISOR_LOOPS
"""Supervisor 最大循环次数，防止无限循环。"""

MAX_RETRY_PER_AGENT: int = config.MAX_RETRY_PER_AGENT
"""单个子 Agent 最大重试次数。"""

HALLUCINATION_THRESHOLD: float = config.HALLUCINATION_THRESHOLD
"""幻觉检测分数阈值，超过此值触发 Fallback。"""

COMPRESS_THRESHOLD: int = config.COMPRESS_THRESHOLD
"""消息数超过此阈值时触发对话摘要压缩。"""

KEEP_RECENT_TURNS: int = config.KEEP_RECENT_TURNS
"""摘要压缩时保留的最近消息轮数。"""

SUMMARY_MAX_TOKENS: int = config.SUMMARY_MAX_TOKENS
"""摘要最大长度。"""
# ────────────────────────────────────────────────────────────
# 意图类型枚举
# ────────────────────────────────────────────────────────────

INTENT_TYPES = Literal[
    "EMERGENCY",
    "MEDICAL_QUERY",
    "DEVICE_CONTROL",
    "CHITCHAT",
]

RISK_LEVELS = Literal["low", "medium", "high", "critical"]

INPUT_MODALITIES = Literal["text", "voice", "image", "multimodal"]


# ────────────────────────────────────────────────────────────
# 核心状态定义
# ────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """
    Agent 系统的全局共享状态。

    每个 LangGraph 节点通过读写此状态实现信息传递。
    ``messages`` 字段使用 ``add_messages`` 注解实现自动追加语义，
    其余字段为普通覆盖式更新。

    字段分组：
        - 对话核心：消息历史 + 对话摘要
        - 感知层输出：情感标签、图像上下文、输入模态
        - 规划层：意图队列、当前 Agent、风险等级、循环计数
        - 知识层：RAG 上下文、链接实体
        - 记忆层：用户画像
        - 安全层：幻觉分数、安全标记
        - 执行层：工具调用与结果、重试计数
        - 输出：最终响应文本
    """

    # ── 对话核心 ──
    messages: Annotated[list[AnyMessage], add_messages]
    """LangGraph 管理的消息列表，使用 add_messages 注解自动追加。"""

    conversation_summary: str
    """当消息历史过长时，由 ConversationSummarizer 生成的压缩摘要。"""

    # ── 感知层输出 ──
    user_emotion: str
    """ASR 输出的情感标签，如 "NEUTRAL", "SAD", "ANGRY" 等。"""

    current_audio_context: str
    """ASR 的输出结果，注入查询处理阶段。"""

    current_image_context: str
    """Qwen 的 OCR/图像描述结果，注入查询处理阶段。"""

    input_modality: INPUT_MODALITIES
    """当前输入的模态类型。"""

    # ── 规划层 ──
    pending_intents: list[dict]
    """待处理的意图队列，每项包含 type、sub_query、priority 字段。"""

    current_agent: str
    """当前正在执行的子 Agent 名称。"""

    risk_level: RISK_LEVELS
    """当前操作的风险等级。"""

    loop_count: int
    """Supervisor 已执行的循环次数，达到 MAX_SUPERVISOR_LOOPS 时强制终止。"""

    total_turns: int
    """累计对话轮次（只增不减，不受摘要压缩影响）。"""

    last_profile_extract_at: int
    """上次用户画像提取的对话轮次。"""

    # ── 知识层 ──
    rag_context: str
    """RAGPipeline 返回的检索上下文。"""

    linked_entities: list[dict]
    """实体链接阶段的结果列表。"""

    # ── 记忆层 ──
    user_profile: dict
    """从长期记忆加载的用户画像，包含慢性病、过敏、用药等信息。"""

    # ── 安全层 ──
    hallucination_score: float
    """幻觉检测分数（0.0 ~ 1.0），越高越可能是幻觉。"""

    safety_flags: list[str]
    """触发的安全规则列表，如 ["political_sensitive", "medical_disclaimer"]。"""

    # ── 执行层 ──
    tool_calls: list[dict]
    """待执行的工具调用列表，每项包含 tool_name 和 arguments。"""

    tool_results: list[dict]
    """工具执行结果列表。"""

    retry_count: int
    """当前子 Agent 的重试次数。"""

    # ── 输出 ──
    final_response: str
    """经过安全校验后的最终输出文本。"""


# ────────────────────────────────────────────────────────────
# 状态初始化工厂
# ────────────────────────────────────────────────────────────


def create_initial_state() -> dict:
    """
    创建 AgentState 的初始值字典。

    LangGraph 在首次调用时需要提供初始状态，此工厂函数
    为所有非 ``messages`` 字段提供合理的默认值。

    Returns:
        dict: 可直接传入 ``graph.invoke()`` 的初始状态字典
    """
    return {
        "messages": [],
        "conversation_summary": "",
        "user_emotion": "NEUTRAL",
        "current_image_context": "",
        "input_modality": "text",
        "pending_intents": [],
        "current_agent": "",
        "risk_level": "low",
        "loop_count": 0,
        "total_turns": 0,
        "last_profile_extract_at": 0,
        "rag_context": "",
        "linked_entities": [],
        "user_profile": {},
        "hallucination_score": 0.0,
        "safety_flags": [],
        "tool_calls": [],
        "tool_results": [],
        "retry_count": 0,
        "final_response": "",
    }
