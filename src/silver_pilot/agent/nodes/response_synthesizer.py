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
import subprocess

from langchain_core.messages import AIMessage
from openai import OpenAI

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

# ================= 本地 Synthesizer 配置（vLLM/OpenAI 兼容） =================
USE_LOCAL_SYNTHESIZER: bool = bool(getattr(config, "RESPONSE_SYNTHESIZER_USE_LOCAL", True))
LOCAL_SYNTHESIZER_MODEL: str = getattr(
    config,
    "RESPONSE_SYNTHESIZER_LOCAL_MODEL",
    "elderly-care-assistant",
)
LOCAL_SYNTHESIZER_PORT: int = int(getattr(config, "RESPONSE_SYNTHESIZER_LOCAL_PORT", 8000))
LOCAL_SYNTHESIZER_BASE_URL: str = str(
    getattr(config, "RESPONSE_SYNTHESIZER_LOCAL_BASE_URL", "")
).strip()
LOCAL_SYNTHESIZER_WSL_DISTRO: str = getattr(
    config, "RESPONSE_SYNTHESIZER_WSL_DISTRO", "Ubuntu"
)
LOCAL_SYNTHESIZER_API_KEY: str = getattr(
    config, "RESPONSE_SYNTHESIZER_LOCAL_API_KEY", "dummy"
)
LOCAL_SYNTHESIZER_TIMEOUT: float = float(
    getattr(config, "RESPONSE_SYNTHESIZER_LOCAL_TIMEOUT", 15)
)

# 本地路径失效后是否继续尝试。默认 true，避免 WSL 重启后需要重启主进程。
RETRY_LOCAL_AFTER_FAILURE: bool = bool(
    getattr(config, "RESPONSE_SYNTHESIZER_RETRY_LOCAL_AFTER_FAILURE", True)
)

_local_client: OpenAI | None = None
_local_available: bool | None = None
_local_base_url: str | None = None

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
        4. 将结果写入 final_response（供 output_guard 消费）

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 final_response 的状态更新
    """
    messages = state.get("messages", [])
    user_query, ai_messages = extract_ai_messages_after_last_human(messages)

    logger.info(
        f"Response Synthesizer 开始 | 用户查询长度={len(user_query)} | AI 回复数={len(ai_messages)}"
    )

    # 无内容：兜底
    if not ai_messages:
        logger.warning("无 AI 回复可综合，返回空")
        return {"final_response": ""}

    # 单条回复：直接透传
    if len(ai_messages) == 1:
        logger.info("仅一条 AI 回复，跳过 LLM 综合，直接透传")
        return {"final_response": message_to_text(ai_messages[0])}

    # 多条回复：LLM 综合
    synthesized = _synthesize_responses(user_query, ai_messages)

    logger.info(f"Response Synthesizer 完成 | 综合回复长度={len(synthesized)}")
    return {"final_response": synthesized}


def initialize_synthesizer_backend() -> bool:
    """
    启动阶段探测本地 Synthesizer 可用性。

    Returns:
        bool: 本地模型是否可用。
    """
    if not USE_LOCAL_SYNTHESIZER:
        logger.info("本地 Synthesizer 已关闭，使用 API 模式")
        return False

    available = _ensure_local_client(force_retry=True)
    if available:
        logger.info(
            f"本地 Synthesizer 就绪 | model={LOCAL_SYNTHESIZER_MODEL} | base_url={_local_base_url}"
        )
    else:
        logger.warning(
            "本地 Synthesizer 不可用，运行时将自动降级到 API"
        )
    return available


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

    # 优先走本地 vLLM（Qwen3.5_0.8B），失败时自动降级到 API。
    local_result = _call_local_synthesizer(messages)
    if local_result:
        return local_result

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
        logger.error(f"API 综合调用失败: {e}，回退到简单拼接")

    # 降级：简单拼接
    return messages_to_text(ai_messages)


def _call_local_synthesizer(messages: list[dict[str, str]]) -> str | None:
    """优先调用本地 OpenAI 兼容服务（vLLM）。"""
    if not USE_LOCAL_SYNTHESIZER:
        return None

    if not _ensure_local_client(force_retry=False):
        return None

    try:
        assert _local_client is not None  # for type checker
        response = _local_client.chat.completions.create(
            model=LOCAL_SYNTHESIZER_MODEL,
            messages=messages,
            temperature=SYNTHESIZER_TEMPERATURE,
            max_tokens=SYNTHESIZER_MAX_TOKENS,
        )
        content = response.choices[0].message.content
        if content:
            return content
        logger.warning("本地 Synthesizer 返回空内容，降级到 API")
    except Exception as e:
        logger.warning(f"本地 Synthesizer 调用失败: {e}，降级到 API")
        _mark_local_unavailable()

    return None


def _ensure_local_client(*, force_retry: bool) -> bool:
    global _local_client, _local_available, _local_base_url

    if _local_available is True and _local_client is not None:
        return True

    if _local_available is False and not (force_retry or RETRY_LOCAL_AFTER_FAILURE):
        return False

    for base_url in _iter_local_base_urls():
        try:
            client = OpenAI(
                api_key=LOCAL_SYNTHESIZER_API_KEY,
                base_url=base_url,
                timeout=LOCAL_SYNTHESIZER_TIMEOUT,
            )
            # 轻量探活：vLLM OpenAI 兼容接口支持 /models
            client.models.list()

            _local_client = client
            _local_available = True
            _local_base_url = base_url
            logger.info(f"本地 Synthesizer 可用 | base_url={base_url}")
            return True
        except Exception as e:
            logger.debug(f"探测本地 Synthesizer 失败 | base_url={base_url} | error={e}")

    _mark_local_unavailable()
    return False


def _mark_local_unavailable() -> None:
    global _local_client, _local_available
    _local_client = None
    _local_available = False


def _iter_local_base_urls() -> list[str]:
    candidates: list[str] = []
    if LOCAL_SYNTHESIZER_BASE_URL:
        candidates.append(_normalize_base_url(LOCAL_SYNTHESIZER_BASE_URL))

    candidates.append(f"http://127.0.0.1:{LOCAL_SYNTHESIZER_PORT}/v1")
    candidates.append(f"http://localhost:{LOCAL_SYNTHESIZER_PORT}/v1")

    wsl_ip = _get_wsl_ip()
    if wsl_ip:
        candidates.append(f"http://{wsl_ip}:{LOCAL_SYNTHESIZER_PORT}/v1")

    # 去重并保持顺序
    unique_candidates: list[str] = []
    for url in candidates:
        if url not in unique_candidates:
            unique_candidates.append(url)
    return unique_candidates


def _normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _get_wsl_ip() -> str | None:
    """在 Windows 端尝试读取 WSL IP，便于访问 Ubuntu 中的 vLLM 服务。"""
    try:
        result = subprocess.run(
            ["wsl", "-d", LOCAL_SYNTHESIZER_WSL_DISTRO, "hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        raw = (result.stdout or "").strip()
        if not raw:
            return None
        return raw.split()[0]
    except Exception:
        return None
