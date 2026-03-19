"""
模块名称：llm
功能描述：Agent 系统的 LLM 统一调用层。提供单例客户端和两种调用模式
         （自由文本生成 / Pydantic 结构化输出），所有节点共享此入口，
         消除各节点重复的客户端初始化和异常处理代码。
"""

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

logger = get_channel_logger(config.LOG_DIR / "agent", "llm")

T = TypeVar("T", bound=BaseModel)

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 800

# ── 单例客户端 ──────────────────────────────────────────────

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """获取 OpenAI 兼容客户端单例（连接 DashScope）。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.QWEN_URL[config.QWEN_REGION],
        )
    return _client


# ── 文本生成 ────────────────────────────────────────────────


def call_llm(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str | None:
    """
    调用 LLM 生成自由文本。

    Args:
        model: 模型名称（如 "qwen-max", "qwen3.5-flash"）
        messages: OpenAI 格式的消息列表
        temperature: 采样温度
        max_tokens: 最大生成 token 数

    Returns:
        str | None: 生成的文本。调用失败时返回 None。
    """
    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM 调用失败 | model={model} | error={e}")
        return None


# ── 结构化输出 ──────────────────────────────────────────────


def call_llm_parse[T: BaseModel](
    model: str,
    messages: list[dict[str, str]],
    response_format: type[T],
    *,
    temperature: float = 0.1,
) -> T | None:
    """
    调用 LLM 并解析为 Pydantic 结构化对象。

    Args:
        model: 模型名称
        messages: OpenAI 格式的消息列表
        response_format: Pydantic BaseModel 子类
        temperature: 采样温度

    Returns:
        T | None: 解析后的 Pydantic 对象。调用或解析失败时返回 None。
    """
    try:
        response = get_client().chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error(
            f"LLM 结构化调用失败 | model={model} | format={response_format.__name__} | error={e}"
        )
        return None
