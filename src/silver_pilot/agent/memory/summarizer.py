"""
模块名称：summarizer
功能描述：对话摘要压缩器，当消息历史超过阈值时触发 LLM 摘要压缩，
         将早期对话压缩为简洁摘要，保留最近数轮原始消息，有效控制
         上下文长度并保持对话连贯性。
"""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

from ..state import AgentState

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "summarizer")

# ================= 默认配置 =================
DEFAULT_LLM_MODEL: str = config.MEMORY_SUMMARIZE_MODEL
DEFAULT_REGION: str = config.QWEN_REGION


class ConversationSummarizer:
    """
    对话摘要压缩器。

    当 ``messages`` 长度超过 ``max_turns_before_compress`` 时，将前 N-K 轮
    消息压缩为一段摘要文本，仅保留最近 K 轮原始消息。摘要以 SystemMessage
    的形式注入消息列表头部，确保后续 LLM 调用能感知历史上下文。

    使用示例::

        summarizer = ConversationSummarizer()
        if summarizer.should_compress(state):
            state = summarizer.compress(state)
    """

    MAX_TURNS_BEFORE_COMPRESS: int = config.SUMMARY_MAX_TURNS_BEFORE_COMPRESS
    """触发压缩的消息数阈值。"""

    KEEP_RECENT_TURNS: int = config.SUMMARY_KEEP_RECENT_TURNS
    """压缩时保留的最近消息轮数。"""

    SUMMARY_MAX_TOKENS: int = config.SUMMARY_MAX_TOKENS
    """摘要最大长度。"""

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        max_turns: int = MAX_TURNS_BEFORE_COMPRESS,
        keep_recent: int = KEEP_RECENT_TURNS,
        summary_max_tokens: int = SUMMARY_MAX_TOKENS,
    ) -> None:
        self.llm_model = llm_model
        self.max_turns = max_turns
        self.keep_recent = keep_recent
        self.summary_max_tokens = summary_max_tokens
        self.client = OpenAI(
            api_key=api_key or config.DASHSCOPE_API_KEY,
            base_url=config.QWEN_URL[DEFAULT_REGION],
        )
        logger.info(
            f"ConversationSummarizer 初始化完成 | max_turns={max_turns} | keep_recent={keep_recent}"
        )

    def should_compress(self, state: AgentState) -> bool:
        """
        判断当前对话是否需要压缩。

        Args:
            state: 当前 AgentState

        Returns:
            bool: 消息数超过阈值时返回 True
        """
        return len(state["messages"]) >= self.max_turns

    def compress(self, state: AgentState) -> dict:
        """
        执行对话摘要压缩。

        将早期消息交给 LLM 生成摘要，然后用摘要 SystemMessage + 最近 K 轮
        消息替换原始消息列表。

        Args:
            state: 当前 AgentState

        Returns:
            dict: 包含更新后的 messages 和 conversation_summary 的部分状态
        """
        messages = state["messages"]
        old_messages = messages[: -self.keep_recent]
        recent_messages = messages[-self.keep_recent :]

        logger.info(
            f"开始对话摘要压缩 | 总消息数={len(messages)} | "
            f"压缩消息数={len(old_messages)} | 保留消息数={len(recent_messages)}"
        )

        summary = self._llm_summarize(old_messages)

        logger.info(f"摘要压缩完成 | 摘要长度={len(summary)} 字符")

        return {
            "conversation_summary": summary,
            "messages": [SystemMessage(content=f"[对话历史摘要] {summary}")] + recent_messages,
        }

    def _llm_summarize(self, messages: list) -> str:
        """
        调用 LLM 将消息列表压缩为摘要。

        Args:
            messages: 待压缩的消息列表

        Returns:
            str: 压缩后的摘要文本
        """
        # 将消息格式化为对话文本
        conversation_text = self._format_messages(messages)

        prompt = (
            "请将以下对话历史压缩为一段简洁的摘要，保留关键信息：\n"
            "1. 用户提到的健康状况、症状、用药信息\n"
            "2. 用户已完成的操作（设备控制、提醒设置等）\n"
            "3. 重要的对话结论和待办事项\n\n"
            f"对话历史：\n{conversation_text}\n\n"
            "请直接输出摘要，不超过 200 字。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=self.summary_max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"摘要压缩 LLM 调用失败: {e}")
            return f"[自动摘要失败，保留最近 {self.keep_recent} 轮对话]"

    @staticmethod
    def _format_messages(messages: list) -> str:
        """将 LangChain Message 列表格式化为纯文本对话。"""
        lines: list[str] = []
        for msg in messages:
            lines.append(f"{msg.type}: {msg.content}")
        return "\n".join(lines)
