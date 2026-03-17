"""
模块名称：context_builder
功能描述：上下文构建器，将重排后的多路检索结果组装为最终的 prompt context。
         支持两种模式：直接拼接（快速）和 LLM 压缩提炼（高质量）。
"""

from openai import OpenAI

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from .models import RetrievalResult, RetrievalSource

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "context_builder")

# ================= Prompt 路径 =================
COMPRESS_PROMPT_PATH = "context_compress"
DEFAULT_LLM_MODEL = config.CONTEXT_BUILDER_MODEL
MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS
DEFAULT_REGION = config.QWEN_REGION

# ================= 来源标签映射 =================
SOURCE_LABELS: dict[RetrievalSource, str] = {
    RetrievalSource.NEO4J_GRAPH: "知识图谱",
    RetrievalSource.MILVUS_QA: "医疗问答库",
    RetrievalSource.MILVUS_KNOWLEDGE: "医学文献",
}


class ContextBuilder:
    """
    上下文构建器：将检索结果组装为生成模型可用的 context 文本。

    支持两种模式：
    - **direct** (默认): 直接拼接，快速低成本，适合结果已经很精准的场景
    - **compress**: 通过 LLM 提炼压缩，去重去噪，适合结果较多或质量参差的场景

    使用示例::

        builder = ContextBuilder(mode="direct")
        context_text = builder.build(ranked_results, user_query="阿司匹林能和华法林一起吃吗")
    """

    def __init__(
        self,
        mode: str = "direct",
        llm_model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        base_url: str = config.QWEN_URL[DEFAULT_REGION],
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ) -> None:
        """
        Args:
            mode: "direct" 或 "compress"
            llm_model: 压缩模式使用的 LLM 模型名
            api_key: DashScope API Key
            base_url: DashScope API Base URL
            max_context_chars: 直接拼接模式的最大字符数限制
        """
        self.mode = mode
        self.max_context_chars = max_context_chars

        if mode == "compress":
            self.client = OpenAI(
                api_key=api_key or config.DASHSCOPE_API_KEY,
                base_url=base_url,
            )
            self.llm_model = llm_model

        logger.info(f"ContextBuilder 初始化完成 | mode={self.mode}")

    def build(
        self,
        ranked_results: list[RetrievalResult],
        user_query: str = "",
    ) -> str:
        """
        将重排后的检索结果组装为 context 文本。

        Args:
            ranked_results: 重排序后的检索结果
            user_query: 用户原始查询（压缩模式需要）

        Returns:
            str: 组装好的 context 文本
        """
        if not ranked_results:
            return "未检索到相关参考资料。"

        if self.mode == "compress" and user_query:
            return self._build_compressed(ranked_results, user_query)
        else:
            return self._build_direct(ranked_results)

    def _build_direct(self, results: list[RetrievalResult]) -> str:
        """
        直接拼接模式：图谱感知的层级化组装。

        输出格式:
            【推理路径】
            1. [推理路径] 阿司匹林通过抑制血小板聚集可能增强华法林的抗凝效果...

            【知识概览】
            2. [社区摘要] 抗凝血药物群组包含华法林、阿司匹林等...

            【确定性事实】
            3. 阿司匹林的禁忌包括活动性消化道溃疡
            4. 华法林的不良反应包括出血

            【医疗问答参考】
            5. 问: 阿司匹林能和抗凝药一起吃吗...

            【医学文献参考】
            6. [阿司匹林说明书] 药物相互作用：...
        """
        # 按 layer 和 source 细分
        reasoning_paths: list[RetrievalResult] = []
        community_summaries: list[RetrievalResult] = []
        local_facts: list[RetrievalResult] = []
        qa_results: list[RetrievalResult] = []
        kb_results: list[RetrievalResult] = []

        for r in results:
            if r.source == RetrievalSource.NEO4J_GRAPH:
                layer = r.metadata.get("layer", "local_fact")
                if layer == "reasoning_path":
                    reasoning_paths.append(r)
                elif layer == "community":
                    community_summaries.append(r)
                else:
                    local_facts.append(r)
            elif r.source == RetrievalSource.MILVUS_QA:
                qa_results.append(r)
            elif r.source == RetrievalSource.MILVUS_KNOWLEDGE:
                kb_results.append(r)

        parts: list[str] = []
        idx = 1
        current_chars = 0

        # 层级化组装顺序
        sections = [
            ("推理路径", reasoning_paths),
            ("知识概览", community_summaries),
            ("确定性事实", local_facts),
            ("医疗问答参考", qa_results),
            ("医学文献参考", kb_results),
        ]

        for section_name, items in sections:
            if not items:
                continue

            section_lines: list[str] = [f"**{section_name}**"]

            for item in items:
                line = f"{idx}. {item.content.strip()}"
                if current_chars + len(line) > self.max_context_chars:
                    break
                section_lines.append(line)
                current_chars += len(line)
                idx += 1

            if len(section_lines) > 1:
                parts.append("\n".join(section_lines))

        context = "\n\n".join(parts)
        logger.info(f"上下文层级化拼接完成 | 条目数={idx - 1} | 字符数={len(context)}")
        return context

    def _build_compressed(self, results: list[RetrievalResult], user_query: str) -> str:
        """
        LLM 压缩模式：将所有检索结果交给 LLM 提炼去重。
        """
        # 先用直接拼接构造原始上下文
        raw_context = self._build_direct(results)

        # 调用 LLM 压缩
        try:
            messages = prompt_manager.build_prompt(
                COMPRESS_PROMPT_PATH,
                user_query=user_query,
                raw_context=raw_context,
            )

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=1500,
            )

            compressed = response.choices[0].message.content or raw_context
            logger.info(
                f"上下文压缩完成 | 压缩前={len(raw_context)} 字符 | "
                f"压缩后={len(compressed)} 字符 | "
                f"压缩率={len(compressed) / len(raw_context):.1%}"
            )
            return compressed

        except Exception as e:
            logger.error(f"LLM 上下文压缩失败: {e}，回退到直接拼接")
            return raw_context
