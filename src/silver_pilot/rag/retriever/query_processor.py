"""
模块名称：query_processor
功能描述：查询处理器，通过一次 LLM 调用同时完成查询重写（Rewrite）、
         查询分解（Decompose）和医学实体抽取（NER）三个子任务。
         使用 Pydantic 结构化输出确保 LLM 返回可解析的 JSON。
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from .models import EntityLabel, ExtractedEntity, ProcessedQuery

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "query_processor")

# ================= Prompt 模板 =================
PROMPT_TEMPLATE = "query_process"

# ================= 默认配置 =================
DEFAULT_MODEL = config.QUERY_PROCESS_MODEL


# ────────────────────────────────────────────────────────────
# Pydantic 结构化输出 Schema（供 LLM 解析用）
# ────────────────────────────────────────────────────────────


class EntityItem(BaseModel):
    name: str = Field(description="实体名称")
    label: str = Field(description="实体类型")


class QueryProcessOutput(BaseModel):
    rewritten_query: str = Field(description="改写后的规范化查询")
    sub_queries: list[str] = Field(default_factory=list, description="分解后的子查询列表")
    entities: list[EntityItem] = Field(default_factory=list, description="抽取的医学实体")


# ────────────────────────────────────────────────────────────
# 核心处理器
# ────────────────────────────────────────────────────────────


class QueryProcessor:
    """
    查询处理器：一次 LLM 调用完成重写 + 分解 + NER。

    使用示例::

        processor = QueryProcessor()
        result = processor.process("那个降压药能跟头孢一块儿吃不")
        print(result.rewritten_query)   # "降压药与头孢类抗生素是否存在药物相互作用"
        print(result.entities)          # [ExtractedEntity(name="降压药", label=Drug), ...]
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = config.QWEN_URL["cn"],
    ) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key or config.DASHSCOPE_API_KEY,
            base_url=base_url,
        )
        logger.info(f"QueryProcessor 初始化完成 | model={self.model}")

    def process(
        self,
        user_query: str,
        *,
        image_context: str = "",
        conversation_context: str = "",
    ) -> ProcessedQuery:
        """
        处理用户查询，返回结构化的 ProcessedQuery。

        Args:
            user_query: 用户原始输入文本
            image_context: （可选）图像 OCR/描述结果，来自多模态感知层
            conversation_context: （可选）最近几轮对话的摘要，用于指代消解

        Returns:
            ProcessedQuery: 包含重写、子查询、实体的结构化结果
        """
        logger.info(f"开始处理查询 | query={user_query[:30]}...")

        # 1. 构建 Prompt
        messages = prompt_manager.build_prompt(
            PROMPT_TEMPLATE,
            user_query=user_query,
            image_context=image_context,
            conversation_context=conversation_context,
        )

        # 2. 调用 LLM（结构化输出）
        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=QueryProcessOutput,
                temperature=0.1,
            )
            parsed: QueryProcessOutput | None = response.choices[0].message.parsed

            if parsed is None:
                logger.warning("LLM 返回解析失败，回退到原始查询")
                return self._fallback(user_query)

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}，回退到原始查询")
            return self._fallback(user_query)

        # 3. 转换为内部数据模型
        entities = []
        for ent in parsed.entities:
            try:
                label = EntityLabel(ent.label)
            except ValueError:
                label = EntityLabel.UNKNOWN
                logger.debug(f"未知实体类型 '{ent.label}'，归类为 UNKNOWN")

            entities.append(ExtractedEntity(name=ent.name, label=label))

        result = ProcessedQuery(
            original_query=user_query,
            rewritten_query=parsed.rewritten_query,
            sub_queries=parsed.sub_queries,
            entities=entities,
        )

        logger.info(
            f"查询处理完成 | rewritten={result.rewritten_query[:50]}... | "
            f"sub_queries={len(result.sub_queries)} | entities={len(result.entities)}"
        )
        return result

    @staticmethod
    def _fallback(user_query: str) -> ProcessedQuery:
        """LLM 调用失败时的降级方案：直接使用原始查询，不做实体抽取。"""
        return ProcessedQuery(
            original_query=user_query,
            rewritten_query=user_query,
            sub_queries=[],
            entities=[],
        )
