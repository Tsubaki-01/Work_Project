"""
模块名称：pipeline
功能描述：RAG 检索流水线编排器，串联查询处理、实体链接、多路检索、重排序和上下文构建
         各阶段组件，提供统一的 retrieve() 接口供上层 Agent 调用。
"""

import time
from dataclasses import dataclass

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

from .context_builder import ContextBuilder
from .entity_linker import EntityLinker
from .graph_retriever import GraphRetriever
from .models import (
    LinkedEntity,
    RetrievalContext,
    RetrievalResult,
    RetrievalSource,
)
from .query_processor import QueryProcessor
from .reranker import BaseReranker, create_reranker
from .vector_retriever import VectorRetriever

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "pipeline")


# ────────────────────────────────────────────────────────────
# 流水线配置
# ────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """RAG 检索流水线的可调参数集合。"""

    # 查询处理
    query_model: str = config.QUERY_PROCESS_MODEL

    # 实体链接
    linking_threshold: float = config.ENTITY_LINK_THRESHOLD
    linking_model: str = config.ENTITY_LINK_MODEL

    # 向量检索
    embedding_backend: str = config.EMBEDDER_MODE
    vector_top_k: int = config.VECTOR_RETRIEVAL_TOP_K
    qa_enabled: bool = config.VECTOR_QA_ENABLED
    kb_enabled: bool = config.VECTOR_KB_ENABLED

    # 图谱检索 (GraphRAG)
    graph_max_per_entity: int = config.GRAPH_RETRIEVER_MAX_RESULTS_PER_ENTITY
    graph_enable_community: bool = config.GRAPH_RETRIEVER_ENABLE_COMMUNITY
    graph_enable_reasoning: bool = config.GRAPH_RETRIEVER_ENABLE_REASONING

    # 重排序
    reranker_backend: str = config.RERANK_MODE
    rerank_top_k: int = config.RERANK_TOP_K

    # 上下文构建
    context_mode: str = config.CONTEXT_BUILDER_MODE
    max_context_chars: int = config.MAX_CONTEXT_CHARS


# ────────────────────────────────────────────────────────────
# 核心流水线
# ────────────────────────────────────────────────────────────


class RAGPipeline:
    """
    RAG 检索流水线：查询处理 → 实体链接 → 多路检索 → 重排序 → 上下文构建。

    **设计原则**:
    - 各阶段组件松耦合，可独立替换
    - 每个阶段有独立的降级策略（某阶段失败不影响整体流水线）
    - 保留完整的中间状态用于调试和评估（RAGAS）

    使用示例::

        pipeline = RAGPipeline()
        pipeline.initialize()  # 启动时调用一次

        # 运行时调用
        context = pipeline.retrieve("阿司匹林和华法林能一起吃吗")
        print(context.context_text)
    """

    def __init__(self, pipeline_config: PipelineConfig | None = None) -> None:
        self.config = pipeline_config or PipelineConfig()

        # 各阶段组件（延迟初始化）
        self._query_processor: QueryProcessor | None = None
        self._entity_linker: EntityLinker | None = None
        self._graph_retriever: GraphRetriever | None = None
        self._vector_retriever: VectorRetriever | None = None
        self._reranker: BaseReranker | None = None
        self._context_builder: ContextBuilder | None = None

        self._initialized = False
        logger.info("RAGPipeline 实例已创建，请调用 initialize() 完成初始化")

    # ──────────────────────────────────────────────────
    # 初始化
    # ──────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        初始化所有组件。应在系统启动时调用一次。

        包含耗时操作：加载模型权重、构建 Faiss 索引等。
        """
        logger.info("=" * 60)
        logger.info("RAG Pipeline 初始化开始")
        logger.info("=" * 60)

        if self._initialized:
            logger.info("RAG Pipeline 已初始化，无需重复初始化")
            return

        t_start = time.time()

        # 1. 查询处理器
        self._query_processor = QueryProcessor(model=self.config.query_model)

        # 2. 实体链接器
        self._entity_linker = EntityLinker(
            model_name=self.config.linking_model,
            threshold=self.config.linking_threshold,
        )
        self._entity_linker.load_index()  # 需要预先使用 scripts/rag/build_entity_index.py 构建索引

        # 3. 图谱检索器 (GraphRAG: 社区 + 推理路径 + 局部事实)
        self._graph_retriever = GraphRetriever(
            max_results_per_entity=self.config.graph_max_per_entity,
            enable_community=self.config.graph_enable_community,
            enable_reasoning=self.config.graph_enable_reasoning,
        )
        self._graph_retriever.initialize()  # 加载社区缓存、初始化路径推理器

        # 4. 向量检索器
        self._vector_retriever = VectorRetriever(backend=self.config.embedding_backend)

        # 5. 重排序器
        self._reranker = create_reranker(backend=self.config.reranker_backend)

        # 6. 上下文构建器
        self._context_builder = ContextBuilder(
            mode=self.config.context_mode,
            max_context_chars=self.config.max_context_chars,
        )

        assert self._query_processor is not None, "查询处理器初始化失败"
        assert self._entity_linker is not None, "实体链接器初始化失败"
        assert self._graph_retriever is not None, "图谱检索器初始化失败"
        assert self._vector_retriever is not None, "向量检索器初始化失败"
        assert self._reranker is not None, "重排序器初始化失败"
        assert self._context_builder is not None, "上下文构建器初始化失败"

        self._initialized = True
        elapsed = time.time() - t_start
        logger.info(f"RAG Pipeline 初始化完成 | 耗时={elapsed:.1f}s")
        logger.info("=" * 60)

    # ──────────────────────────────────────────────────
    # 核心检索接口
    # ──────────────────────────────────────────────────

    def retrieve(
        self,
        user_query: str,
        *,
        image_context: str = "",
        conversation_context: str = "",
        kb_filters: str | None = None,
    ) -> RetrievalContext:
        """
        执行完整的 RAG 检索流水线。

        Args:
            user_query: 用户原始输入文本
            image_context: 图像 OCR/描述结果（来自多模态感知层）
            conversation_context: 近几轮对话摘要（用于指代消解）
            kb_filters: 知识库的标量过滤表达式

        Returns:
            RetrievalContext: 包含最终 context 和完整中间状态的结果对象
        """
        if not self._initialized:
            raise RuntimeError("Pipeline 尚未初始化，请先调用 initialize()")
        if self._query_processor is None:
            raise RuntimeError("查询处理器未初始化")
        if self._entity_linker is None:
            raise RuntimeError("实体链接器未初始化")
        if self._graph_retriever is None:
            raise RuntimeError("图谱检索器未初始化")
        if self._vector_retriever is None:
            raise RuntimeError("向量检索器未初始化")
        if self._reranker is None:
            raise RuntimeError("重排序器未初始化")
        if self._context_builder is None:
            raise RuntimeError("上下文构建器未初始化")

        t_start = time.time()
        logger.info(f"开始 RAG 检索 | query={user_query[:50]}...")

        # ────── 阶段 1: 查询处理 ──────
        processed_query = self._query_processor.process(
            user_query,
            image_context=image_context,
            conversation_context=conversation_context,
        )

        # ────── 阶段 2: 实体链接 ──────
        linked_entities: list[LinkedEntity] = []
        if processed_query.entities:
            linked_entities = self._entity_linker.link(processed_query.entities)
        else:
            logger.debug("无抽取实体，跳过实体链接阶段")

        # ────── 阶段 3: 多路检索 ──────
        all_results: list[RetrievalResult] = []

        # 路径 A: GraphRAG 三层检索（社区摘要 + 推理路径 + 局部事实）
        if linked_entities:
            graph_results = self._graph_retriever.retrieve(
                linked_entities,
                query=processed_query.rewritten_query,
            )
            all_results.extend(graph_results)

        # 路径 B+C: 向量检索（QA 库 + 知识库）
        vector_results = self._vector_retriever.retrieve(
            processed_query,
            top_k=self.config.vector_top_k,
            qa_enabled=self.config.qa_enabled,
            kb_enabled=self.config.kb_enabled,
            kb_filters=kb_filters,
        )
        all_results.extend(vector_results)

        # 统计各路召回数量
        retrieval_stats = {
            "graph": len([r for r in all_results if r.source == RetrievalSource.NEO4J_GRAPH]),
            "qa": len([r for r in all_results if r.source == RetrievalSource.MILVUS_QA]),
            "knowledge": len(
                [r for r in all_results if r.source == RetrievalSource.MILVUS_KNOWLEDGE]
            ),
            "total": len(all_results),
        }
        logger.info(
            f"多路召回完成 | graph={retrieval_stats['graph']} | "
            f"qa={retrieval_stats['qa']} | kb={retrieval_stats['knowledge']} | "
            f"total={retrieval_stats['total']}"
        )

        # ────── 阶段 4: 重排序 ──────
        if all_results:
            ranked_results = self._reranker.rerank(
                query=processed_query.rewritten_query,
                results=all_results,
                top_k=self.config.rerank_top_k,
            )
        else:
            ranked_results = []
            logger.warning("无召回结果，跳过重排序")

        # ────── 阶段 5: 上下文构建 ──────
        context_text = self._context_builder.build(
            ranked_results,
            user_query=user_query,
        )

        # ────── 组装输出 ──────
        elapsed = time.time() - t_start
        logger.info(f"RAG 检索完成 | 耗时={elapsed:.2f}s | context 长度={len(context_text)}")

        return RetrievalContext(
            context_text=context_text,
            ranked_results=ranked_results,
            processed_query=processed_query,
            linked_entities=linked_entities,
            retrieval_stats=retrieval_stats,
        )

    # ──────────────────────────────────────────────────
    # 资源管理
    # ──────────────────────────────────────────────────

    def close(self) -> None:
        """释放所有资源。"""
        if self._graph_retriever:
            self._graph_retriever.close()
        logger.info("RAG Pipeline 资源已释放")
