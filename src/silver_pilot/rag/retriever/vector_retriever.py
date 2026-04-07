"""
模块名称：vector_retriever
功能描述：Milvus 向量数据库检索器，封装对 medical_qa_lite（问答库）和
         medical_knowledge_base（知识库）两个集合的差异化检索策略。
         支持纯向量检索和 dense + BM25 混合检索两种模式。
         QA 库用完整 query 检索相似问题，知识库用子问题/关键词检索文档片段。
"""

from __future__ import annotations

from pymilvus import AnnSearchRequest

from silver_pilot.config import config
from silver_pilot.dao import MilvusManager
from silver_pilot.perception import create_embedder
from silver_pilot.perception.embedder import BaseEmbedder
from silver_pilot.utils import get_channel_logger

from .models import ProcessedQuery, RetrievalResult, RetrievalSource

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "vector_retriever")

# ================= 默认配置 =================
DEFAULT_QA_COLLECTION = config.MILVUS_QA_COLLECTION
DEFAULT_KB_COLLECTION = config.MILVUS_KB_COLLECTION
DEFAULT_TOP_K = config.VECTOR_RETRIEVAL_TOP_K
DEFAULT_QA_HYBRID_MIN_SCORE = float(getattr(config, "VECTOR_QA_HYBRID_MIN_SCORE", 0.005))
DEFAULT_KB_HYBRID_MIN_SCORE = float(getattr(config, "VECTOR_KB_HYBRID_MIN_SCORE", 0.005))


class VectorRetriever:
    """
    Milvus 双集合向量检索器。

    两个集合的检索策略不同：
    - **medical_qa_lite**: 用重写后的完整 query 检索相似问题，适合有明确问答模式的需求
    - **medical_knowledge_base**: 用子问题/关键实体短语检索文档片段，适合需要定位具体段落的需求

    支持模式：
    - Dense + BM25 混合检索，利用 Milvus 原生 BM25 全文搜索与 Dense 结果进行 RRF 融合去重输出

    使用示例::

        retriever = VectorRetriever(backend="qwen")
        results = retriever.retrieve(processed_query, top_k=5)
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        backend: str = "local",
        qa_collection: str = DEFAULT_QA_COLLECTION,
        kb_collection: str = DEFAULT_KB_COLLECTION,
    ) -> None:
        # Embedding 模型（两个集合共用同一个 Embedder）
        self.embedder = embedder or create_embedder(backend)

        # DAO 实例
        self.qa_manager = MilvusManager(collection_name=qa_collection)
        self.kb_manager = MilvusManager(collection_name=kb_collection)

        logger.info(
            f"VectorRetriever 初始化完成 | qa={qa_collection} | kb={kb_collection} "
            "| mode=hybrid(dense+BM25)"
        )

    # ──────────────────────────────────────────────────
    # 统一入口
    # ──────────────────────────────────────────────────

    def retrieve(
        self,
        processed_query: ProcessedQuery,
        *,
        top_k: int = DEFAULT_TOP_K,
        qa_enabled: bool = True,
        kb_enabled: bool = True,
        kb_filters: str | None = None,
    ) -> list[RetrievalResult]:
        """
        对两个集合同时执行检索并合并结果。

        使用 Milvus 原生 hybrid_search 同时执行
        dense 向量检索和 BM25 全文检索，RRF 融合后去重输出。

        Args:
            processed_query: 查询处理阶段的输出
            top_k: 每个集合返回的最大结果数
            qa_enabled: 是否启用 QA 库检索
            kb_enabled: 是否启用知识库检索
            kb_filters: 知识库的标量过滤表达式（如 'doc_type == "drug_manual"'）

        Returns:
            list[RetrievalResult]: 合并后的检索结果
        """
        all_results: list[RetrievalResult] = []
        seen_keys: set[str] = set()  # 跨集合去重，优先使用 chunk_id

        if qa_enabled:
            qa_results = self.retrieve_qa(processed_query.rewritten_query, top_k=top_k)
            for r in qa_results:
                result_key = self._result_identity(r)
                if result_key not in seen_keys:
                    seen_keys.add(result_key)
                    all_results.append(r)

        if kb_enabled:
            kb_results = self.retrieve_knowledge(processed_query, top_k=top_k, expr=kb_filters)
            for r in kb_results:
                result_key = self._result_identity(r)
                if result_key not in seen_keys:
                    seen_keys.add(result_key)
                    all_results.append(r)

        # 统计
        qa_count = len([r for r in all_results if r.source == RetrievalSource.MILVUS_QA])
        kb_count = len([r for r in all_results if r.source == RetrievalSource.MILVUS_KNOWLEDGE])

        all_results.sort(key=lambda item: item.score, reverse=True)

        logger.info(f"向量检索完成 | QA={qa_count} | KB={kb_count} |模式=hybrid(dense+BM25)")
        return all_results

    # ──────────────────────────────────────────────────
    # QA 库检索
    # ──────────────────────────────────────────────────

    def retrieve_qa(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_QA_HYBRID_MIN_SCORE,
    ) -> list[RetrievalResult]:
        """
        在 medical_qa_lite 中利用 Milvus 原生 hybrid_search
        同时执行 dense 向量检索和 BM25 全文检索。
        """
        logger.debug(f"QA 库检索 | query={query[:30]}... | top_k={top_k}")

        try:
            query_vector = self.embedder.encode_query(query)

            # 构建 dense 检索请求
            dense_req = AnnSearchRequest(
                data=[query_vector],
                anns_field="question_vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
            )

            # 构建 BM25 检索请求（传入原始文本）
            bm25_req = AnnSearchRequest(
                data=[query],
                anns_field="question_sparse",
                param={"metric_type": "BM25"},
                limit=top_k,
            )

            # 执行混合检索
            results = self.qa_manager.hybrid_search(
                reqs=[dense_req, bm25_req],
                limit=top_k + top_k,
                output_fields=["question_text", "answer_text", "department", "score"],
            )

            return self._parse_hybrid_qa_results(results, min_score=min_score)

        except Exception as e:
            logger.error(f"QA 库 hybrid 检索失败: {e}")
            return []

    # ──────────────────────────────────────────────────
    # 知识库检索
    # ──────────────────────────────────────────────────

    def retrieve_knowledge(
        self,
        processed_query: ProcessedQuery,
        *,
        top_k: int = DEFAULT_TOP_K,
        expr: str | None = None,
        min_score: float = DEFAULT_KB_HYBRID_MIN_SCORE,
    ) -> list[RetrievalResult]:
        """
        在 medical_knowledge_base 中检索相关文档片段。

        策略：
        - 如果有子查询，逐个子查询检索后合并去重
        - 如果无子查询，用重写后的主 query 检索
        - 支持标量过滤（doc_type, group_name 等）
        """
        search_queries = (
            processed_query.sub_queries
            if processed_query.sub_queries
            else [processed_query.rewritten_query]
        )

        logger.debug(f"知识库检索 | queries={len(search_queries)} | expr={expr}")

        all_results: list[RetrievalResult] = []
        seen_keys: set[str] = set()

        for query in search_queries:
            results = self._search_knowledge_single(
                query, top_k=top_k, expr=expr, min_score=min_score
            )
            for r in results:
                result_key = self._result_identity(r)
                if result_key not in seen_keys:
                    seen_keys.add(result_key)
                    all_results.append(r)

        all_results.sort(key=lambda item: item.score, reverse=True)
        logger.debug(f"知识库检索返回 {len(all_results)} 条去重结果")
        return all_results

    def _search_knowledge_single(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        expr: str | None = None,
        min_score: float = DEFAULT_KB_HYBRID_MIN_SCORE,
    ) -> list[RetrievalResult]:
        """对知识库执行混合检索 (Dense + BM25)。"""
        try:
            query_vector = self.embedder.encode_query(query)

            # Dense 检索请求
            dense_req = AnnSearchRequest(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                expr=expr,
            )

            # BM25 检索请求
            bm25_req = AnnSearchRequest(
                data=[query],
                anns_field="content_sparse",
                param={"metric_type": "BM25"},
                limit=top_k,
                expr=expr,
            )

            results = self.kb_manager.hybrid_search(
                reqs=[dense_req, bm25_req],
                limit=top_k + top_k,
                output_fields=[
                    "chunk_id",
                    "content",
                    "title",
                    "doc_type",
                    "group_name",
                    "source_file",
                    "meta",
                ],
            )

            return self._parse_hybrid_kb_results(results, min_score=min_score)

        except Exception as e:
            logger.error(f"知识库 hybrid 检索失败: {e}")
            return []

    # ──────────────────────────────────────────────────
    # 结果解析 — QA
    # ──────────────────────────────────────────────────

    @staticmethod
    def _parse_hybrid_qa_results(
        results: list,
        *,
        min_score: float = DEFAULT_QA_HYBRID_MIN_SCORE,
    ) -> list[RetrievalResult]:
        """解析 QA 库的 hybrid_search 结果（RRF 融合后）。"""
        retrieval_results: list[RetrievalResult] = []

        for hits in results:
            for hit in hits:
                rrf_score = hit.score
                if rrf_score < min_score:
                    continue

                entity = hit.fields if hasattr(hit, "fields") else hit.entity
                question = entity.get("question_text", "")
                answer = entity.get("answer_text", "")
                department = entity.get("department", "")

                content = f"问: {question}\n答: {answer}"

                retrieval_results.append(
                    RetrievalResult(
                        content=content,
                        source=RetrievalSource.MILVUS_QA,
                        score=rrf_score,
                        metadata={
                            "question": question,
                            "department": department,
                            "qa_quality_score": entity.get("score", 0),
                        },
                    )
                )

        logger.debug(f"QA 库 hybrid 检索返回 {len(retrieval_results)} 条结果")
        return retrieval_results

    # ──────────────────────────────────────────────────
    # 结果解析 — 知识库
    # ──────────────────────────────────────────────────

    @staticmethod
    def _parse_hybrid_kb_results(
        results: list,
        *,
        min_score: float = DEFAULT_KB_HYBRID_MIN_SCORE,
    ) -> list[RetrievalResult]:
        """解析知识库的 hybrid_search 结果（RRF 融合后）。"""
        retrieval_results: list[RetrievalResult] = []

        for hits in results:
            for hit in hits:
                rrf_score = hit.score
                if rrf_score < min_score:
                    continue

                entity = hit.fields if hasattr(hit, "fields") else hit.entity
                chunk_id = entity.get("chunk_id", "")
                content = entity.get("content", "")
                title = entity.get("title", "")
                meta = entity.get("meta", {})

                display_content = f"[{title}] {content}" if title else content

                retrieval_results.append(
                    RetrievalResult(
                        content=display_content,
                        source=RetrievalSource.MILVUS_KNOWLEDGE,
                        score=rrf_score,
                        metadata={
                            "chunk_id": chunk_id,
                            "title": title,
                            "doc_type": entity.get("doc_type", ""),
                            "group_name": entity.get("group_name", ""),
                            "source_file": entity.get("source_file", ""),
                            "metadata": meta,
                        }
                        if entity.get("doc_type", "") == "drug_manual"
                        else {
                            "chunk_id": chunk_id,
                            "title": title,
                            "doc_type": entity.get("doc_type", ""),
                            "group_name": entity.get("group_name", ""),
                            "source_file": entity.get("source_file", ""),
                            "section_path": meta.get("section_path", ""),
                        },
                    )
                )

        logger.debug(f"知识库 hybrid 检索返回 {len(retrieval_results)} 条结果")
        return retrieval_results

    @staticmethod
    def _result_identity(result: RetrievalResult) -> str:
        """优先使用稳定的 chunk_id 做去重，缺失时再回退到内容前缀。"""
        chunk_id = str(result.metadata.get("chunk_id", "")).strip()
        if chunk_id:
            return f"chunk:{chunk_id}"
        return f"content:{result.content[:100]}"
