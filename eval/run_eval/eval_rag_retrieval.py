"""
模块名称：eval_rag_retrieval
功能描述：对 RAG 检索链路做分阶段评测，覆盖查询处理、Milvus 混合检索、
         GraphRAG 扩展召回和统一重排后的最终结果。

评测链路（与当前生产实现对齐，并补上原脚本缺失部分）：
     1. 数据集 query 直通检索
         数据集中的 query 已视为最终改写结果，默认直接作为检索 query 使用。
     2. QueryProcessor.process()（可选）
         仅用于查询分解和实体抽取，不再覆盖数据集中的 query 文本。
     3. VectorRetriever.retrieve_knowledge()
       使用 Milvus dense + BM25 hybrid_search 召回知识库候选 chunk。
     4. EntityLinker.link() + GraphRetriever.retrieve()
       从知识图谱获取社区摘要、推理路径、局部事实，并将这些线索转成图谱扩展查询。
     5. Graph-guided KB retrieval
       用图谱扩展查询再次回打 Milvus 知识库，扩大 chunk 候选池。
     6. Reranker.rerank()
       对合并后的 chunk 候选统一重排，截断到 top_k。

指标设计：
    - vector_hybrid：仅看 Milvus 混合检索候选池的命中情况
    - graph_augmented：加入 KG 扩展召回后的候选池命中情况
    - final：统一重排后 top_k 的最终 Hit Rate@K / Recall@K

注意：
    - 标注集仍是 chunk_id 级别，因此 KG 结果不会直接算命中；
      KG 只作为“扩展召回器”帮助找回更多 chunk 候选。
    - top-level 的 hit_rate / recall_at_k 仍表示最终重排后的指标，兼容现有报告脚本。

输入：data/eval/rag_retrieval.jsonl（Dataset A）
输出：data/eval/results/rag_retrieval_result.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from silver_pilot.config import config
from silver_pilot.rag.retriever import (
    EntityLinker,
    GraphRetriever,
    ProcessedQuery,
    QueryProcessor,
    RetrievalResult,
    VectorRetriever,
    create_reranker,
)
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "eval", "eval_rag_retrieval")

# ================= 路径配置 =================
DATASET_PATH = config.DATA_DIR / "eval" / "rag_retrieval.jsonl"
OUTPUT_PATH = config.DATA_DIR / "eval" / "results" / "rag_retrieval_result.json"

# ================= 评估配置 =================
TOP_K: int = 3
RERANK_FETCH_K: int = TOP_K * 4
MERGED_CANDIDATE_LIMIT: int = RERANK_FETCH_K * 2
GRAPH_EXPANSION_MAX_QUERIES: int = 8
GRAPH_EXPANSION_TOP_K: int = max(TOP_K * 2, RERANK_FETCH_K // 2)
EMBEDDING_BACKEND: str = config.EMBEDDER_MODE
RERANKER_BACKEND: str = config.RERANK_MODE

StageMetricValue = bool | float | int | list[str]
StageMetricSummaryValue = float | int


# ────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────


@dataclass
class SingleQueryResult:
    """单条 query 的分阶段检索评估结果。"""

    query_id: str
    query: str
    rewritten_query: str
    sub_queries: list[str]
    linked_entities: list[dict]
    relevant_chunk_ids: list[str]
    vector_chunk_ids: list[str]
    graph_augmented_chunk_ids: list[str]
    reranked_chunk_ids: list[str]
    vector_scores: list[float]
    graph_augmented_scores: list[float]
    rerank_scores: list[float]
    graph_clues: list[str]
    stage_metrics: dict[str, dict] = field(default_factory=dict)

    @property
    def hit(self) -> bool:
        return bool(self.stage_metrics.get("final", {}).get("hit", False))

    @property
    def recall(self) -> float:
        return float(self.stage_metrics.get("final", {}).get("recall", 0.0))


@dataclass
class EvalResult:
    """全量评估汇总结果。"""

    total_queries: int = 0
    hit_rate: float = 0.0
    recall_at_k: float = 0.0
    top_k: int = TOP_K
    rerank_fetch_k: int = RERANK_FETCH_K
    stage_metrics: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_category: dict[str, dict] = field(default_factory=dict)
    failed_cases: list[dict] = field(default_factory=list)
    per_query_results: list[dict] = field(default_factory=list)


# ────────────────────────────────────────────────────────────
# 核心评估函数
# ────────────────────────────────────────────────────────────


def run_eval(
    dataset_path: Path = DATASET_PATH,
    output_path: Path = OUTPUT_PATH,
    top_k: int = TOP_K,
    rerank_fetch_k: int = RERANK_FETCH_K,
    merged_candidate_limit: int = MERGED_CANDIDATE_LIMIT,
    graph_expansion_max_queries: int = GRAPH_EXPANSION_MAX_QUERIES,
    graph_expansion_top_k: int = GRAPH_EXPANSION_TOP_K,
    embedding_backend: str = EMBEDDING_BACKEND,
    reranker_backend: str = RERANKER_BACKEND,
    kb_filters: str | None = None,
    enable_query_processing: bool = True,
    use_dataset_query_as_rewritten: bool = True,
    enable_graph_guided_retrieval: bool = True,
) -> EvalResult:
    """
    执行分阶段 RAG 检索评估。

    Args:
        dataset_path: Dataset A 路径（含 query 和 relevant_chunk_ids）
        output_path: 评估结果输出路径
        top_k: 最终重排后的评估深度
        rerank_fetch_k: 向量混合检索的基础候选池大小
        merged_candidate_limit: 图谱扩展后送入重排器的最大候选数
        graph_expansion_max_queries: 从图谱结果生成的扩展查询数量上限
        graph_expansion_top_k: 每条图谱扩展查询从知识库取回的候选数
        embedding_backend: 向量编码后端
        reranker_backend: 重排后端
        kb_filters: 知识库检索过滤条件
        enable_query_processing: 是否启用查询分解/NER
        use_dataset_query_as_rewritten: 是否直接把数据集 query 当作最终检索 query
        enable_graph_guided_retrieval: 是否启用图谱扩展召回

    Returns:
        EvalResult: 包含最终指标、分阶段指标和逐条结果的评估对象
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset A 不存在: {dataset_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_records(dataset_path)
    if not records:
        raise ValueError(f"数据集为空: {dataset_path}")

    record_index = {record["_eval_query_id"]: record for record in records}
    logger.info(f"加载 Dataset A：{len(records)} 条 query")

    query_processor = (
        QueryProcessor(model=config.QUERY_PROCESS_MODEL) if enable_query_processing else None
    )

    logger.info("初始化 EntityLinker ...")
    entity_linker = EntityLinker(
        model_name=config.ENTITY_LINK_MODEL,
        threshold=float(config.ENTITY_LINK_THRESHOLD),
    )
    entity_linker.load_index()

    logger.info(f"初始化 VectorRetriever（backend={embedding_backend}）...")
    vector_retriever = VectorRetriever(backend=embedding_backend)

    graph_retriever: GraphRetriever | None = None
    if enable_graph_guided_retrieval:
        logger.info("初始化 GraphRetriever ...")
        graph_retriever = GraphRetriever(
            max_results_per_entity=int(config.GRAPH_RETRIEVER_MAX_RESULTS_PER_ENTITY),
            enable_community=config.GRAPH_RETRIEVER_ENABLE_COMMUNITY,
            enable_reasoning=config.GRAPH_RETRIEVER_ENABLE_REASONING,
        )
        graph_retriever.initialize()

    logger.info(f"初始化 Reranker（backend={reranker_backend}）...")
    reranker = create_reranker(backend=reranker_backend)

    logger.info(
        "评估配置 | "
        f"top_k={top_k} | rerank_fetch_k={rerank_fetch_k} | "
        f"merged_candidate_limit={merged_candidate_limit} | "
        f"graph_expansion_max_queries={graph_expansion_max_queries} | "
        f"graph_expansion_top_k={graph_expansion_top_k} | "
        f"query_processing={enable_query_processing} | "
        f"dataset_query_as_rewritten={use_dataset_query_as_rewritten} | "
        f"graph_guided={enable_graph_guided_retrieval}"
    )

    query_results: list[SingleQueryResult] = []

    try:
        for idx, record in enumerate(records):
            query_id = record["_eval_query_id"]
            query = record["query"]
            relevant_ids = [
                str(chunk_id) for chunk_id in record.get("relevant_chunk_ids", []) if chunk_id
            ]

            logger.info(f"[{idx + 1}/{len(records)}] {query_id}: {query[:40]}...")

            if not relevant_ids:
                logger.warning(f"  {query_id} 没有标注的 relevant_chunk_ids，跳过")
                continue

            processed_query = _process_query(
                query,
                query_processor,
                use_dataset_query_as_rewritten=use_dataset_query_as_rewritten,
            )
            linked_entities = _link_entities(processed_query, entity_linker)

            query_embedding = None
            try:
                query_embedding = vector_retriever.embedder.encode_query(
                    processed_query.rewritten_query
                )
            except Exception as exc:
                logger.warning(f"  查询向量编码失败，将跳过社区语义检索 | error={exc}")

            vector_candidates = vector_retriever.retrieve_knowledge(
                processed_query,
                top_k=rerank_fetch_k,
                expr=kb_filters,
                min_score=0.0,
            )
            _tag_results(vector_candidates, stage="vector_hybrid")
            vector_candidates = _sort_results(vector_candidates)[:rerank_fetch_k]

            graph_results: list[RetrievalResult] = []
            graph_clues: list[str] = []
            graph_augmented_candidates = list(vector_candidates)

            if graph_retriever is not None and linked_entities:
                try:
                    graph_results = graph_retriever.retrieve(
                        linked_entities,
                        query=processed_query.rewritten_query,
                        query_embedding=query_embedding,
                    )
                except Exception as exc:
                    logger.error(f"  GraphRetriever 失败: {exc}")
                    graph_results = []

                graph_clues = _build_graph_expansion_queries(
                    linked_entities=linked_entities,
                    graph_results=graph_results,
                    max_queries=graph_expansion_max_queries,
                )

                if graph_clues:
                    graph_expansion_query = ProcessedQuery(
                        original_query=query,
                        rewritten_query=processed_query.rewritten_query,
                        sub_queries=graph_clues,
                    )
                    graph_guided_candidates = vector_retriever.retrieve_knowledge(
                        graph_expansion_query,
                        top_k=graph_expansion_top_k,
                        expr=kb_filters,
                        min_score=0.0,
                    )
                    _tag_results(graph_guided_candidates, stage="graph_guided")
                    graph_augmented_candidates = _merge_candidates(
                        vector_candidates,
                        graph_guided_candidates,
                        limit=merged_candidate_limit,
                    )

            reranked_results = _rerank_results(
                reranker=reranker,
                query=processed_query.rewritten_query,
                candidates=graph_augmented_candidates,
                top_k=top_k,
            )

            vector_chunk_ids = _extract_chunk_ids(vector_candidates)
            graph_augmented_chunk_ids = _extract_chunk_ids(graph_augmented_candidates)
            reranked_chunk_ids = _extract_chunk_ids(reranked_results)

            stage_metrics = {
                "vector_hybrid": _compute_stage_metrics(vector_chunk_ids, relevant_ids),
                "graph_augmented": _compute_stage_metrics(graph_augmented_chunk_ids, relevant_ids),
                "final": _compute_stage_metrics(reranked_chunk_ids, relevant_ids),
            }

            result = SingleQueryResult(
                query_id=query_id,
                query=query,
                rewritten_query=processed_query.rewritten_query,
                sub_queries=processed_query.sub_queries,
                linked_entities=[entity.to_dict() for entity in linked_entities],
                relevant_chunk_ids=relevant_ids,
                vector_chunk_ids=vector_chunk_ids,
                graph_augmented_chunk_ids=graph_augmented_chunk_ids,
                reranked_chunk_ids=reranked_chunk_ids,
                vector_scores=[float(item.score) for item in vector_candidates],
                graph_augmented_scores=[float(item.score) for item in graph_augmented_candidates],
                rerank_scores=[
                    float(item.rerank_score) if item.rerank_score is not None else float(item.score)
                    for item in reranked_results
                ],
                graph_clues=graph_clues,
                stage_metrics=stage_metrics,
            )
            query_results.append(result)

            logger.info(
                "  "
                f"vector_hit={stage_metrics['vector_hybrid']['hit']} | "
                f"graph_hit={stage_metrics['graph_augmented']['hit']} | "
                f"final_hit={stage_metrics['final']['hit']} | "
                f"final_recall={stage_metrics['final']['recall']:.3f}"
            )
    finally:
        if graph_retriever is not None:
            graph_retriever.close()

    if not query_results:
        logger.error("没有有效的评估结果，请检查数据集和外部依赖")
        return EvalResult(top_k=top_k, rerank_fetch_k=rerank_fetch_k)

    aggregate_metrics = _aggregate_stage_metrics(query_results)
    final_metrics = aggregate_metrics["final"]

    logger.info("=" * 60)
    logger.info(f"评估完成 | 总 query 数={len(query_results)}")
    logger.info(
        f"Vector Hybrid Recall@{rerank_fetch_k}: "
        f"{aggregate_metrics['vector_hybrid']['recall_at_k']:.4f} | "
        f"Hit Rate={aggregate_metrics['vector_hybrid']['hit_rate']:.4f}"
    )
    logger.info(
        f"Graph Augmented Recall@{merged_candidate_limit}: "
        f"{aggregate_metrics['graph_augmented']['recall_at_k']:.4f} | "
        f"Hit Rate={aggregate_metrics['graph_augmented']['hit_rate']:.4f}"
    )
    logger.info(
        f"Final Recall@{top_k}: {final_metrics['recall_at_k']:.4f} | "
        f"Hit Rate@{top_k}: {final_metrics['hit_rate']:.4f}"
    )

    by_difficulty = _group_stats(
        query_results,
        key_fn=lambda item: record_index[item.query_id].get("difficulty", "medium"),
        stage_name="final",
    )
    by_category = _group_stats(
        query_results,
        key_fn=lambda item: record_index[item.query_id].get("category", "unknown"),
        stage_name="final",
    )

    failed_cases = [
        {
            "query_id": result.query_id,
            "query": result.query,
            "rewritten_query": result.rewritten_query,
            "relevant_chunk_ids": result.relevant_chunk_ids,
            "vector_chunk_ids": result.vector_chunk_ids[:rerank_fetch_k],
            "graph_augmented_chunk_ids": result.graph_augmented_chunk_ids[:merged_candidate_limit],
            "reranked_chunk_ids": result.reranked_chunk_ids,
            "linked_entities": result.linked_entities,
            "graph_clues": result.graph_clues,
            "stage_metrics": result.stage_metrics,
        }
        for result in query_results
        if not result.hit
    ]
    logger.info(f"失败案例数（final hit=False）: {len(failed_cases)}")

    eval_result = EvalResult(
        total_queries=len(query_results),
        hit_rate=round(final_metrics["hit_rate"], 4),
        recall_at_k=round(final_metrics["recall_at_k"], 4),
        top_k=top_k,
        rerank_fetch_k=rerank_fetch_k,
        stage_metrics=aggregate_metrics,
        by_difficulty=by_difficulty,
        by_category=by_category,
        failed_cases=failed_cases[:10],
        per_query_results=[
            {
                "query_id": result.query_id,
                "query": result.query[:80],
                "rewritten_query": result.rewritten_query[:80],
                "linked_entities": [
                    entity.get("neo4j_name") or entity.get("original_name")
                    for entity in result.linked_entities
                ],
                "stage_metrics": result.stage_metrics,
                "vector_ids": result.vector_chunk_ids[:rerank_fetch_k],
                "graph_augmented_ids": result.graph_augmented_chunk_ids[:merged_candidate_limit],
                "reranked_ids": result.reranked_chunk_ids,
                "rerank_scores": [round(score, 4) for score in result.rerank_scores],
            }
            for result in query_results
        ],
    )

    _save_result(eval_result, output_path)
    return eval_result


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────


def _load_records(dataset_path: Path) -> list[dict]:
    """读取 JSONL 数据集，并补齐评测内部使用的 query_id。"""
    records: list[dict] = []
    with open(dataset_path, encoding="utf-8") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_eval_query_id"] = record.get("id") or f"q_{idx}"
            records.append(record)
    return records


def _process_query(
    query: str,
    query_processor: QueryProcessor | None,
    *,
    use_dataset_query_as_rewritten: bool = True,
) -> ProcessedQuery:
    """统一封装查询处理，默认保留数据集 query 作为最终检索 query。"""
    if query_processor is None:
        return ProcessedQuery(original_query=query, rewritten_query=query)

    try:
        processed = query_processor.process(query)
        if not use_dataset_query_as_rewritten:
            return processed

        sub_queries: list[str] = []
        seen_queries: set[str] = {query}
        for sub_query in processed.sub_queries:
            normalized = str(sub_query).strip()
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            sub_queries.append(normalized)

        return ProcessedQuery(
            original_query=query,
            rewritten_query=query,
            sub_queries=sub_queries,
            entities=processed.entities,
        )
    except Exception as exc:
        logger.error(f"  QueryProcessor 失败，回退到原始 query | error={exc}")
        return ProcessedQuery(original_query=query, rewritten_query=query)


def _link_entities(processed_query: ProcessedQuery, entity_linker: EntityLinker) -> list[Any]:
    """执行实体链接，失败时返回空列表。"""
    if not processed_query.entities:
        return []

    try:
        return entity_linker.link(processed_query.entities)
    except Exception as exc:
        logger.error(f"  EntityLinker 失败，跳过图谱扩展 | error={exc}")
        return []


def _tag_results(results: list[RetrievalResult], stage: str) -> None:
    """给候选结果打上召回阶段标签，便于失败分析。"""
    for result in results:
        stages = set(result.metadata.get("retrieval_stages", []))
        stages.add(stage)
        result.metadata["retrieval_stages"] = sorted(stages)


def _sort_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """按当前检索分数从高到低排序。"""
    return sorted(results, key=lambda item: item.score, reverse=True)


def _candidate_key(result: RetrievalResult) -> str:
    """构建候选去重键，优先使用 chunk_id。"""
    chunk_id = str(result.metadata.get("chunk_id", "")).strip()
    if chunk_id:
        return f"chunk:{chunk_id}"
    return f"content:{result.content[:120]}"


def _merge_candidates(
    *candidate_lists: list[RetrievalResult], limit: int | None = None
) -> list[RetrievalResult]:
    """合并多路 chunk 候选，按 chunk_id 去重并保留最高分。"""
    merged: dict[str, RetrievalResult] = {}

    for candidates in candidate_lists:
        for candidate in candidates:
            key = _candidate_key(candidate)
            if key not in merged:
                merged[key] = candidate
                continue

            existing = merged[key]
            stage_tags = set(existing.metadata.get("retrieval_stages", []))
            stage_tags.update(candidate.metadata.get("retrieval_stages", []))

            if candidate.score > existing.score:
                candidate.metadata["retrieval_stages"] = sorted(stage_tags)
                merged[key] = candidate
            else:
                existing.metadata["retrieval_stages"] = sorted(stage_tags)

    merged_results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    if limit is not None:
        return merged_results[:limit]
    return merged_results


def _build_graph_expansion_queries(
    linked_entities: list,
    graph_results: list[RetrievalResult],
    max_queries: int,
) -> list[str]:
    """
    将图谱召回结果转换成知识库扩展查询。

    思路：
    - 直接用已链接实体名补召回
    - 用局部事实中的起止实体补共现检索
    - 用推理路径和社区关键词补充 BM25 线索
    """
    queries: list[str] = []
    seen: set[str] = set()

    def add_query(text: str) -> None:
        normalized = " ".join(str(text).split())[:120].strip()
        if len(normalized) < 2 or normalized in seen:
            return
        seen.add(normalized)
        queries.append(normalized)

    entity_names = []
    for entity in linked_entities:
        best_name = getattr(entity, "best_name", "") or getattr(entity, "original_name", "")
        if best_name:
            entity_names.append(best_name)
            add_query(best_name)

    if len(entity_names) >= 2:
        add_query(" ".join(entity_names[:2]))

    for result in sorted(graph_results, key=lambda item: item.score, reverse=True):
        layer = result.metadata.get("layer", "")

        if layer == "community":
            for keyword in result.metadata.get("community_keywords", []):
                add_query(keyword)
            add_query(result.content.replace("[医学知识社区摘要]", ""))
        elif layer == "reasoning_path":
            add_query(result.metadata.get("triplet_chain", ""))
            add_query(result.content.replace("[推理路径]", ""))
        else:
            start_entity = result.metadata.get("start_entity", "")
            end_entity = result.metadata.get("end_entity", "")
            add_query(start_entity)
            add_query(end_entity)
            if start_entity and end_entity:
                add_query(f"{start_entity} {end_entity}")
            add_query(result.content)

        if len(queries) >= max_queries:
            break

    return queries[:max_queries]


def _rerank_results(
    reranker: Any,
    query: str,
    candidates: list[RetrievalResult],
    top_k: int,
) -> list[RetrievalResult]:
    """统一封装重排，失败时按原始分数回退。"""
    if not candidates:
        return []

    try:
        return reranker.rerank(query=query, results=candidates, top_k=top_k)
    except Exception as exc:
        logger.error(f"  Reranker 失败: {exc}，回退到原始分数排序")
        return _sort_results(candidates)[:top_k]


def _extract_chunk_ids(results: list[RetrievalResult]) -> list[str]:
    """从 RetrievalResult 列表中提取 chunk_id，保留顺序并去掉空值。"""
    chunk_ids: list[str] = []
    for result in results:
        chunk_id = str(result.metadata.get("chunk_id", "")).strip()
        if chunk_id:
            chunk_ids.append(chunk_id)
    return chunk_ids


def _compute_stage_metrics(
    retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str]
) -> dict[str, StageMetricValue]:
    """计算单阶段 Hit / Recall 指标。"""
    relevant_set = {str(chunk_id) for chunk_id in relevant_chunk_ids if chunk_id}
    retrieved_set = {str(chunk_id) for chunk_id in retrieved_chunk_ids if chunk_id}
    intersection = sorted(relevant_set & retrieved_set)

    recall = len(intersection) / len(relevant_set) if relevant_set else 0.0
    return {
        "hit": bool(intersection),
        "recall": recall,
        "matched_chunk_ids": intersection,
        "retrieved_count": len(retrieved_chunk_ids),
    }


def _aggregate_stage_metrics(
    results: list[SingleQueryResult],
) -> dict[str, dict[str, StageMetricSummaryValue]]:
    """聚合所有 query 的分阶段指标。"""
    stage_names = ["vector_hybrid", "graph_augmented", "final"]
    aggregate: dict[str, dict[str, StageMetricSummaryValue]] = {}

    for stage_name in stage_names:
        count = len(results)
        hit_rate = sum(bool(item.stage_metrics[stage_name]["hit"]) for item in results) / count
        recall_at_k = (
            sum(float(item.stage_metrics[stage_name]["recall"]) for item in results) / count
        )
        aggregate[stage_name] = {
            "count": count,
            "hit_rate": round(hit_rate, 4),
            "recall_at_k": round(recall_at_k, 4),
        }

    return aggregate


def _group_stats(
    results: list[SingleQueryResult],
    key_fn: Callable[[SingleQueryResult], str],
    stage_name: str = "final",
) -> dict[str, dict[str, StageMetricSummaryValue]]:
    """按指定维度对某个阶段的 Hit Rate / Recall 做分层统计。"""
    groups: dict[str, list[SingleQueryResult]] = {}

    for result in results:
        try:
            group_key = key_fn(result)
        except Exception:
            group_key = "unknown"
        groups.setdefault(group_key, []).append(result)

    stats: dict[str, dict[str, StageMetricSummaryValue]] = {}
    for group_key, group in groups.items():
        size = len(group)
        stats[group_key] = {
            "count": size,
            "hit_rate": round(
                sum(bool(item.stage_metrics[stage_name]["hit"]) for item in group) / size,
                4,
            ),
            "recall_at_k": round(
                sum(float(item.stage_metrics[stage_name]["recall"]) for item in group) / size,
                4,
            ),
        }
        logger.info(
            f"  [{group_key}] count={size} | "
            f"hit_rate={stats[group_key]['hit_rate']:.4f} | "
            f"recall={stats[group_key]['recall_at_k']:.4f}"
        )

    return stats


def _save_result(result: EvalResult, output_path: Path) -> None:
    """将 EvalResult 序列化为 JSON 并保存。"""
    data = asdict(result)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存至: {output_path}")


if __name__ == "__main__":
    run_eval(enable_query_processing=False)
