"""
模块名称：community_builder
功能描述：GraphRAG 社区检测与摘要生成器。

离线阶段（系统初始化时）执行：
1. 从 Neo4j 导出子图到 NetworkX
2. 使用 Leiden 算法进行多层级社区检测
3. 对每个社区调用 LLM 生成摘要文本
4. 将摘要向量化后缓存（内存 / JSON 文件）

运行时阶段：
- 根据用户 query 中的实体快速定位相关社区
- 返回社区摘要供上下文组装使用

依赖：networkx graspologic
"""

import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
from openai import OpenAI
from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.dao import Neo4jManager
from silver_pilot.perception import create_embedder
from silver_pilot.perception.embedder import BaseEmbedder
from silver_pilot.utils import get_channel_logger

from .graph_models import Community

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "community_builder")

# ================= 默认配置 =================
DEFAULT_LLM_MODEL = config.COMMUNITY_BUILDER_MODEL
COMMUNITY_CACHE_DIR = config.DATA_DIR / "rag" / "communities"
MAX_COMMUNITY_SIZE_FOR_SUMMARY = int(
    config.MAX_COMMUNITY_SIZE_FOR_SUMMARY
)  # 超过此节点数的社区跳过摘要
MIN_COMMUNITY_SIZE = int(config.MIN_COMMUNITY_SIZE)  # 小于此节点数的社区忽略
MAX_COMMUNITIES = int(config.MAX_COMMUNITIES)  # 每个查询返回的最大社区数
COMMUNITY_ENTITY_RETRIEVAL_WEIGHT = float(config.COMMUNITY_ENTITY_RETRIEVAL_WEIGHT)  # 实体检索权重
COMMUNITY_QUERY_RETRIEVAL_WEIGHT = float(config.COMMUNITY_QUERY_RETRIEVAL_WEIGHT)  # 查询检索权重


# ────────────────────────────────────────────────────────────
# Pydantic 结构化输出 Schema（供 LLM 解析用）
# ────────────────────────────────────────────────────────────


class CommunitySummaryOutput(BaseModel):
    summary: str = Field(description="社区摘要")
    keywords: list[str] = Field(description="社区关键词")


# ============================================================


class CommunityBuilder:
    """
    GraphRAG 社区检测与摘要生成器。

    使用示例（离线构建）::

        builder = CommunityBuilder()
        communities = builder.build()
        builder.save_cache(communities)

    使用示例（运行时加载）::

        builder = CommunityBuilder()
        builder.load_cache()
        relevant = builder.find_relevant_communities(["阿司匹林", "华法林"])
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager | None = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedder: BaseEmbedder | None = None,
        embedding_backend: str = config.EMBEDDER_MODE,
        api_key: str | None = None,
    ) -> None:
        self.manager = neo4j_manager
        self.llm_model = llm_model
        self.client = OpenAI(
            api_key=api_key or config.DASHSCOPE_API_KEY,
            base_url=config.QWEN_URL[config.QWEN_REGION],
        )
        self.embedder = embedder or create_embedder(embedding_backend)

        # 运行时缓存
        self._communities: list[Community] = []
        self._entity_to_communities: dict[str, list[int]] = defaultdict(
            list
        )  # name -> [community_ids]
        self._id_to_community: dict[int, Community] = {}

        logger.info("CommunityBuilder 初始化完成")

    # ──────────────────────────────────────────────────
    # 离线构建流程
    # ──────────────────────────────────────────────────

    def build(self, resolution: float = 1.0, max_levels: int = 2) -> list[Community]:
        """
        完整的离线构建流程：Neo4j 导出 → 社区检测 → 摘要生成 → 向量化。

        Args:
            resolution: Leiden 算法的分辨率参数，越大社区越小
            max_levels: 最大层级数

        Returns:
            list[Community]: 构建完成的社区列表
        """
        t_start = time.time()
        logger.info("=" * 50)
        logger.info("开始 GraphRAG 社区构建流程")

        # 1. 从 Neo4j 导出图到 NetworkX
        G = self._export_to_networkx()

        if G.number_of_nodes() == 0:
            logger.warning("图谱为空，跳过社区构建")
            return []

        # 2. Leiden 社区检测
        communities = self._detect_communities(G, resolution=resolution)

        # 3. 为每个社区生成 LLM 摘要
        communities = self._generate_summaries(communities, G)

        # 4. 向量化摘要
        communities = self._embed_summaries(communities)

        # 5. 构建反向索引
        self._communities = communities
        # 根据社区列表构建 ID -> Community 映射与实体反向索引
        self._build_id_to_community()
        self._build_entity_index()

        elapsed = time.time() - t_start
        logger.info(f"社区构建完成 | 社区数={len(communities)} | 耗时={elapsed:.1f}s")
        logger.info("=" * 50)
        return communities

    def _export_to_networkx(self) -> nx.Graph:
        """从 Neo4j 导出所有节点和关系到 NetworkX 无向图。"""
        if self.manager is None:
            self.manager = Neo4jManager()

        G = nx.Graph()

        # 导出节点
        batch_size = 50000  # 每次拉取 5 万条
        skip = 0
        try:
            with self.manager.driver.session() as session:
                while True:
                    node_query = (
                        f"MATCH (n) WHERE n.name IS NOT NULL "
                        f"RETURN n.name AS name, labels(n) AS labels "
                        f"SKIP {skip} LIMIT {batch_size}"
                    )
                    records = session.run(node_query)

                    if not records.peek():
                        break

                    for record in records:
                        name = record["name"]
                        labels = record["labels"]
                        label = labels[0] if labels else "Unknown"
                        G.add_node(name, label=label)

                    skip += batch_size
        except Exception as e:
            logger.error(f"导出节点失败: {e}")
            return G

        # 导出关系
        batch_size = 250000  # 每次拉取 25 万条
        skip = 0
        try:
            with self.manager.driver.session() as session:
                while True:
                    edge_query = (
                        "MATCH (a)-[r]->(b) WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
                        "RETURN a.name AS source, b.name AS target, type(r) AS rel_type "
                        f"SKIP {skip} LIMIT {batch_size}"
                    )
                    records = session.run(edge_query)

                    if not records.peek():
                        break

                    for record in records:
                        G.add_edge(
                            record["source"],
                            record["target"],
                            relation=record["rel_type"],
                        )

                    skip += batch_size
        except Exception as e:
            logger.error(f"导出关系失败: {e}")

        logger.info(f"图谱导出完成 | 节点={G.number_of_nodes()} | 边={G.number_of_edges()}")
        return G

    def _detect_communities(self, G: nx.Graph, resolution: float = 1.0) -> list[Community]:
        """
        使用 Leiden 算法进行社区检测。

        回退策略：如果 graspologic 不可用，使用 NetworkX 的 Louvain 算法。
        """
        try:
            from graspologic.partition import leiden

            partition = leiden(G, resolution=resolution)
            logger.info("使用 Leiden 算法进行社区检测")
        except ImportError:
            logger.warning("graspologic 未安装，回退到 NetworkX Louvain 算法")
            from networkx.algorithms.community import louvain_communities

            louvain_result = louvain_communities(G, resolution=resolution)
            partition = {}
            for idx, nodes in enumerate(louvain_result):
                for node in nodes:
                    partition[node] = idx

        # 按社区 ID 分组
        community_nodes: dict[int, list[str]] = defaultdict(list)
        for node, comm_id in partition.items():
            community_nodes[comm_id].append(node)

        communities: list[Community] = []
        for comm_id, nodes in community_nodes.items():
            if len(nodes) < MIN_COMMUNITY_SIZE:
                continue

            # 收集节点标签
            node_labels = []
            for n in nodes:
                data = G.nodes.get(n, {})
                node_labels.append(data.get("label", "Unknown"))

            # 计算社区内部边数
            subgraph = G.subgraph(nodes)
            edge_count = subgraph.number_of_edges()

            communities.append(
                Community(
                    community_id=comm_id,
                    level=0,
                    node_names=nodes,
                    node_labels=node_labels,
                    edge_count=edge_count,
                )
            )

        logger.info(
            f"社区检测完成 | 总社区数={len(communities)} | "
            f"平均社区={sum(c.size for c in communities) / len(communities) if communities else 0} 节点"
            f"最大社区={max(c.size for c in communities) if communities else 0} 节点"
        )
        return communities

    def _generate_summaries(self, communities: list[Community], G: nx.Graph) -> list[Community]:
        """为每个社区调用 LLM 生成摘要文本。"""
        logger.info(f"开始为 {len(communities)} 个社区生成摘要...")

        for i, community in enumerate(communities):
            if community.size > MAX_COMMUNITY_SIZE_FOR_SUMMARY:
                community.summary = (
                    f"大型社区（{community.size}个实体），"
                    f"包含: {', '.join(community.node_names[:10])}等"
                )
                continue

            # 提取社区内的关系信息
            subgraph = G.subgraph(community.node_names)
            edges_text = []
            for u, v, data in subgraph.edges(data=True):
                rel = data.get("relation", "相关")
                edges_text.append(f"{u} --[{rel}]--> {v}")

            # 构建 prompt
            nodes_with_labels = []
            for name, label in zip(
                community.node_names[:50], community.node_labels[:50], strict=True
            ):
                nodes_with_labels.append(f"{name}({label})")

            prompt = (
                f"以下是一组紧密关联的医学实体及其关系，请用2-3句话概括这个知识群组的核心主题和关键信息，并提取6个关键词。\n\n"
                f"实体: {', '.join(nodes_with_labels)}\n\n"
                f"关系（部分）:\n" + "\n".join(edges_text[:30]) + "\n\n"
                "请直接输出json数据，不要有前缀。"
            )

            try:
                response = self.client.chat.completions.parse(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                    response_format=CommunitySummaryOutput,
                )
                parsed: CommunitySummaryOutput | None = response.choices[0].message.parsed
                if parsed is None:
                    logger.warning(f"社区 {community.community_id} 摘要解析失败")
                    continue
                community.summary = parsed.summary
                logger.debug(f"社区 {community.community_id} 摘要: {community.summary}")

                community.keywords = parsed.keywords
                logger.debug(f"社区 {community.community_id} 关键词: {community.keywords[:3]}")

            except Exception as e:
                logger.warning(f"社区 {community.community_id} 摘要生成失败: {e}")
                community.summary = (
                    f"包含 {', '.join(community.node_names[:5])} 等{community.size}个医学实体"
                )

            if (i + 1) % 10 == 0:
                logger.info(f"  摘要进度: {i + 1}/{len(communities)}")

        logger.info("社区摘要生成完成")
        return communities

    def _embed_summaries(self, communities: list[Community]) -> list[Community]:
        """将社区摘要向量化，用于运行时的语义检索。"""
        summaries = [c.summary for c in communities if c.summary]
        if not summaries:
            return communities

        try:
            embeddings = self.embedder.encode(summaries)
            idx = 0
            for community in communities:
                if community.summary:
                    community.summary_embedding = embeddings[idx]
                    idx += 1
            logger.info(f"社区摘要向量化完成 | 数量={len(summaries)}")
        except Exception as e:
            logger.error(f"社区摘要向量化失败: {e}")

        return communities

    def _build_entity_index(self) -> None:
        """构建实体名 → 社区 ID 的反向索引。"""
        self._entity_to_communities.clear()
        for community in self._communities:
            for name in community.node_names:
                self._entity_to_communities[name].append(community.community_id)

    def _build_id_to_community(self) -> None:
        """构建社区 ID → 社区对象的索引。"""
        self._id_to_community.clear()
        self._id_to_community = {c.community_id: c for c in self._communities}

    # ──────────────────────────────────────────────────
    # 运行时查询
    # ──────────────────────────────────────────────────

    def find_relevant_communities(
        self,
        query: str,
        *,
        entity_names: list[str] | None = None,
        query_embedding: list[float] | None = None,
        max_communities: int = MAX_COMMUNITIES,
    ) -> list[Community]:
        """
        根据实体名称和/或 query 语义查找相关社区。

        策略：
        1. 精确匹配：实体名直接命中社区
        2. 语义匹配：用 query embedding 与社区摘要 embedding 比较
        两种分数加权合并后取 top-K。

        Args:
            query: 用户 query
            entity_names: 已链接的实体名称列表（可选，用于语义匹配）
            query_embedding: 用户 query 的向量（可选，用于语义匹配）
            max_communities: 返回的最大社区数

        Returns:
            list[Community]: 按相关性排序的社区列表
        """
        if not self._communities:
            logger.warning("社区缓存为空，请先调用 build() 或 load_cache()")
            return []

        hit_scores: dict[int, float] = {}  # community_id -> score

        # 1. 精确匹配：找到包含任何 query 实体的社区
        if entity_names:
            for name in entity_names:
                comm_ids = self._entity_to_communities.get(name, [])
                for cid in comm_ids:
                    hit_scores[cid] = hit_scores.get(cid, 0.0) + COMMUNITY_ENTITY_RETRIEVAL_WEIGHT

        # 2. 语义匹配：用 query embedding 与社区摘要 embedding 比较
        if query_embedding is None:
            query_embedding = self.embedder.encode_query(query)

        import numpy as np

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        for community in self._communities:
            if not community.summary_embedding:
                continue
            comm_vec = np.array(community.summary_embedding, dtype=np.float32)
            comm_vec = comm_vec / (np.linalg.norm(comm_vec) + 1e-8)
            sim = float(np.dot(query_vec, comm_vec))

            cid = community.community_id
            hit_scores[cid] = hit_scores.get(cid, 0.0) + sim * COMMUNITY_QUERY_RETRIEVAL_WEIGHT

        # 3. 排序取 top-K
        sorted_ids = sorted(hit_scores.keys(), key=lambda x: hit_scores[x], reverse=True)
        top_ids = sorted_ids[:max_communities]

        results = [
            self._id_to_community[cid] for cid in top_ids if cid in self._id_to_community.keys()
        ]

        logger.debug(f"社区检索完成 | 命中={len(results)}")
        return results

    # ──────────────────────────────────────────────────
    # 缓存持久化
    # ──────────────────────────────────────────────────

    def save_cache(
        self, communities: list[Community] | None = None, cache_dir: Path | None = None
    ) -> None:
        """将社区数据序列化为 JSON 缓存文件。"""
        communities = communities or self._communities
        cache_dir = cache_dir or COMMUNITY_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = cache_dir / "communities.json"
        # data = []
        # for c in communities:
        #     data.append(
        #         {
        #             "community_id": c.community_id,
        #             "level": c.level,
        #             "node_names": c.node_names,
        #             "node_labels": c.node_labels,
        #             "edge_count": c.edge_count,
        #             "summary": c.summary,
        #             "summary_embedding": c.summary_embedding,
        #             "keywords": c.keywords,
        #         }
        #     )

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                [dataclasses.asdict(community) for community in communities],
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"社区缓存已保存 | 路径={cache_path} | 数量={len(communities)}")

    def load_cache(self, cache_dir: Path | None = None) -> list[Community]:
        """从 JSON 缓存文件加载社区数据。"""
        cache_dir = cache_dir or COMMUNITY_CACHE_DIR
        cache_path = cache_dir / "communities.json"

        if not cache_path.exists():
            logger.warning(f"社区缓存文件不存在: {cache_path}")
            return []

        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)

        self._communities = []
        for item in data:
            self._communities.append(Community(**item))

        self._build_entity_index()
        self._build_id_to_community()
        logger.info(f"社区缓存已加载 | 数量={len(self._communities)}")
        return self._communities

    def close(self) -> None:
        if self.manager:
            self.manager.close()
