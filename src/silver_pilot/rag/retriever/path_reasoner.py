"""
模块名称：path_reasoner
功能描述：GraphRAG 多跳推理路径发现器。

核心能力：
1. 在 Neo4j 中发现两个实体之间的所有 K-hop 路径
2. 对路径进行相关性评分和去冗余
3. 调用 LLM 将结构化路径转化为自然语言推理链
4. 支持单实体放射状探索（从一个实体出发的关键路径）

"""

from openai import OpenAI
from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.dao import Neo4jManager
from silver_pilot.utils import get_channel_logger

from .graph_models import GraphEdge, GraphNode, ReasoningPath
from .models import LinkedEntity

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "path_reasoner")

# ================= 默认配置 =================
DEFAULT_LLM_MODEL = config.PATH_REASONER_MODEL
DEFAULT_MAX_HOPS = int(config.PATH_REASONER_MAX_HOPS)
DEFAULT_MAX_PATHS = int(config.PATH_REASONER_MAX_PATHS)


# ────────────────────────────────────────────────────────────
# Pydantic 结构化输出 Schema（供 LLM 解析用）
# ────────────────────────────────────────────────────────────


class PathItem(BaseModel):
    id: int = Field(description="id")
    natural_language: str = Field(description="自然语言描述路径的医学含义")
    relevance_score: float = Field(description="路径与用户查询的相关性分数")


class ExplainAndScoreOutput(BaseModel):
    paths: list[PathItem] = Field(description="路径列表")


class PathReasoner:
    """
    多跳推理路径发现器。

    核心思想：用户问"糖尿病患者能吃阿司匹林吗"，系统不是分别查两个实体，
    而是在图上找到 糖尿病 → ... → 阿司匹林 的所有路径，
    这些路径本身就是推理链——每一条都是一个"为什么能/不能"的论据。

    使用示例::

        reasoner = PathReasoner()

        # 两实体间的推理路径
        paths = reasoner.find_paths_between(entity_a, entity_b, max_hops=3)

        # 单实体放射状探索
        paths = reasoner.explore_entity(entity, max_hops=2)

        # 批量发现（自动处理所有实体对）
        paths = reasoner.discover_paths(linked_entities, query="...")
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager | None = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        api_region: str | None = None,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_paths: int = DEFAULT_MAX_PATHS,
    ) -> None:
        self.manager = neo4j_manager or Neo4jManager()
        self.max_hops = max_hops
        self.max_paths = max_paths
        self.llm_model = llm_model
        self.api_region = api_region or config.QWEN_REGION
        self.client = OpenAI(
            api_key=api_key or config.DASHSCOPE_API_KEY,
            base_url=config.QWEN_URL[self.api_region],
        )
        logger.info(f"PathReasoner 初始化完成 | max_hops={max_hops} | max_paths={max_paths}")

    # ──────────────────────────────────────────────────
    # 核心接口：批量路径发现
    # ──────────────────────────────────────────────────

    def discover_paths(
        self,
        linked_entities: list[LinkedEntity],
        query: str = "",
    ) -> list[ReasoningPath]:
        """
        智能路径发现：根据实体数量自动选择策略。

        - 2+ 个实体：查找所有实体对之间的路径
        - 1 个实体：从该实体出发做放射状探索
        - 0 个实体：返回空

        Args:
            linked_entities: 已链接的实体列表
            query: 用户原始查询（用于路径相关性评分和 NL 生成）

        Returns:
            list[ReasoningPath]: 发现的推理路径列表（已去重、评分、排序）
        """
        valid = [e for e in linked_entities if e.is_linked and e.neo4j_name]

        if len(valid) == 0:
            return []

        all_paths: list[ReasoningPath] = []

        if len(valid) >= 2:
            # 两两配对查找路径
            seen_pairs: set[tuple[str, str]] = set()
            for i, entity_a in enumerate(valid):
                for entity_b in valid[i + 1 :]:
                    pair_key = tuple(sorted([entity_a.neo4j_name or "", entity_b.neo4j_name or ""]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    paths = self.find_paths_between(entity_a, entity_b, max_hops=self.max_hops)
                    all_paths.extend(paths)

        # 无论几个实体，都对每个实体做放射状探索（补充单实体信息）
        for entity in valid:
            explore_paths = self.explore_entity(
                entity, max_hops=self.max_hops, max_paths=self.max_paths
            )
            all_paths.extend(explore_paths)

        # 去重
        all_paths = self._deduplicate_paths(all_paths)

        # 用 LLM 生成自然语言解释并评分
        if all_paths and query:
            all_paths = self._explain_and_score(all_paths, query)

        # 按相关性排序，截断
        all_paths.sort(key=lambda p: p.relevance_score, reverse=True)
        all_paths = all_paths[: self.max_paths]

        logger.info(f"路径发现完成 | 总路径数={len(all_paths)}")
        return all_paths

    # ──────────────────────────────────────────────────
    # 两实体间路径查找
    # ──────────────────────────────────────────────────

    def find_paths_between(
        self,
        entity_a: LinkedEntity,
        entity_b: LinkedEntity,
        max_hops: int | None = None,
    ) -> list[ReasoningPath]:
        """
        在图谱中查找两个实体之间的所有最短路径（最多 K 跳）。

        使用 Neo4j 的 shortestPath 和 allShortestPaths 实现。
        """
        name_a = entity_a.neo4j_name
        name_b = entity_b.neo4j_name
        hops = max_hops or self.max_hops

        if not name_a or not name_b:
            return []

        query = f"""
        MATCH path = allShortestPaths(
            (a {{name: $name_a}})-[*..{hops}]-(b {{name: $name_b}})
        )
        RETURN [n IN nodes(path) | {{name: n.name, labels: labels(n)}}] AS path_nodes,
               [r IN relationships(path) | {{
                   source: startNode(r).name,
                   target: endNode(r).name,
                   type: type(r)
               }}] AS path_edges
        LIMIT 10
        """

        paths: list[ReasoningPath] = []

        try:
            with self.manager.driver.session() as session:
                records = session.run(query, name_a=name_a, name_b=name_b)

                for record in records:
                    raw_nodes = record["path_nodes"]
                    raw_edges = record["path_edges"]

                    nodes = [
                        GraphNode(
                            name=n["name"],
                            label=n["labels"][0] if n["labels"] else "Unknown",
                        )
                        for n in raw_nodes
                    ]
                    edges = [
                        GraphEdge(
                            source=e["source"],
                            target=e["target"],
                            relation=e["type"],
                        )
                        for e in raw_edges
                    ]

                    paths.append(
                        ReasoningPath(
                            nodes=nodes,
                            edges=edges,
                            hops=len(edges),
                        )
                    )

        except Exception as e:
            logger.error(f"路径查找失败 | {name_a} <-> {name_b} | error={e}")

        logger.debug(f"找到 {len(paths)} 条路径 | {name_a} <-> {name_b}")
        return paths

    # ──────────────────────────────────────────────────
    # 单实体放射状探索
    # ──────────────────────────────────────────────────

    def explore_entity(
        self,
        entity: LinkedEntity,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_paths: int = DEFAULT_MAX_PATHS,
    ) -> list[ReasoningPath]:
        """
        从单个实体出发，探索其最关键的多跳路径。

        策略：查找该实体的 2-hop 子图中"信息量最大"的路径，
        优先保留涉及多种关系类型的路径（信息多样性）。
        """
        name = entity.neo4j_name
        label = entity.label.value

        if not name:
            return []

        query = f"""
        MATCH path = (a:`{label}` {{name: $name}})-[*1..{max_hops}]->(end_node)
        WHERE end_node.name IS NOT NULL AND end_node <> a
        WITH path,
             [n IN nodes(path) | n.name] AS node_names,
             [r IN relationships(path) | type(r)] AS rel_types
        RETURN [n IN nodes(path) | {{name: n.name, labels: labels(n)}}] AS path_nodes,
               [r IN relationships(path) | {{
                   source: startNode(r).name,
                   target: endNode(r).name,
                   type: type(r)
               }}] AS path_edges,
               size(apoc.coll.toSet(rel_types)) AS diversity
        ORDER BY diversity DESC, length(path) DESC
        LIMIT $limit
        """

        # 回退查询（不依赖 APOC）
        fallback_query = f"""
        MATCH path = (a:`{label}` {{name: $name}})-[*1..{max_hops}]->(end_node)
        WHERE end_node.name IS NOT NULL AND end_node <> a
        RETURN [n IN nodes(path) | {{name: n.name, labels: labels(n)}}] AS path_nodes,
               [r IN relationships(path) | {{
                   source: startNode(r).name,
                   target: endNode(r).name,
                   type: type(r)
               }}] AS path_edges
        LIMIT $limit
        """

        paths: list[ReasoningPath] = []

        try:
            with self.manager.driver.session() as session:
                try:
                    records = session.run(query, name=name, limit=max_paths * 2)
                except Exception:
                    logger.debug("APOC 不可用，使用回退查询")
                    records = session.run(fallback_query, name=name, limit=max_paths * 2)

                for record in records:
                    raw_nodes = record["path_nodes"]
                    raw_edges = record["path_edges"]

                    nodes = [
                        GraphNode(
                            name=n["name"],
                            label=n["labels"][0] if n["labels"] else "Unknown",
                        )
                        for n in raw_nodes
                    ]
                    edges = [
                        GraphEdge(
                            source=e["source"],
                            target=e["target"],
                            relation=e["type"],
                        )
                        for e in raw_edges
                    ]

                    paths.append(
                        ReasoningPath(
                            nodes=nodes,
                            edges=edges,
                            hops=len(edges),
                        )
                    )

        except Exception as e:
            logger.error(f"实体探索失败 | entity={name} | error={e}")

        return paths[:max_paths]

    # ──────────────────────────────────────────────────
    # LLM 路径解释与评分
    # ──────────────────────────────────────────────────

    def _explain_and_score(self, paths: list[ReasoningPath], query: str) -> list[ReasoningPath]:
        """
        对每条推理路径调用 LLM：
        1. 将结构化路径翻译为自然语言推理链
        2. 评估路径与用户 query 的相关性 (0.0~1.0)

        为了减少 LLM 调用次数，所有路径放在一次请求中批量处理。
        """
        if not paths:
            return paths

        # 构建批量 prompt
        path_descriptions = []
        for i, path in enumerate(paths):
            chain = path.to_triplet_chain()
            path_descriptions.append(f"路径{i + 1}: {chain}")

        paths_text = "\n".join(path_descriptions)

        prompt = (
            f"用户问题: {query}\n\n"
            f"以下是从医学知识图谱中发现的推理路径，每条路径是一系列实体和关系的链接：\n\n"
            f"{paths_text}\n\n"
            f"请对每条路径：\n"
            f"1. 用一句通顺的中文解释这条路径的医学含义\n"
            f"2. 评估它与用户问题的相关性（0.0到1.0之间的小数）\n\n"
            f"请严格按以下 JSON 格式输出，不要有其他内容(id从0开始)：\n"
            f'{{"paths": [{{"id": 0, "natural_language": "...", "relevance_score": 0.8}}, ...]}}'
        )

        try:
            logger.info("开始路径解释与评分")
            response = self.client.chat.completions.parse(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000,
                response_format=ExplainAndScoreOutput,
            )
            content: ExplainAndScoreOutput | None = response.choices[0].message.parsed
            if content is None:
                raise ValueError("LLM 返回内容为空")
            for item in content.paths:
                idx = item.id
                if 0 <= idx < len(paths):
                    paths[idx].natural_language = item.natural_language
                    paths[idx].relevance_score = item.relevance_score

        except Exception as e:
            logger.warning(f"路径解释与评分失败: {e}，使用默认分数")
            for path in paths:
                path.natural_language = path.to_triplet_chain()
                path.relevance_score = 0.5
        logger.info(f"路径解释与评分完成 | 总路径数={len(paths)}")
        return paths

    # ──────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_paths(paths: list[ReasoningPath]) -> list[ReasoningPath]:
        """基于路径签名去重。"""
        seen: set[str] = set()
        unique: list[ReasoningPath] = []
        for path in paths:
            sig = path.path_signature
            if sig not in seen:
                seen.add(sig)
                unique.append(path)
        return unique

    def close(self) -> None:
        self.manager.close()
