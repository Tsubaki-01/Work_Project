"""
模块名称：graph_retriever
功能描述：GraphRAG 知识图谱检索器。

三层检索架构（由粗到细）：
1. 社区摘要层：回答全局性问题（"老年人高血压有哪些治疗方案"）
2. 推理路径层：回答多跳推理问题（"阿司匹林和华法林能一起吃吗"）
3. 局部事实层：回答直接查询（"阿司匹林的禁忌是什么"）

"""

from silver_pilot.config import config
from silver_pilot.dao import Neo4jManager
from silver_pilot.utils import get_channel_logger

from .community_builder import CommunityBuilder
from .graph_models import SubgraphContext
from .models import LinkedEntity, RetrievalResult, RetrievalSource
from .path_reasoner import PathReasoner

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "graph_retriever")


# ================= 默认配置 =================
DEFAULT_MAX_RESULTS_PER_ENTITY = int(config.GRAPH_RETRIEVER_MAX_RESULTS_PER_ENTITY)
DEFAULT_ENABLE_COMMUNITY = config.GRAPH_RETRIEVER_ENABLE_COMMUNITY
DEFAULT_ENABLE_REASONING = config.GRAPH_RETRIEVER_ENABLE_REASONING

# ────────────────────────────────────────────────────────────
# 实体类型到带有上下文描述的中文映射
# ────────────────────────────────────────────────────────────

RELATION_DESCRIPTIONS: dict[str, str] = {
    "Disease": "的疾病为",
    "Symptom": "的症状包括",
    "Complication": "的并发症包括",
    "Treatment": "的治疗方式包括",
    "PathologicalType": "的病理类型为",
    "Diagnosis": "的诊断方式包括",
    "Pathophysiology": "的病理生理为",
    "DiseaseSite": "的发病部位为",
    "Department": "的就诊科室为",
    "MultipleGroups": "的多发群体为",
    "Cause": "的病因包括",
    "PrognosticSurvivalTime": "的预后生存时间为",
    "Prognosis": "的预后为",
    "Attribute": "的属性为",
    "DiseaseRate": "的发病率为",
    "Drug": "的相关药物包括",
    "DrugTherapy": "的药物治疗包括",
    "AdjuvantTherapy": "的辅助治疗包括",
    "Operation": "可进行的手术包括",
    "Pathogenesis": "的发病机制为",
    "SymptomAndSign": "的症状和体征包括",
    "TreatmentPrograms": "的治疗方案包括",
    "RelatedDisease": "的相关疾病包括",
    "RelatedSymptom": "的相关症状包括",
    "Check": "需要进行的检查包括",
    "Infectious": "的传染性为",
    "RelatedTo": "与{end}相关",
    "AuxiliaryExamination": "的辅助检查包括",
    "Stage": "的分期为",
    "SpreadWay": "的传播途径包括",
    "Type": "的类型包括",
    "Precautions": "的注意事项包括",
    "Subject": "所属的主体/学科为",
    "Ingredients": "的成分包括",
    "OTC": "的非处方药属性为",
    "AdverseReactions": "的不良反应包括",
    "Indications": "的适应症包括",
    "CheckSubject": "的检查项目包括",
    # 兜底类型
    "Unknown": "的未知信息包括",
}


class GraphRetriever:
    """
    GraphRAG 知识图谱检索器：三层结构化检索。

    Layer 1 - 社区摘要 (Global)：
        通过 CommunityBuilder 预构建的社区摘要回答全局性问题。
        适合"老年人常见慢性病有哪些""糖尿病的综合治疗方案"这类跨实体聚合问题。

    Layer 2 - 推理路径 (Reasoning)：
        通过 PathReasoner 在图谱中发现实体间的多跳推理链路。
        适合"阿司匹林和华法林能一起吃吗""糖尿病和高血压有什么关系"这类需要推理的问题。

    Layer 3 - 局部事实 (Local)：
        传统的 1-hop 邻居查询，获取实体的直接属性和关系。
        适合"阿司匹林的禁忌是什么""这个药的不良反应"这类直接查询。

    使用示例::

        retriever = GraphRetriever()
        retriever.initialize()  # 加载社区缓存
        results = retriever.retrieve(linked_entities, query="阿司匹林和华法林能一起吃吗")
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager | None = None,
        max_results_per_entity: int = DEFAULT_MAX_RESULTS_PER_ENTITY,
        enable_community: bool = DEFAULT_ENABLE_COMMUNITY,
        enable_reasoning: bool = DEFAULT_ENABLE_REASONING,
    ) -> None:
        self.manager = neo4j_manager or Neo4jManager()
        self.max_per_entity: int = max_results_per_entity
        self.enable_community = enable_community
        self.enable_reasoning = enable_reasoning

        # GraphRAG 子组件（延迟初始化）
        self._community_builder: CommunityBuilder | None = None
        self._path_reasoner: PathReasoner | None = None

        logger.info(
            f"GraphRetriever 初始化完成 | "
            f"community={enable_community} | reasoning={enable_reasoning}"
        )

    def initialize(self) -> None:
        """
        初始化 GraphRAG 子组件。

        - 加载社区缓存（如果存在）
        - 初始化路径推理器
        """
        if self.enable_community:
            self._community_builder = CommunityBuilder(neo4j_manager=self.manager)
            cached = self._community_builder.load_cache()
            if not cached:
                logger.warning(
                    "社区缓存为空。请先运行离线构建: CommunityBuilder().build() + save_cache()"
                )

        if self.enable_reasoning:
            self._path_reasoner = PathReasoner(neo4j_manager=self.manager)

        logger.info("GraphRetriever GraphRAG 组件初始化完成")

    # ──────────────────────────────────────────────────
    # 核心检索接口
    # ──────────────────────────────────────────────────

    def retrieve(
        self,
        linked_entities: list[LinkedEntity],
        query: str = "",
        query_embedding: list[float] | None = None,
    ) -> list[RetrievalResult]:
        """
        执行三层 GraphRAG 检索，返回统一的 RetrievalResult 列表。

        Args:
            linked_entities: 实体链接阶段的输出
            query: 用户原始查询（用于推理路径评分和社区匹配）
            query_embedding: 查询向量（用于社区语义匹配）

        Returns:
            list[RetrievalResult]: 图谱检索结果列表
        """
        valid_entities = [e for e in linked_entities if e.is_linked and e.neo4j_name]

        if not valid_entities:
            logger.info("无已链接实体，跳过图谱检索")
            return []

        logger.info(f"开始 GraphRAG 三层检索 | 实体数={len(valid_entities)}")

        # 同时构建 SubgraphContext 和 RetrievalResult 列表
        subgraph = SubgraphContext()
        all_results: list[RetrievalResult] = []

        # ── Layer 1: 社区摘要 ──
        if self.enable_community and self._community_builder:
            entity_names = [e.neo4j_name for e in valid_entities if e.neo4j_name]
            communities, scores = self._community_builder.find_relevant_communities(
                entity_names=entity_names,
                query=query,
                query_embedding=query_embedding,
            )
            for comm, score in zip(communities, scores, strict=True):
                if comm.summary:
                    subgraph.community_summaries.append(comm)
                    all_results.append(
                        RetrievalResult(
                            content=f"[医学知识社区摘要] {comm.summary}",
                            source=RetrievalSource.NEO4J_GRAPH,
                            score=score,
                            metadata={
                                "layer": "community",
                                "community_id": comm.community_id,
                                "community_size": comm.size,
                                "community_keywords": comm.keywords[:3],
                            },
                        )
                    )
            subgraph.total_communities = len(communities)

        # ── Layer 2: 多跳推理路径 ──
        if self.enable_reasoning and self._path_reasoner:
            paths = self._path_reasoner.discover_paths(valid_entities, query=query)
            for path in paths:
                subgraph.reasoning_paths.append(path)
                # 优先使用 LLM 生成的自然语言解释，否则用三元组链
                content = path.natural_language or path.to_triplet_chain()
                if content:
                    all_results.append(
                        RetrievalResult(
                            content=f"[推理路径] {content}",
                            source=RetrievalSource.NEO4J_GRAPH,
                            score=path.relevance_score,
                            metadata={
                                "layer": "reasoning_path",
                                "hops": path.hops,
                                "path_signature": path.path_signature,
                                "triplet_chain": path.to_triplet_chain(),
                            },
                        )
                    )
            subgraph.total_paths = len(paths)

        # ── Layer 3: 局部事实（1-hop）──
        for entity in valid_entities:
            one_hop = self._query_one_hop(entity)
            for result in one_hop:
                result.metadata["layer"] = "local_fact"
            all_results.extend(one_hop)
            subgraph.local_facts.extend([r.content for r in one_hop])

        subgraph.total_local_facts = len(subgraph.local_facts)

        logger.info(
            f"GraphRAG 检索完成 | "
            f"communities={subgraph.total_communities} | "
            f"paths={subgraph.total_paths} | "
            f"local_facts={subgraph.total_local_facts} | "
            f"total_results={len(all_results)}"
        )
        return all_results

    # ──────────────────────────────────────────────────
    # Layer 3: 局部事实查询
    # ──────────────────────────────────────────────────

    def _query_one_hop(self, entity: LinkedEntity) -> list[RetrievalResult]:
        """查询实体的所有直接关系（1-hop 邻居）。"""
        label = entity.label.value
        name = entity.neo4j_name

        query = f"""
        MATCH (n:`{label}` {{name: $name}})-[r]->(m)
        RETURN type(r) AS rel_type, m.name AS target_name, labels(m) AS target_labels
        LIMIT $limit
        """

        results: list[RetrievalResult] = []

        try:
            with self.manager.driver.session() as session:
                records = session.run(query, name=name, limit=self.max_per_entity)

                for record in records:
                    rel_type = record["rel_type"]
                    target_name = record["target_name"]
                    nl_text = self._triplet_to_text(name, rel_type, target_name)

                    results.append(
                        RetrievalResult(
                            content=nl_text,
                            source=RetrievalSource.NEO4J_GRAPH,
                            score=1.0,
                            metadata={
                                "start_entity": name,
                                "relation": rel_type,
                                "end_entity": target_name,
                                "hop": 1,
                            },
                        )
                    )
        except Exception as e:
            logger.error(f"1-hop 查询失败 | entity={name} | error={e}")

        return results

    @staticmethod
    def _triplet_to_text(start_name: str, rel_type: str, end_name: str) -> str:
        """将三元组转化为自然语言描述。"""
        desc_template = RELATION_DESCRIPTIONS.get(rel_type)
        if desc_template:
            if "{end}" in desc_template:
                return start_name + desc_template.format(end=end_name)
            return start_name + desc_template + end_name
        else:
            return f"{start_name} --[{rel_type}]--> {end_name}"

    def close(self) -> None:
        """关闭所有连接。"""
        if self._community_builder:
            self._community_builder.close()
        if self._path_reasoner:
            self._path_reasoner.close()
        self.manager.close()
