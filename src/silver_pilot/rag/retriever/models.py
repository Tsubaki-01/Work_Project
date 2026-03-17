"""
模块名称：models
功能描述：RAG 检索流水线的核心数据模型定义。
         使用 Pydantic 和 dataclass 定义各阶段的输入输出结构，
         确保流水线各组件之间的数据契约清晰、类型安全。
"""

from dataclasses import dataclass, field
from enum import StrEnum

# ────────────────────────────────────────────────────────────
# 实体相关
# ────────────────────────────────────────────────────────────


class EntityLabel(StrEnum):
    """与 Neo4j Schema 对齐的医学实体类型枚举。"""

    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    COMPLICATION = "Complication"
    TREATMENT = "Treatment"
    PATHOLOGICAL_TYPE = "PathologicalType"
    DIAGNOSIS = "Diagnosis"
    PATHOPHYSIOLOGY = "Pathophysiology"
    DISEASE_SITE = "DiseaseSite"
    DEPARTMENT = "Department"
    MULTIPLE_GROUPS = "MultipleGroups"
    CAUSE = "Cause"
    PROGNOSTIC_SURVIVAL_TIME = "PrognosticSurvivalTime"
    PROGNOSIS = "Prognosis"
    ATTRIBUTE = "Attribute"
    DISEASE_RATE = "DiseaseRate"
    DRUG = "Drug"
    DRUG_THERAPY = "DrugTherapy"
    ADJUVANT_THERAPY = "AdjuvantTherapy"
    OPERATION = "Operation"
    PATHOGENESIS = "Pathogenesis"
    SYMPTOM_AND_SIGN = "SymptomAndSign"
    TREATMENT_PROGRAMS = "TreatmentPrograms"
    RELATED_DISEASE = "RelatedDisease"
    RELATED_SYMPTOM = "RelatedSymptom"
    CHECK = "Check"
    INFECTIOUS = "Infectious"
    RELATED_TO = "RelatedTo"
    AUXILIARY_EXAMINATION = "AuxiliaryExamination"
    STAGE = "Stage"
    SPREAD_WAY = "SpreadWay"
    TYPE = "Type"
    PRECAUTIONS = "Precautions"
    SUBJECT = "Subject"
    INGREDIENTS = "Ingredients"
    OTC = "OTC"
    ADVERSE_REACTIONS = "AdverseReactions"
    INDICATIONS = "Indications"
    CHECK_SUBJECT = "CheckSubject"
    # 兜底类型：无法明确归类时使用
    UNKNOWN = "Unknown"


@dataclass
class ExtractedEntity:
    """NER 阶段从用户 Query 中抽取的原始实体。"""

    name: str
    label: EntityLabel
    confidence: float = 1.0


@dataclass
class LinkedEntity:
    """
    经过实体链接后的标准化实体。

    如果链接成功（is_linked=True），则 neo4j_name 为图谱中的标准名称，
    可直接用于 Cypher 查询；否则回退到原始名称，仅走向量检索路径。
    """

    original_name: str
    label: EntityLabel
    neo4j_name: str | None = None
    neo4j_id: str | None = None
    similarity_score: float = 0.0
    is_linked: bool = False

    @property
    def best_name(self) -> str:
        """优先返回链接后的标准名称，未链接则返回原始名称。"""
        return self.neo4j_name if self.is_linked and self.neo4j_name else self.original_name


# ────────────────────────────────────────────────────────────
# 查询处理相关
# ────────────────────────────────────────────────────────────


@dataclass
class ProcessedQuery:
    """
    查询处理阶段的完整输出。

    包含重写后的规范化 query、分解后的子问题列表，
    以及从 query 中提取的医学实体。
    """

    original_query: str
    rewritten_query: str
    sub_queries: list[str] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)

    @property
    def all_queries(self) -> list[str]:
        """返回所有需要用于检索的 query（重写后的主 query + 子 query）。"""
        queries = [self.rewritten_query]
        for sq in self.sub_queries:
            if sq != self.rewritten_query:
                queries.append(sq)
        return queries


# ────────────────────────────────────────────────────────────
# 检索结果相关
# ────────────────────────────────────────────────────────────


class RetrievalSource(StrEnum):
    """检索结果的来源标识，用于溯源和重排时的来源感知。"""

    NEO4J_GRAPH = "neo4j_graph"
    MILVUS_QA = "milvus_qa"
    MILVUS_KNOWLEDGE = "milvus_knowledge"


@dataclass
class RetrievalResult:
    """
    单条检索结果的统一抽象。

    无论来自 Neo4j 图谱、Milvus QA 库还是知识库，
    都统一为此结构，便于下游重排和上下文组装。
    """

    content: str
    source: RetrievalSource
    score: float = 0.0
    # 附加元数据：来源文件、实体名称、关系路径等
    metadata: dict = field(default_factory=dict)

    # 重排后的分数（由 Reranker 填充）
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        """优先使用重排分数，否则使用原始检索分数。"""
        return self.rerank_score if self.rerank_score is not None else self.score


# ────────────────────────────────────────────────────────────
# 流水线最终输出
# ────────────────────────────────────────────────────────────


@dataclass
class RetrievalContext:
    """
    RAG 检索流水线的最终输出，供生成模型使用。

    包含组装好的 context 文本、溯源信息，以及流水线各阶段的中间结果（用于调试和评估）。
    """

    # 最终拼装的 context 文本，直接注入 prompt
    context_text: str
    # 经过重排和筛选后的检索结果列表
    ranked_results: list[RetrievalResult] = field(default_factory=list)
    # 流水线中间状态（用于调试、日志、RAGAS 评估）
    processed_query: ProcessedQuery | None = None
    linked_entities: list[LinkedEntity] = field(default_factory=list)
    # 各路检索的原始结果数量统计
    retrieval_stats: dict[str, int] = field(default_factory=dict)
