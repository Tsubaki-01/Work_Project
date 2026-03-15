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
