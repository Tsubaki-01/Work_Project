"""
模块名称：graph_models
功能描述：GraphRAG 专用数据模型，定义社区（Community）、推理路径（ReasoningPath）、
         子图上下文（SubgraphContext）等核心结构，供社区检测、多跳推理和
         图谱感知上下文组装使用。
"""

from dataclasses import dataclass, field

# ────────────────────────────────────────────────────────────
# 图谱基础元素
# ────────────────────────────────────────────────────────────


@dataclass
class GraphNode:
    """图谱中的一个节点。"""

    name: str
    label: str  # 节点类型，如 "Drug", "Disease"
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """图谱中的一条关系边。"""

    source: str  # 起始节点 name
    target: str  # 目标节点 name
    relation: str  # 关系类型，如 "contraindications"
    source_label: str = ""
    target_label: str = ""
    properties: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 推理路径
# ────────────────────────────────────────────────────────────


@dataclass
class ReasoningPath:
    """
    从图谱中发现的一条推理路径。

    示例：
        阿司匹林 --[adverseReactions]--> 胃肠道出血 <--[complication]-- 消化性溃疡
        路径语义：阿司匹林的不良反应是胃肠道出血，而胃肠道出血是消化性溃疡的并发症

    属性：
        nodes: 路径上的有序节点列表
        edges: 路径上的有序边列表
        hops: 跳数
        natural_language: LLM 生成的自然语言解释（延迟填充）
        relevance_score: 路径与用户 query 的相关性分数
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    hops: int = 0
    natural_language: str = ""
    relevance_score: float = 0.0

    @property
    def path_signature(self) -> str:
        """路径的唯一签名，用于去重。"""
        node_names = [n.name for n in self.nodes]
        return " -> ".join(node_names)

    def to_triplet_chain(self) -> str:
        """将路径转化为三元组链式表示。"""
        if not self.edges:
            return ""
        parts = []
        for edge in self.edges:
            parts.append(f"({edge.source})-[{edge.relation}]->({edge.target})")
        return " | ".join(parts)


# ────────────────────────────────────────────────────────────
# 社区
# ────────────────────────────────────────────────────────────


@dataclass
class Community:
    """
    图谱中通过社区检测算法发现的实体聚类。

    每个社区包含一组紧密关联的实体，以及由 LLM 生成的摘要文本。
    摘要在离线阶段（系统启动或定期更新时）生成并缓存。

    属性：
        community_id: 社区唯一标识
        level: 层级（0=最细粒度，越大越粗）
        node_names: 社区内的节点名称列表
        node_labels: 社区内各节点的类型
        edge_count: 社区内部的边数
        summary: LLM 生成的社区摘要文本
        summary_embedding: 摘要的向量表示（用于检索）
        keywords: 从摘要中提取的关键词（用于快速匹配）
    """

    community_id: int
    level: int = 0
    node_names: list[str] = field(default_factory=list)
    node_labels: list[str] = field(default_factory=list)
    edge_count: int = 0
    summary: str = ""
    summary_embedding: list[float] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.node_names)


# ────────────────────────────────────────────────────────────
# 子图上下文（GraphRAG 的核心输出）
# ────────────────────────────────────────────────────────────


@dataclass
class SubgraphContext:
    """
    GraphRAG 从图谱中提取的结构化上下文。

    与普通 RAG 的扁平文本列表不同，SubgraphContext 保留了
    知识的结构化组织：推理路径有先后顺序，社区摘要有层级关系，
    事实有确定性保证。

    这个结构会被 ContextBuilder 的图谱感知模式使用，
    按照"推理路径 → 社区摘要 → 局部事实"的层次组装 context。
    """

    # 多跳推理路径
    reasoning_paths: list[ReasoningPath] = field(default_factory=list)
    # 相关社区的摘要
    community_summaries: list[Community] = field(default_factory=list)
    # 局部事实
    local_facts: list[str] = field(default_factory=list)
    # 统计信息
    total_paths: int = 0
    total_communities: int = 0
    total_local_facts: int = 0
