"""
模块名称：retriever
功能描述：RAG 混合检索子包，提供 GraphRAG + 向量多路召回的完整检索流水线。

核心组件：
- QueryProcessor: 查询重写 + 分解 + NER
- EntityLinker: 运行时实体链接
- GraphRetriever: Neo4j 知识图谱检索
- VectorRetriever: Milvus 双集合向量检索
- Reranker: 检索结果重排序
- ContextBuilder: 上下文组装与压缩
- RAGPipeline: 流水线编排器（统一入口）

使用示例::

    from silver_pilot.rag.retriever import RAGPipeline, PipelineConfig

    config = PipelineConfig(reranker_backend="qwen", context_mode="direct")
    pipeline = RAGPipeline(config)
    pipeline.initialize()

    result = pipeline.retrieve("阿司匹林和华法林能一起吃吗")
    print(result.context_text)
"""

from .community_builder import CommunityBuilder
from .context_builder import ContextBuilder
from .entity_linker import EntityLinker
from .graph_models import Community, GraphEdge, GraphNode, ReasoningPath, SubgraphContext
from .graph_retriever import GraphRetriever
from .models import (
    ExtractedEntity,
    LinkedEntity,
    ProcessedQuery,
    RetrievalContext,
    RetrievalResult,
    RetrievalSource,
)
from .path_reasoner import PathReasoner
from .pipeline import PipelineConfig, RAGPipeline
from .query_processor import QueryProcessor
from .reranker import BaseReranker, BGEReranker, QwenReranker, create_reranker
from .vector_retriever import VectorRetriever

__all__ = [
    # 流水线
    "RAGPipeline",
    "PipelineConfig",
    # 各阶段组件
    "QueryProcessor",
    "EntityLinker",
    "GraphRetriever",
    "VectorRetriever",
    "ContextBuilder",
    # GraphRAG 专用组件
    "CommunityBuilder",
    "PathReasoner",
    # 重排序
    "BaseReranker",
    "BGEReranker",
    "QwenReranker",
    "create_reranker",
    # 数据模型
    "ProcessedQuery",
    "ExtractedEntity",
    "LinkedEntity",
    "RetrievalResult",
    "RetrievalContext",
    "RetrievalSource",
    # GraphRAG 数据模型
    "GraphNode",
    "GraphEdge",
    "ReasoningPath",
    "Community",
    "SubgraphContext",
]
