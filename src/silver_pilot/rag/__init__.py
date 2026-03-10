"""
模块名称：rag
功能描述：RAG 知识库构建模块，涵盖文档解析、分块、入库等能力。
"""

from .chunk_builder import ChunkBuilder

__all__ = [
    "ChunkBuilder",
]
