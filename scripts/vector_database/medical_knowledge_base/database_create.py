from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, Function, FunctionType

# ================= 导入项目配置与基建 =================
from silver_pilot.dao import MilvusManager

# ===================================================

db_manager = MilvusManager(collection_name="medical_knowledge_base")
vector_dim = 1024

fields: list[FieldSchema] = [
    # ── 主键 ────────────────────────────────────────────
    FieldSchema(
        name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False
    ),
    # ── 向量 ────────────────────────────────────────────
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    # ── 核心文本 ─────────────────────────────────────────
    FieldSchema(
        name="content",
        dtype=DataType.VARCHAR,
        max_length=16384,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    ),
    # ── 高频过滤字段 ──────────────────────────────────────
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512, nullable=True),
    FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64, nullable=True),
    FieldSchema(name="group_name", dtype=DataType.VARCHAR, max_length=256, nullable=True),
    # ── 位置字段 ─────────────────────────────────────────
    FieldSchema(name="sub_index", dtype=DataType.INT32, nullable=True),
    # ── 溯源 ─────────────────────────────────────────────
    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=1024),
    # ── 原生 JSON 扩展字段 ────────────────────────────────
    FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
    # ── 时间戳 ───────────────────────────────────────────
    FieldSchema(name="created_at", dtype=DataType.INT64),
    # ── BM25 稀疏向量 ────────────────────────────────────
    FieldSchema(name="content_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
]

schema = CollectionSchema(
    fields=fields, description="医学文档 RAG 知识库", enable_dynamic_field=True
)

# 添加 BM25 数据转换函数
bm25_function = Function(
    name="kb_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["content"],
    output_field_names=["content_sparse"],
)
schema.add_function(bm25_function)

# 创建集合，并为主向量创建索引
db_manager.create_collection(schema=schema, index_field_name="embedding")

# 为稀疏向量创建倒排索引
if isinstance(db_manager.collection, Collection):
    db_manager.collection.create_index(
        field_name="content_sparse",
        index_params={
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX",
        },
    )
print("✅ QA Collection Schema and Indexes created successfully!")
