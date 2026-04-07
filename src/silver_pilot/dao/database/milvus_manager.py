"""
模块名称：milvus_manager
功能描述：Milvus 向量数据库的通用 DAO 管理器，提供集合创建与索引挂载、数据批量插入与
         Upsert、向量相似度检索（支持标量过滤）、标量精确查询以及条件删除等完整的
         CRUD 操作封装。
"""

from pathlib import Path
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    RRFRanker,
    WeightedRanker,
    connections,
    utility,
)

# ================= 导入 =================
from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志与配置初始化 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "milvus_logs"
logger = get_channel_logger(LOG_FILE_DIR, "milvus_manager")
# ===================================================


class MilvusManager:
    """
    通用的 Milvus 向量数据库 DAO 管理器。
    """

    def __init__(
        self,
        collection_name: str,
        host: str | None = config.MILVUS_HOST,
        port: str | None = config.MILVUS_PORT,
    ) -> None:
        """
        初始化 DAO 实例，绑定具体的集合名称，并自动建立连接。

        :param collection_name: 集合(表)名称
        :param host: Milvus 服务地址
        :param port: Milvus 服务端口
        """
        self.collection_name: str = collection_name
        self.collection: Collection | None = None

        self._connect(host, port)

        # 如果集合已存在，自动挂载并 Load 到内存
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"✅ 检测到已存在 Collection: {self.collection_name}，已自动挂载。")

    def _connect(self, host: str | None, port: str | None) -> None:
        """建立数据库底层连接"""
        try:
            if host is None or port is None:
                raise ValueError("Milvus host 或 port 未配置")
            connections.connect("default", host=host, port=port)
            logger.info(f"🔗 成功连接至 Milvus 服务 -> {host}:{port}")
        except Exception as e:
            logger.error(f"❌ 连接 Milvus 失败: {e}")
            raise

    def create_collection(
        self,
        schema: CollectionSchema,
        index_field_name: str,
        index_params: dict[str, Any] | None = None,
    ) -> Collection:
        """
        根据外部传入的 Schema 动态创建集合并挂载索引。

        :param schema: 外部定义好的 CollectionSchema
        :param index_field_name: 需要挂载向量索引的字段名 (例如 "question_vector" 或 "doc_vector")
        :param index_params: 索引构建参数。如果不传，默认使用 HNSW 余弦相似度索引。
        :return: 创建好的 Collection 实例
        """
        if utility.has_collection(self.collection_name):
            logger.warning(f"⚠️ 集合 {self.collection_name} 已存在，跳过创建。")
            return Collection(self.collection_name)

        logger.info(f"🛠️ 开始创建集合: {self.collection_name} ...")

        # 1. 动态创建 Collection
        self.collection = Collection(name=self.collection_name, schema=schema)

        # 2. 配置默认索引参数
        if index_params is None:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            }

        # 3. 为指定的向量字段创建索引
        self.collection.create_index(field_name=index_field_name, index_params=index_params)
        logger.success(
            f"🚀 集合 {self.collection_name} 初始化成功，已为字段 '{index_field_name}' 挂载索引！"
        )
        return self.collection

    def insert_data(self, entities: list[list[Any]]) -> Any:
        """
        增：批量插入数据。

        :param entities: 列式数据列表，按 schema 定义的顺序排列。
        :return: 插入结果对象
        """
        if not self.collection:
            raise ValueError("Collection 未初始化，请先调用 create_collection。")

        try:
            insert_result = self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"📥 成功插入并持久化数据，本次影响行数: {insert_result.insert_count}")
            return insert_result
        except Exception as e:
            logger.error(f"❌ 插入数据失败: {e}")
            raise

    def upsert_data(self, entities: list[list[Any]]) -> Any:
        """
        改 (Upsert)：更新或插入数据。
        如果主键 (如 qa_id) 已存在，则覆盖更新该条目的所有标量和向量字段；
        如果主键不存在，则作为新数据插入。

        :param entities: 列式数据列表，必须包含完整字段，按 schema 定义的顺序排列。
        :return: 插入/更新结果对象
        """
        if not self.collection:
            raise ValueError("Collection 未初始化，请先调用 create_collection。")

        try:
            logger.info("🔄 开始执行 Upsert (更新/插入) 操作...")
            upsert_result = self.collection.upsert(entities)
            self.collection.flush()

            logger.info(f"✅ Upsert 成功持久化落盘，本次影响行数: {upsert_result.upsert_count}")
            return upsert_result
        except Exception as e:
            logger.error(f"❌ Upsert 数据失败: {e}")
            raise

    def search_vectors(
        self,
        query_vectors: list[list[float]],
        anns_field: str,
        limit: int = 5,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        查：通用相似度检索，支持标量过滤。

        :param query_vectors: 查询的向量列表
        :param anns_field: 要进行相似度搜索的向量字段名
        :param limit: Top-K 召回数量
        :param expr: 标量过滤表达式 (如 "score > 3")
        :param output_fields: 需要随结果返回的字段列表
        :param search_params: 搜索控制参数 (如果不传则使用默认的 ef 探测深度)
        :return: 检索结果列表
        """
        if not self.collection:
            raise ValueError("Collection 未初始化。")

        # 确保集合在内存中
        self.collection.load()

        if search_params is None:
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64},
            }

        try:
            logger.debug(f"🔍 发起检索, 目标字段='{anns_field}', TopK={limit}, expr={expr}")
            results = self.collection.search(
                data=query_vectors,
                anns_field=anns_field,
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=output_fields,
            )
            return results
        except Exception as e:
            logger.error(f"❌ 检索失败: {e}")
            raise

    def query_data(self, expr: str, output_fields: list[str] | None = None) -> list[Any]:
        """
        查：基于标量表达式进行精确查询 (Exact Match)，不走向量计算。
        常用于精准打捞、存在性验证或数据对齐。

        :param expr: 标量过滤表达式，例如 "qa_id == 102"
        :param output_fields: 需要返回的字段列表
        :return: 包含字典结果的列表
        """
        if not self.collection:
            raise ValueError("Collection 未初始化。")

        # 查询前需确保集合已加载
        self.collection.load()
        if output_fields is None:
            output_fields = ["qa_id"]

        try:
            logger.debug(f"🔎 发起标量精确查询, expr={expr}")
            # query 方法直接返回 list[dict]
            results = self.collection.query(expr=expr, output_fields=output_fields)
            return results
        except Exception as e:
            logger.error(f"❌ 查询失败: {e}")
            raise

    def delete_by_expr(self, expr: str) -> Any:
        """
        删：根据标量表达式删除实体。
        """
        if not self.collection:
            raise ValueError("Collection 未初始化。")

        try:
            res = self.collection.delete(expr=expr)
            self.collection.flush()
            logger.info(f"🗑️ 成功执行删除操作，条件: {expr}")
            return res
        except Exception as e:
            logger.error(f"❌ 删除操作失败 (expr={expr}): {e}")
            raise

    def hybrid_search(
        self,
        reqs: list[AnnSearchRequest],
        rerank: RRFRanker | WeightedRanker | None = None,
        limit: int = 5,
        output_fields: list[str] | None = None,
    ) -> list[Any]:
        """
        混合检索：同时执行多路 ANN 检索（如 dense + BM25 sparse），
        用 Reranker 策略融合多路结果后返回 Top-K。

        :param reqs: AnnSearchRequest 列表，每个请求对应一路检索
        :param rerank: 融合策略，默认 RRFRanker(k=60)
        :param limit: 最终返回的 Top-K 数量
        :param output_fields: 需要随结果返回的字段列表
        :return: 融合后的检索结果
        """
        if not self.collection:
            raise ValueError("Collection 未初始化。")

        self.collection.load()

        if rerank is None:
            rerank = RRFRanker(k=60)

        try:
            results = self.collection.hybrid_search(
                reqs=reqs,
                rerank=rerank,
                limit=limit,
                output_fields=output_fields,
            )
            return results
        except Exception as e:
            logger.error(f"❌ 混合检索失败: {e}")
            raise
