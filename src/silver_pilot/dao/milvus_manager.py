"""
模块名称：milvus_manager
功能描述：Milvus 向量数据库的通用 DAO 管理器，提供集合创建与索引挂载、数据批量插入与
         Upsert、向量相似度检索（支持标量过滤）、标量精确查询以及条件删除等完整的
         CRUD 操作封装。
"""

from pathlib import Path
from typing import Any

from pymilvus import Collection, CollectionSchema, connections, utility

# ================= 导入 =================
from ..config import config
from ..utils import get_channel_logger

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


# ===================================================
# 测试代码
# ===================================================
if __name__ == "__main__":
    import random

    from pymilvus import DataType, FieldSchema

    logger.info("🧪 开始执行 MilvusManager 的本地测试...")

    # 1. 定义测试用例的参数
    TEST_COLLECTION = "test_huatuo_lite"
    VECTOR_DIM = 1024

    try:
        # 2. 实例化通用管理器
        manager = MilvusManager(collection_name=TEST_COLLECTION)

        # 3. 外部定义具体的业务 Schema (适配 huatuo26m-lite)
        fields: list[FieldSchema] = [
            FieldSchema(
                name="qa_id", dtype=DataType.INT64, is_primary=True, description="唯一QA对ID"
            ),
            FieldSchema(
                name="question", dtype=DataType.VARCHAR, max_length=1000, description="患者问题"
            ),
            FieldSchema(
                name="answer", dtype=DataType.VARCHAR, max_length=4000, description="医生回答"
            ),
            FieldSchema(name="score", dtype=DataType.INT32, description="回答质量分数"),
            FieldSchema(
                name="label", dtype=DataType.VARCHAR, max_length=100, description="科室标签"
            ),
            FieldSchema(
                name="question_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTOR_DIM,
                description="问题向量",
            ),
        ]
        schema = CollectionSchema(fields=fields, description="Huatuo26M-Lite 测试集合")

        # 4. 创建集合与索引 (告诉管理器我们要对哪个字段建索引)
        manager.create_collection(schema=schema, index_field_name="question_vector")

        # 5. 准备 Mock 数据 (列式存储格式 Column-based)
        # 模拟生成 3 条测试数据
        mock_ids = [101, 102, 103]
        mock_questions = ["头疼怎么办", "高血压吃什么药", "心脏突然狂跳是怎么回事"]
        mock_answers = [
            "建议多休息，必要时吃布洛芬",
            "建议服用降压药，如氨氯地平",
            "可能是心律失常，建议立即做心电图",
        ]
        mock_scores = [3, 4, 5]
        mock_labels = ["神经内科", "心内科", "心内科"]
        # 模拟 BGE-M3 的 1024 维向量
        mock_vectors = [[random.random() for _ in range(VECTOR_DIM)] for _ in range(3)]

        entities = [mock_ids, mock_questions, mock_answers, mock_scores, mock_labels, mock_vectors]

        # 6. 执行批量插入
        manager.insert_data(entities)

        # 7. 测试：混合检索 (查询向量 + 标量过滤)
        # 模拟用户的提问向量（随机生成 1 个作为 query）
        query_vector = [[random.random() for _ in range(VECTOR_DIM)]]

        # Pre-filtering：只搜索 心内科 且 score >= 4 的高质量回答
        filter_expr = "label == '心内科' and score >= 4"
        output_fields = ["qa_id", "question", "answer", "score", "label"]

        search_res = manager.search_vectors(
            query_vectors=query_vector,
            anns_field="question_vector",
            limit=2,
            expr=filter_expr,
            output_fields=output_fields,
        )

        logger.info("✅ 检索测试成功！输出检索结果：")
        for hits in search_res:
            for hit in hits:
                logger.info(
                    f"匹配 ID: {hit.entity.get('qa_id')}, 距离(距离越近越相似): {hit.distance:.4f}"
                )
                logger.info(f"问题: {hit.entity.get('question')}")
                logger.info(f"回答: {hit.entity.get('answer')} (得分: {hit.entity.get('score')})")
                logger.info("-" * 30)
        # 8. 测试修改 (Upsert) 逻辑
        logger.info("🔄 开始测试 Upsert (修改) 逻辑：更新 qa_id == 102 的回答和分数...")

        # 必须传入完整的一行列式数据（注意：外层依然是按列聚合的 list）
        updated_entities = [
            [102],  # qa_id 保持不变，触发覆盖更新
            ["高血压吃什么药"],  # 问题保持不变
            ["建议服用长效降压药如氨氯地平，并严格控制每日食盐摄入量低于5g。"],  # 回答内容更新了！
            [5],  # score 从 4 更新为 5！
            ["心内科"],  # 标签保持不变
            [[random.random() for _ in range(VECTOR_DIM)]],  # 模拟重新生成的向量
        ]

        manager.upsert_data(updated_entities)

        # 立刻用 query_data 去精准打捞，验证是否真的改掉了
        logger.info("🔎 验证 Upsert 结果：精确查询 qa_id == 102...")
        upsert_verify = manager.query_data(
            expr="qa_id == 102", output_fields=["qa_id", "answer", "score"]
        )

        if upsert_verify and upsert_verify[0].get("score") == 5:
            logger.success(f"🎉 Upsert 验证成功！数据已更新: {upsert_verify[0]}")
        else:
            logger.error(f"❌ Upsert 验证失败！拉取到的数据: {upsert_verify}")
        # 9. 测试单条删除
        delete_target_id = 102
        logger.info(f"🗑️ 开始测试删除逻辑：目标 qa_id == {delete_target_id}")
        manager.delete_by_expr(expr=f"qa_id == {delete_target_id}")

        logger.info(f"🔍 执行第二轮检索：尝试精准打捞刚才删除的 qa_id == {delete_target_id} ...")
        verify_res = manager.query_data(
            expr=f"qa_id == {delete_target_id}", output_fields=["qa_id", "question"]
        )

        if len(verify_res) == 0:
            logger.success(
                f"🎉 验证成功：已无法查询到 qa_id == {delete_target_id} 的数据，删除机制生效！"
            )
        else:
            logger.error(f"❌ 验证失败：竟然还能查到已删除的数据！返回结果: {verify_res}")

    except Exception as e:
        logger.error(f"❌ 测试过程中发生严重错误: {e}")

    finally:
        # 10. 清理测试环境 (保持环境整洁是好习惯)
        logger.info("🧹 开始清理测试环境...")
        if utility.has_collection(TEST_COLLECTION):
            utility.drop_collection(TEST_COLLECTION)
            logger.info(f"♻️ 测试集合 {TEST_COLLECTION} 已删除。")
        logger.info("🏁 测试流程结束。")
