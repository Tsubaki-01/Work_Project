"""
模块名称: huatuo_ingestion.py
功能描述: Huatuo26M-Lite 数据集的高吞吐向量化与入库流水线
"""

import os
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from datasets import load_dataset
from FlagEmbedding import FlagModel
from pydantic import BaseModel, Field, ValidationError
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, Function, FunctionType
from tqdm import tqdm

# ================= 导入项目配置与基建 =================
from silver_pilot.config import config
from silver_pilot.dao import MilvusManager
from silver_pilot.utils import get_channel_logger

# 初始化日志
LOG_FILE_DIR: Path = config.LOG_DIR / "Huatuo_pipline_logs"
logger = get_channel_logger(LOG_FILE_DIR, "HuatuoPipeline")
# ===================================================

DATABASE_JSON_PATH: Path = config.DATA_DIR / "raw/databases/milvus/huatuo_lite.jsonl"
DLQ_PATH: Path = config.TMP_DIR / "milvus/huatuo_dlq.jsonl"


# 1. 定义严格的数据清洗和映射模型 (Data Validation Model)
class HuatuoQAModel(BaseModel):
    """
    接收 HuggingFace 原始数据，并在实例化时完成校验与默认值清洗
    """

    qa_id: int = Field(alias="id", description="原始数据的 id 映射为 qa_id")
    question_text: str = Field(alias="question", min_length=2, max_length=1000)
    answer_text: str = Field(alias="answer", min_length=2, max_length=4000)
    score: int = Field(default=3, description="质量分数")
    department: str = Field(alias="label", default="通用", max_length=100)
    source: str = Field(default="huatuo26m-lite")

    # Pydantic V2 配置：允许通过属性名或 alias 传参
    class Config:
        populate_by_name = True


def write_to_dlq(path: Path | str, entities: list[dict[str, Any]], error_msg: str) -> None:
    """写入死信队列，确保目录存在"""
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 写入DLQ
    with open(path, "a", encoding="utf-8") as f:
        import json

        # 用JSON格式写入，便于后续解析
        f.write(json.dumps({"error": error_msg, "data": entities}, ensure_ascii=False) + "\n")


class HuatuoDataPipeline:
    """
    Huatuo26M-Lite 数据集处理与灌库流水线
    """

    def __init__(
        self,
        collection_name: str = "medical_qa_lite",
        vector_dim: int = 1024,
        batch_size: int = 500,
    ) -> None:
        self.batch_size: int = batch_size
        self.vector_dim: int = vector_dim

        # 初始化 DAO 管理器
        self.db_manager = MilvusManager(collection_name=collection_name)
        self._init_milvus_schema()

        # 延迟加载模型 (提升初始化速度)
        self.embed_model: FlagModel | None = None

    def _init_milvus_schema(self) -> None:
        """定义并初始化集合 Schema"""
        fields: list[FieldSchema] = [
            FieldSchema(name="qa_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(
                name="question_text",
                dtype=DataType.VARCHAR,
                max_length=1000,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},
            ),
            FieldSchema(name="answer_text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="score", dtype=DataType.INT16),
            FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="question_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="question_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        schema = CollectionSchema(fields=fields, description="Huatuo26M-Lite 医疗问答库")

        bm25_function = Function(
            name="qa_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["question_text"],
            output_field_names=["question_sparse"],
        )
        schema.add_function(bm25_function)

        self.db_manager.create_collection(schema=schema, index_field_name="question_vector")

        # 为稀疏向量创建索引
        if isinstance(self.db_manager.collection, Collection):
            self.db_manager.collection.create_index(
                field_name="question_sparse",
                index_params={
                    "metric_type": "BM25",
                    "index_type": "SPARSE_INVERTED_INDEX",
                },
            )
            logger.info("已创建 question_sparse BM25 索引")

    def _load_model(self) -> None:
        """加载 BGE-M3 模型，开启 fp16 加速显存吞吐"""
        if self.embed_model is None:
            import torch

            logger.info("🔧 检查 CUDA 设备状态...")
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA 不可用，将使用 CPU 模式运行 BGE-M3 模型。")
            else:
                logger.info("✅ CUDA 设备已就绪，将使用 GPU 模式运行 BGE-M3 模型。")

            logger.info("⏳ 正在加载 BAAI/bge-m3 Embedding 模型 (使用 FP16 加速)...")
            self.embed_model = FlagModel("BAAI/bge-m3", use_fp16=True)
            logger.success("✅ BGE-M3 模型加载完毕！")

    def run_pipeline(self) -> None:
        """核心业务流：拉取数据 -> 校验 -> 批处理向量化 -> 灌库"""
        logger.info("🚀 启动数据灌库流水线...")
        self._load_model()

        try:
            # 从 HuggingFace Datasets 动态拉取数据
            logger.info("📥 正在加载 huatuo26m-lite 数据集...")
            # dataset = load_dataset("FreedomIntelligence/huatuo26M-lite", split="train")
            dataset = load_dataset("json", data_files=str(DATABASE_JSON_PATH), split="train")
            total_records: int = len(dataset)
            logger.info(f"📊 成功拉取数据集，共计 {total_records} 条数据。")

        except Exception as e:
            logger.error(f"❌ 加载数据集失败: {e}")
            return

        # 进度统计
        success_count: int = 0
        fail_count: int = 0
        start_time: float = time.time()

        # 分批次处理 (Batch Processing)
        for i in tqdm(range(0, total_records, self.batch_size)):
            batch_raw = dataset[i : i + self.batch_size]

            # HuggingFace Datasets 切片返回的是 Dict[List]，将其转换为 List[Dict] 以便 Pydantic 校验
            keys = batch_raw.keys()
            batch_list: list[dict[str, Any]] = [
                dict(zip(keys, vals, strict=False))
                for vals in zip(*batch_raw.values(), strict=False)
            ]

            valid_items: list[HuatuoQAModel] = []

            # 1. Pydantic 数据清洗与校验
            for item in batch_list:
                try:
                    valid_items.append(HuatuoQAModel(**item))
                except ValidationError as e:
                    logger.warning(
                        f"⚠️ 脏数据拦截 (ID: {item.get('id', 'N/A')}): {e.errors()[0]['msg']}"
                    )
                    fail_count += 1
                    continue

            if not valid_items:
                continue

            try:
                # 2. 构造 BGE-M3 推荐的 Instruction 提升检索质量
                instruction: str = "为这个医学问题生成表示，用于检索相关的专业解答："
                queries: list[str] = [f"{instruction}{item.question_text}" for item in valid_items]

                # 3. 批量计算向量 (构建黑洞屏蔽 FlagEmbedding 底层强制输出的 tqdm 进度条)
                with open(os.devnull, "w") as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        assert self.embed_model is not None, "BGE-M3 模型未加载"
                        embeddings = self.embed_model.encode(queries)
                vector_list: list[list[float]] = [emb.tolist() for emb in embeddings]

                # 4. 组装 Milvus 列式实体 (Column-based)
                entities = [
                    [item.qa_id for item in valid_items],
                    [item.question_text for item in valid_items],
                    [item.answer_text for item in valid_items],
                    [item.score for item in valid_items],
                    [item.department for item in valid_items],
                    [item.source for item in valid_items],
                    vector_list,
                ]

                # 5. 调用之前写好的通用 DAO 执行插入
                self.db_manager.upsert_data(entities)
                success_count += len(valid_items)

                # 每10次打印一次进度日志
                if (i // self.batch_size) % 10 == 0:
                    logger.info(
                        f"📈 灌库进度: {success_count}/{total_records} \
                            (成功: {success_count}, 拦截/失败: {fail_count})"
                    )

            except Exception as e:
                logger.error(f"❌ 批次 {i} - {i + self.batch_size} 严重失败: {e}")
                fail_count += len(valid_items)
                # ==========================================
                # 🚀 DLQ (死信队列) 实现
                # ==========================================
                dlq_file_path = DLQ_PATH
                logger.warning(
                    f"正在将该批次 {len(valid_items)} 条数据写入死信文件: {dlq_file_path}"
                )
                dlq_items: list[dict[str, Any]] = [item.model_dump() for item in valid_items]
                write_to_dlq(dlq_file_path, dlq_items, str(e))

                # 写入完毕，处理下一个批次
                continue

        # 记录耗时与总结
        time_cost: float = time.time() - start_time
        logger.success("-" * 40)
        logger.success("🎉 灌库流水线执行完毕！")
        logger.success(f"⏱️ 总耗时: {time_cost:.2f} 秒")
        logger.success(f"✅ 成功入库: {success_count} 条")
        logger.success(f"⚠️ 校验拦截或入库失败: {fail_count} 条")
        logger.success("-" * 40)


if __name__ == "__main__":
    # 执行流水线
    pipeline = HuatuoDataPipeline(batch_size=512)
    pipeline.run_pipeline()
