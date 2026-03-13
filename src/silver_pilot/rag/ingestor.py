"""
模块名称：ingestor
功能描述：Chunk 向量化入库接口类，负责将 JSON chunk 文件解析、embedding、注入 Milvus。
         支持单文件、文件夹两种入库方式。

用法：
    from silver_pilot.rag.ingestor import ChunkIngestor

    ingestor = ChunkIngestor(backend="local")
    stats = ingestor.ingest_dir(Path("/data/chunks"))
    stats = ingestor.ingest_file(Path("/data/chunks/xxx.json"))
    print(stats)
"""

import json
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from silver_pilot.config import config
from silver_pilot.dao import MilvusManager
from silver_pilot.perception import create_embedder
from silver_pilot.utils import get_channel_logger

# ================= 日志与配置初始化 =================
LOG_FILE_DIR = config.LOG_DIR / "milvus_ingest"
logger = get_channel_logger(LOG_FILE_DIR, "milvus_ingest")

# ================= 默认配置 =================
DEFAULT_COLLECTION = config.MILVUS_COLLECTION_NAME
DEFAULT_BACKEND = "local"
DEFAULT_BATCH_SIZE = config.MILVUS_BATCH_SIZE
MILVUS_CONTENT_MAX_BYTES = config.MILVUS_CONTENT_MAX_BYTES

# ────────────────────────────────────────────────────────────
# 统计数据类
# ────────────────────────────────────────────────────────────


@dataclass
class IngestStats:
    """单次 ingest 任务的统计结果。"""

    file_count: int = 0
    inserted_total: int = 0
    skipped_total: int = 0
    failed_batches: int = 0
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"处理文件={self.file_count} | "
            f"成功入库={self.inserted_total} | "
            f"跳过={self.skipped_total} | "
            f"失败批次={self.failed_batches} | "
            f"耗时={self.elapsed_seconds:.1f}s"
        )


# ────────────────────────────────────────────────────────────
# 核心接口类
# ────────────────────────────────────────────────────────────


class ChunkIngestor:
    """
    Chunk 向量化入库接口类。

    职责：
        1. 解析 JSON chunk 文件
        2. 转换为 Milvus 实体格式
        3. 批量 embedding
        4. 注入 Milvus

    示例：
        ingestor = ChunkIngestor(backend="local", batch_size=64)
        stats = ingestor.ingest_dir(Path("/data/chunks"))
        print(stats)
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        backend: str = DEFAULT_BACKEND,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embedder = create_embedder(backend)

        self.db = MilvusManager(collection_name=self.collection_name)
        logger.info(
            f"ChunkIngestor 初始化完成 | backend={backend} | "
            f"collection={self.collection_name} | batch_size={self.batch_size}"
        )

    # ── 公开接口 ──────────────────────────────────────────────

    def ingest_dir(self, input_dir: Path) -> IngestStats:
        """
        遍历文件夹，逐文件流式 embedding + 入库。
        内存中任意时刻只持有一个文件的数据。

        :param input_dir: JSON 文件所在文件夹（递归扫描）
        :return: IngestStats 统计结果
        """
        stats = IngestStats()
        t_start = time.time()

        for file, raw_chunks in self._iter_json_files(input_dir):
            stats.file_count += 1
            file_stats = self._ingest_chunks(file.name, raw_chunks)
            stats.inserted_total += file_stats.inserted_total
            stats.skipped_total += file_stats.skipped_total
            stats.failed_batches += file_stats.failed_batches

        stats.elapsed_seconds = time.time() - t_start
        logger.info(f"ingest_dir 完成 | {stats}")
        return stats

    def ingest_file(self, file: Path) -> IngestStats:
        """
        处理单个 JSON 文件。

        :param file: JSON 文件路径
        :return: IngestStats 统计结果
        """
        stats = IngestStats(file_count=1)
        t_start = time.time()

        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
            raw_chunks = [data] if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f"读取文件失败 | file={file.name} | error={e}")
            return stats

        file_stats = self._ingest_chunks(file.name, raw_chunks)
        stats.inserted_total = file_stats.inserted_total
        stats.skipped_total = file_stats.skipped_total
        stats.failed_batches = file_stats.failed_batches
        stats.elapsed_seconds = time.time() - t_start

        logger.info(f"ingest_file 完成 | {stats}")
        return stats

    # ── 内部方法 ──────────────────────────────────────────────

    def _ingest_chunks(self, filename: str, raw_chunks: list[dict]) -> IngestStats:
        """处理单个文件内的 chunk 列表，返回该文件的统计结果。"""
        stats = IngestStats()

        logger.info(f"处理文件 | {filename} | chunk 数={len(raw_chunks)}")

        entities = []
        for chunk in raw_chunks:
            entity = self._chunk_to_entity(chunk)
            if entity:
                entities.append(entity)
            else:
                stats.skipped_total += 1

        if not entities:
            logger.warning(f"文件无有效 chunk，跳过 | file={filename}")
            return stats

        total_batches = (len(entities) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(entities), self.batch_size):
            batch = entities[i : i + self.batch_size]
            batch_idx = i // self.batch_size + 1

            try:
                vectors = self.embedder.encode([e["content"] for e in batch])
                for entity, vec in zip(batch, vectors, strict=True):
                    entity["embedding"] = vec

                inserted = self.db.upsert_data(batch)
                stats.inserted_total += inserted.upsert_count
                logger.debug(
                    f"  批次 {batch_idx}/{total_batches} | 入库={inserted.upsert_count} 条"
                )

            except Exception as e:
                stats.failed_batches += 1
                logger.error(f"  批次 {batch_idx}/{total_batches} 失败，跳过 | error={e}")

        logger.info(f"文件完成 | {filename} | 入库={stats.inserted_total} 条")
        return stats

    @staticmethod
    def _truncate_to_bytes(text: str, max_bytes: int = MILVUS_CONTENT_MAX_BYTES) -> str:
        """按字节截断，避免截断半个 UTF-8 字符导致乱码。"""
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        # errors="ignore" 自动丢弃截断处的不完整字符
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    @staticmethod
    def _chunk_to_entity(chunk: dict) -> dict | None:
        """将原始 chunk dict 转为 Milvus 实体，content 为空则返回 None。"""
        content = chunk.get("content", "").strip()
        if not content:
            logger.warning("content 为空，跳过")
            return None

        metadata = chunk.get("metadata") or {}

        return {
            "chunk_id": str(uuid.uuid4()),
            "content": ChunkIngestor._truncate_to_bytes(content, MILVUS_CONTENT_MAX_BYTES),
            "title": ChunkIngestor._truncate_to_bytes(
                metadata.get("doc_title") or metadata.get("标题", ""), 512
            ),
            "group_name": ChunkIngestor._truncate_to_bytes(chunk.get("group_name", ""), 256),
            "source_file": ChunkIngestor._truncate_to_bytes(chunk.get("source_file", ""), 1024),
            "sub_index": chunk.get("sub_index", 0),
            "meta": metadata,
            "created_at": int(time.time()),
        }

    @staticmethod
    def _iter_json_files(input_dir: Path) -> Generator[tuple[Path, list[dict]], None, None]:
        """生成器：逐文件 yield (file, chunks)，不全量加载。"""
        json_files = sorted(input_dir.rglob("*.json"))
        if not json_files:
            logger.warning(f"未找到任何 JSON 文件 | dir={input_dir}")
            return

        logger.info(f"发现 {len(json_files)} 个 JSON 文件 | dir={input_dir}")

        for file in json_files:
            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    yield file, [data]
                elif isinstance(data, list):
                    yield file, data
                else:
                    logger.warning(f"格式不支持，跳过 | file={file.name}")

            except Exception as e:
                logger.error(f"读取失败，跳过 | file={file.name} | error={e}")
