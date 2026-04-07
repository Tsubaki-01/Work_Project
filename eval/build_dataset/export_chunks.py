"""
模块名称：export_chunks
功能描述：从 Milvus 知识库中分层抽样导出 chunk 样本，供后续问题生成和人工标注使用。
         按 doc_type 分层抽样，优先选取信息密度高的 group_name 类别（临床使用、药理信息）。
         输出为 CSV 文件，由人工审核后作为 Dataset A 的 chunk 来源。

输出文件：data/eval/raw/chunks_sample.csv
下游依赖：generate_questions.py 读取此 CSV 生成问题草稿
"""

import csv
import random
from pathlib import Path

from silver_pilot.config import config
from silver_pilot.dao import MilvusManager
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "eval", "export_chunks")

# ================= 路径配置 =================
OUTPUT_PATH = config.DATA_DIR / "eval" / "raw" / "chunks_sample.csv"

# ================= 抽样配置 =================

# 优先抽取的 group_name——这些分组信息密度高，适合生成有价值的医疗问题
PREFERRED_GROUPS: list[str] = ["临床使用", "药理信息", "基本信息"]

# 各 doc_type 的抽样目标数量，合计 100 条
SAMPLE_CONFIG: dict[str, int] = {
    "drug_manual": 60,  # 药品说明书类：用法用量、禁忌、相互作用等
    "medical_guideline": 40,  # 医学指南类：治疗方案、诊断标准等
}

# chunk 内容最小字符数（过短的 chunk 信息量不足，不适合生成问题）
MIN_CONTENT_LENGTH: int = 80


def export_chunks(
    output_path: Path = OUTPUT_PATH,
    random_seed: int = 42,
) -> Path:
    """
    从 Milvus 知识库分层抽样导出 chunk，保存为 CSV。

    抽样策略：
    1. 在 PREFERRED_GROUPS 中优先抽取（信息密度高）
    2. 数量不足时从该 doc_type 的所有 chunk 中补充
    3. 过滤内容过短的 chunk

    Args:
        output_path: CSV 输出路径
        random_seed:  随机种子，保证结果可复现

    Returns:
        Path: 输出文件路径
    """
    random.seed(random_seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("开始从 Milvus 导出 chunk 样本")

    # 连接 Milvus 知识库集合
    manager = MilvusManager(collection_name=config.MILVUS_KB_COLLECTION)

    # 查询时需要返回的字段（chunk_id 是标注 relevant_chunk_ids 的关键）
    output_fields = ["chunk_id", "content", "title", "group_name", "doc_type", "source_file"]

    all_sampled: list[dict] = []

    for doc_type, target_count in SAMPLE_CONFIG.items():
        logger.info(f"处理 doc_type={doc_type}，目标抽样数={target_count}")

        # ── 阶段 1：从高优先级 group_name 中抽取 ──
        preferred_pool: list[dict] = []
        for group_name in PREFERRED_GROUPS:
            # Milvus 标量过滤表达式，同时限定 doc_type 和 group_name
            expr = f'doc_type == "{doc_type}" && group_name == "{group_name}"'
            results = manager.query_data(expr=expr, output_fields=output_fields)
            preferred_pool.extend(results)
            logger.debug(f"  group_name={group_name} 查询到 {len(results)} 条")

        # ── 阶段 2：数量不足时补充其余 chunk ──
        if len(preferred_pool) < target_count:
            expr = f'doc_type == "{doc_type}"'
            all_results = manager.query_data(expr=expr, output_fields=output_fields)

            # 去重：排除已在 preferred_pool 中的 chunk
            preferred_ids = {c["chunk_id"] for c in preferred_pool}
            supplement = [r for r in all_results if r["chunk_id"] not in preferred_ids]
            preferred_pool.extend(supplement)
            logger.info(f"  补充后候选池大小: {len(preferred_pool)}")

        # ── 阶段 3：内容长度过滤 ──
        valid_pool = [c for c in preferred_pool if len(c.get("content", "")) >= MIN_CONTENT_LENGTH]
        logger.info(f"  长度过滤后: {len(valid_pool)} 条（原 {len(preferred_pool)} 条）")

        # ── 阶段 4：随机抽样 ──
        actual_count = min(target_count, len(valid_pool))
        sampled = random.sample(valid_pool, actual_count)
        all_sampled.extend(sampled)
        logger.info(f"  实际抽样: {actual_count} 条")

    logger.info(f"全部 doc_type 抽样完成，总计 {len(all_sampled)} 条")

    # ── 写入 CSV ──
    fieldnames = ["chunk_id", "doc_type", "group_name", "title", "source_file", "content"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in all_sampled:
            # 只写入指定字段，避免其他字段混入
            writer.writerow({k: chunk.get(k, "") for k in fieldnames})

    logger.info(f"CSV 已保存至: {output_path}")
    logger.info("=" * 50)
    return output_path


if __name__ == "__main__":
    export_chunks()
