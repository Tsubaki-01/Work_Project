"""
模块名称：generate_questions
功能描述：读取 export_chunks.py 导出的 CSV，调用 DashScope LLM 为每条 chunk
         批量生成老年人风格的口语化问题，输出为待人工审核的 JSONL 草稿。

工作流：
    export_chunks.py → chunks_sample.csv
                            ↓（本脚本）
                    draft_questions.jsonl
                            ↓（人工审核：删除错误条目、补充难例、标注 relevant_chunk_ids）
                    data/eval/rag_retrieval.jsonl  ← Dataset A 最终文件

输出字段：
    id, query, relevant_chunk_ids, ground_truth_answer, category, difficulty, chunk_id（来源）
"""

import csv
import json
import time
from pathlib import Path

from openai import OpenAI

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "eval", "generate_questions")

# ================= 路径配置 =================
INPUT_CSV = config.DATA_DIR / "eval" / "raw" / "chunks_sample.csv"
OUTPUT_JSONL = config.DATA_DIR / "eval" / "raw" / "draft_questions.jsonl"

# ================= LLM 配置 =================
# 问题生成使用 flash 模型，速度快、成本低
QUESTION_GEN_MODEL: str = "qwen-flash"
# 每条 chunk 生成的问题数（1-2 条，多了容易质量下降）
QUESTIONS_PER_CHUNK: int = 1
# API 调用间隔（避免触发限流）
API_CALL_INTERVAL: float = 0.5

# ================= Prompt 模板 =================
# 核心要求：口语化、符合老年人表达习惯、不能直接照抄原文
QUESTION_GEN_SYSTEM_PROMPT = (
    """你是一位资深的医疗问卷设计师，擅长设计贴近老年人日常表达的健康咨询问题。"""
)

QUESTION_GEN_USER_TEMPLATE = """请根据以下医学文本，
生成 {n} 个老年人在日常生活中可能提问的问题经过大模型意图识别后的意图。

要求：
1. 问题必须能从文本中找到答案，但不能直接复制文本中的词句
2. 涵盖实际使用场景（用法用量、注意事项、药物相互作用等）
3. 每个问题一行，直接输出问题文本，不要编号

例子：
输入
- 医学文本（来源：“高血压诊治指南”）：
降压药与头孢类抗生素多数情况下无严重直接拮抗...

输出
降压药与头孢类抗生素的药物相互作用


以下是输入：
医学文本（来源：{source}）：
{content}

请直接输出意图，每行一个："""


def generate_questions(
    input_csv: Path = INPUT_CSV,
    output_jsonl: Path = OUTPUT_JSONL,
    questions_per_chunk: int = QUESTIONS_PER_CHUNK,
) -> Path:
    """
    为 CSV 中的每条 chunk 调用 LLM 生成问题草稿并保存为 JSONL。

    JSONL 中的每条记录包含：
    - id: 全局唯一编号
    - query: 生成的问题文本
    - relevant_chunk_ids: 初始仅含来源 chunk_id（人工审核时需补充其他相关 chunk）
    - ground_truth_answer: 占位空字符串（人工审核时填写）
    - source_chunk_id: 来源 chunk 的 ID（人工审核参考用）
    - doc_type / group_name: 分类信息，便于分层统计
    - category / difficulty: 人工审核时填写

    Args:
        input_csv:           export_chunks.py 的输出 CSV 路径
        output_jsonl:        草稿输出路径
        questions_per_chunk: 每条 chunk 生成的问题数

    Returns:
        Path: 输出文件路径
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}，请先运行 export_chunks.py")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # ── 初始化 LLM 客户端（复用项目统一接入方式）──
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.QWEN_URL[config.QWEN_REGION],
    )

    # ── 读取 CSV ──
    chunks: list[dict] = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        chunks = list(reader)
    logger.info(f"读取到 {len(chunks)} 条 chunk，开始批量生成问题")

    draft_records: list[dict] = []
    global_id = 1

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        source = chunk.get("title") or chunk.get("source_file", "未知来源")
        doc_type = chunk.get("doc_type", "")
        group_name = chunk.get("group_name", "")

        logger.info(f"[{idx + 1}/{len(chunks)}] 处理 chunk_id={chunk_id[:16]}...")

        # ── 调用 LLM 生成问题 ──
        user_prompt = QUESTION_GEN_USER_TEMPLATE.format(
            n=questions_per_chunk,
            source=source,
            content=content[:1500],  # 截断超长 chunk，避免超出 context
        )

        try:
            response = client.chat.completions.create(
                model=QUESTION_GEN_MODEL,
                messages=[
                    {"role": "system", "content": QUESTION_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # 适度随机性，避免生成雷同问题
                max_tokens=200,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"  LLM 调用失败 | chunk_id={chunk_id} | error={e}")
            raw_output = ""

        # ── 解析输出（按行分割，过滤空行）──
        questions = [
            q.strip() for q in raw_output.strip().split("\n") if q.strip() and len(q.strip()) > 5
        ]
        questions = questions[:questions_per_chunk]  # 截断多余问题

        if not questions:
            logger.warning(f"  chunk_id={chunk_id} 生成问题为空，跳过")
            continue

        # ── 构建草稿记录 ──
        for question in questions:
            record = {
                "id": f"rag_{global_id:03d}",
                "query": question,
                # 来源 chunk 默认作为 relevant，人工审核时可补充其他相关 chunk
                "relevant_chunk_ids": [chunk_id],
                # 分类信息（人工审核时可修改）
                "category": _infer_category(group_name, doc_type),
                # 难度（默认 medium，人工审核时调整）
                "difficulty": "medium",
                # ── 以下字段仅供人工审核参考，不进入最终 Dataset A ──
                "_source_chunk_id": chunk_id,
                "_source_content_preview": content[:100] + "...",
                "_doc_type": doc_type,
                "_group_name": group_name,
            }
            draft_records.append(record)
            global_id += 1

        # 限流保护
        time.sleep(API_CALL_INTERVAL)

    # ── 写入 JSONL ──
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in draft_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"草稿生成完成 | 总问题数={len(draft_records)} | 文件={output_jsonl}")
    return output_jsonl


def _infer_category(group_name: str, doc_type: str) -> str:
    """
    根据 group_name 和 doc_type 推断问题类别，方便后续分类统计。

    Args:
        group_name: Milvus 中的分组名称
        doc_type:   文档类型

    Returns:
        str: 类别标签
    """
    category_map = {
        "临床使用": "clinical_usage",
        "药理信息": "pharmacology",
        "基本信息": "basic_info",
        "表格数据": "table_data",
    }
    if group_name in category_map:
        return category_map[group_name]
    if doc_type == "drug_manual":
        return "drug_general"
    return "medical_general"


if __name__ == "__main__":
    generate_questions()
