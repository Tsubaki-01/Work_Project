"""
模块名称：build_hallucination
功能描述：构建幻觉检测评估用的测试集（Dataset B）。

构建策略：
    正样本（should_fallback=False，50条）：
        运行 medical_agent 的检索+生成流程，收集 hallucination_score < 0.15 的
        真实输出，代表"忠实答案"。

    负样本（should_fallback=True，50条）：
        复用正样本的 query + rag_context，通过 LLM 对生成的答案注入以下类型的错误：
        1. 剂量篡改（100mg → 500mg）
        2. 添加 context 中不存在的药物相互作用
        3. 颠倒禁忌/适应症结论
        每种错误各占约 1/3。

输出文件：data/eval/hallucination_test.jsonl
"""

import json
import random
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，保证 import silver_pilot 可用
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openai import OpenAI
from pydantic import BaseModel

# 复用 medical_agent 的幻觉检测逻辑（不修改生产代码）
from silver_pilot.agent.nodes.medical_agent import _check_faithfulness, _generate_answer
from silver_pilot.agent.state import create_initial_state
from silver_pilot.config import config
from silver_pilot.rag.retriever import PipelineConfig, RAGPipeline
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "eval", "build_hallucination")

# ================= 路径配置 =================
# 输入：Dataset A（人工审核后的检索标注集，用于提供 query）
INPUT_JSONL = Path("data/eval/rag_retrieval.jsonl")
OUTPUT_JSONL = Path("data/eval/hallucination_test.jsonl")

# ================= 配置 =================
# 正样本的幻觉分数上限（低于此值才被收录为"忠实答案"）
POSITIVE_HALLUCINATION_THRESHOLD: float = 0.15
# 正负样本各目标数量
TARGET_PER_CLASS: int = 50
# 注入错误使用的模型（和项目 MEDICAL_AGENT_FAITHFULNESS_MODEL 保持一致，速度快）
INJECTION_MODEL: str = config.MEDICAL_AGENT_FAITHFULNESS_MODEL


# ────────────────────────────────────────────────────────────
# 幻觉注入 Prompt
# ────────────────────────────────────────────────────────────

INJECTION_SYSTEM_PROMPT = (
    """你是一个医学内容质量测试员，专门构造包含错误的医学答案用于测试幻觉检测系统。"""
)

INJECTION_USER_TEMPLATE = """以下是一段真实的医学问答，请对答案进行修改，注入一处明显但不易察觉的错误。

注入类型（请选择最适合此场景的一种）：
- 剂量篡改：将具体剂量数字修改为错误值（如100mg改为500mg，或每日一次改为每日三次）
- 虚假相互作用：添加一个context中没有提到的药物相互作用禁忌
- 结论颠倒：将禁忌症或适应症的结论反转（如"禁用"改为"可以使用"）

要求：
1. 只修改一处，其余内容保持原样
2. 修改要自然，不能让人一眼看出是故意造假
3. 直接输出修改后的完整答案，不要说明修改了什么

参考资料（context）：
{rag_context}

原始问题：{query}

原始答案：
{original_answer}

修改后的答案（注入一处错误）："""


# ────────────────────────────────────────────────────────────
# 断点续跑：读取已写入的条目，恢复进度
# ────────────────────────────────────────────────────────────


def _load_existing(output_jsonl: Path) -> tuple[list, list, set]:
    """
    读取已有的输出文件，恢复正负样本列表和已处理的 query 集合。

    断点续跑逻辑：
        - 文件不存在 → 全部从头开始
        - 文件存在   → 区分正负样本分别还原，query 集合用于跳过已处理的条目

    Args:
        output_jsonl: 输出文件路径

    Returns:
        tuple:
            positive_records (list[HallucinationRecord]): 已收录的正样本
            negative_records (list[HallucinationRecord]): 已收录的负样本
            processed_queries (set[str]):                 正样本阶段已处理过的 query
                                                          （无论是否通过阈值，均记录）
    """
    positive_records: list = []
    negative_records: list = []
    processed_queries: set[str] = set()

    if not output_jsonl.exists():
        return positive_records, negative_records, processed_queries

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record = HallucinationRecord(**data)
            if record.should_fallback:
                negative_records.append(record)
            else:
                positive_records.append(record)
                # 正样本的 query 记为已处理（负样本的 query 来自正样本，无需重复记录）
                processed_queries.add(record.query)

    logger.info(
        f"断点续跑：从已有文件恢复 | 正样本={len(positive_records)} | "
        f"负样本={len(negative_records)} | 已处理 query={len(processed_queries)}"
    )
    return positive_records, negative_records, processed_queries


# ────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────


class HallucinationRecord(BaseModel):
    """Dataset B 中的单条记录格式。"""

    id: str
    query: str
    rag_context: str
    generated_answer: str
    hallucination_score: float  # 原始幻觉检测分数（人工验证参考）
    should_fallback: bool  # 标签：True = 系统应触发 fallback
    label_reason: str  # 标注理由（方便人工复查）
    injection_type: str  # 错误注入类型（正样本为 "none"）


# ────────────────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────────────────


def build_hallucination_dataset(
    input_jsonl: Path = INPUT_JSONL,
    output_jsonl: Path = OUTPUT_JSONL,
    target_per_class: int = TARGET_PER_CLASS,
) -> Path:
    """
    构建幻觉检测测试集（Dataset B）。

    稳健性设计：
        每条记录处理完成后立即以追加模式写入文件（append-on-success），
        中途崩溃或中断重启后，已写入的条目不会丢失，脚本自动从断点继续。

        写入顺序：正样本阶段逐条追加，负样本阶段逐条追加。
        最终文件不保证正负样本交错（顺序为：全部正样本 + 全部负样本），
        eval_faithfulness.py 读取时按 should_fallback 字段区分，不依赖顺序。

    Args:
        input_jsonl:      Dataset A 路径（提供 query 来源）
        output_jsonl:     Dataset B 输出路径
        target_per_class: 正负样本各自的目标数量

    Returns:
        Path: 输出文件路径
    """
    if not input_jsonl.exists():
        raise FileNotFoundError(
            f"Dataset A 不存在: {input_jsonl}，请先完成 generate_questions.py + 人工审核"
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # ── 断点续跑：读取已有进度 ──
    positive_records, negative_records, processed_queries = _load_existing(output_jsonl)

    # ── 读取 Dataset A 获取 query 列表 ──
    queries: list[str] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line)["query"])
    logger.info(f"从 Dataset A 读取 {len(queries)} 条 query")

    # ── 初始化 RAG Pipeline ──
    logger.info("初始化 RAGPipeline...")
    pipeline = RAGPipeline(PipelineConfig())
    pipeline.initialize()

    # ── LLM 客户端（用于幻觉注入）──
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.QWEN_URL[config.QWEN_REGION],
    )

    state = create_initial_state()  # 空画像状态，不影响生成质量评估

    # ── 阶段 1：收集正样本 ──
    pos_needed = target_per_class - len(positive_records)
    if pos_needed <= 0:
        logger.info(f"正样本已满 {target_per_class} 条，跳过阶段 1")
    else:
        logger.info(f"=== 阶段 1：收集正样本（还需 {pos_needed} 条）===")

        # 以追加模式打开文件，后续每条成功后立即写入
        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for idx, query in enumerate(queries):
                if len(positive_records) >= target_per_class:
                    break

                # 跳过已在上次运行中处理过的 query（无论是否通过阈值）
                if query in processed_queries:
                    logger.debug(f"  跳过已处理 query: {query[:30]}...")
                    continue

                logger.info(f"  [{idx + 1}/{len(queries)}] 运行检索+生成 | query={query[:30]}...")

                try:
                    # 阶段 1a：检索
                    retrieval = pipeline.retrieve(user_query=query)
                    rag_context = retrieval.context_text
                    if not rag_context:
                        logger.debug("    RAG 无结果，跳过")
                        # 无结果的 query 也记为已处理，避免重启后重复尝试
                        processed_queries.add(query)
                        continue

                    # 阶段 1b：生成答案
                    generated_answer = _generate_answer(query, rag_context, state)
                    if not generated_answer:
                        processed_queries.add(query)
                        continue

                    # 阶段 1c：幻觉检测
                    hallucination_score = _check_faithfulness(rag_context, generated_answer, state)

                    # 无论是否通过阈值，都标记为已处理
                    processed_queries.add(query)

                    if hallucination_score >= POSITIVE_HALLUCINATION_THRESHOLD:
                        logger.debug(
                            f"    幻觉分数 {hallucination_score:.2f} ≥ 阈值，跳过（不够忠实）"
                        )
                        continue

                    record = HallucinationRecord(
                        # ID 从当前已有正样本数续接，保证唯一且连续
                        id=f"hal_{len(positive_records) + 1:03d}",
                        query=query,
                        rag_context=rag_context,
                        generated_answer=generated_answer,
                        hallucination_score=hallucination_score,
                        should_fallback=False,
                        label_reason=(
                            f"LLM 幻觉检测分数={hallucination_score:.3f}，"
                            f"低于阈值 {POSITIVE_HALLUCINATION_THRESHOLD}"
                        ),
                        injection_type="none",
                    )

                    # ── 立即写入，不等待循环结束 ──
                    out_f.write(record.model_dump_json(ensure_ascii=False) + "\n")
                    out_f.flush()  # 确保操作系统缓冲区落盘

                    positive_records.append(record)
                    logger.info(
                        f"    ✓ 收录正样本 [{record.id}]，"
                        f"当前 {len(positive_records)}/{target_per_class}"
                    )

                except Exception as e:
                    logger.error(f"    处理失败: {e}")
                    # 异常的 query 也标记为已处理，避免反复触发同一错误
                    processed_queries.add(query)
                    continue

    logger.info(f"正样本收集完成：{len(positive_records)} 条")

    # ── 阶段 2：构建负样本（错误注入）──
    neg_needed = target_per_class - len(negative_records)
    if neg_needed <= 0:
        logger.info(f"负样本已满 {target_per_class} 条，跳过阶段 2")
    else:
        logger.info(f"=== 阶段 2：构建负样本（还需 {neg_needed} 条）===")

        injection_types = ["剂量篡改", "虚假相互作用", "结论颠倒"]

        # 负样本来源：从正样本中取，跳过已在上次运行中处理过的部分
        # 已有负样本的数量即为上次已处理的正样本索引上限
        start_idx = len(negative_records)

        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for idx, pos_record in enumerate(positive_records[start_idx:], start=start_idx):
                if len(negative_records) >= target_per_class:
                    break

                injection_type = injection_types[idx % len(injection_types)]
                logger.info(
                    f"  [{idx + 1}/{len(positive_records)}] 注入错误 | "
                    f"type={injection_type} | query={pos_record.query[:30]}..."
                )

                try:
                    user_prompt = INJECTION_USER_TEMPLATE.format(
                        rag_context=pos_record.rag_context[:1000],
                        query=pos_record.query,
                        original_answer=pos_record.generated_answer,
                    )
                    response = client.chat.completions.create(
                        model=INJECTION_MODEL,
                        messages=[
                            {"role": "system", "content": INJECTION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=500,
                    )
                    injected_answer = response.choices[0].message.content or ""

                    if not injected_answer.strip():
                        logger.warning("    注入结果为空，跳过")
                        continue

                    if injected_answer.strip() == pos_record.generated_answer.strip():
                        logger.warning("    LLM 未修改答案，跳过")
                        continue

                    record = HallucinationRecord(
                        # 负样本 ID 紧接正样本编号之后
                        id=f"hal_{target_per_class + len(negative_records) + 1:03d}",
                        query=pos_record.query,
                        rag_context=pos_record.rag_context,
                        generated_answer=injected_answer,
                        hallucination_score=-1.0,
                        should_fallback=True,
                        label_reason=(f"人工注入错误，类型={injection_type}，原答案与注入答案不同"),
                        injection_type=injection_type,
                    )

                    # ── 立即写入 ──
                    out_f.write(record.model_dump_json(ensure_ascii=False) + "\n")
                    out_f.flush()

                    negative_records.append(record)
                    logger.info(
                        f"    ✓ 构建负样本 [{record.id}]，"
                        f"当前 {len(negative_records)}/{target_per_class}"
                    )

                except Exception as e:
                    logger.error(f"    注入失败: {e}")
                    continue

    logger.info(f"负样本构建完成：{len(negative_records)} 条")

    # ── 输出统计 ──
    all_count = len(positive_records) + len(negative_records)
    logger.info(
        f"Dataset B 构建完成 | 总计={all_count} | "
        f"正样本={len(positive_records)} | 负样本={len(negative_records)} | "
        f"文件={output_jsonl}"
    )

    injection_types_all = ["剂量篡改", "虚假相互作用", "结论颠倒"]
    for inj_type in injection_types_all:
        count = sum(1 for r in negative_records if r.injection_type == inj_type)
        logger.info(f"  注入类型 [{inj_type}]: {count} 条")

    return output_jsonl


if __name__ == "__main__":
    build_hallucination_dataset()
