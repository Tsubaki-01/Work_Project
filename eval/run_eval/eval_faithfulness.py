"""
模块名称：eval_faithfulness
功能描述：评估生成答案的忠实度（Faithfulness）和幻觉检测的 Fallback 触发精确率。

指标一：Faithfulness（基于 Dataset A）
    定义：所有 query 的 hallucination_score 取补集的均值
    Faithfulness = 1 - mean(hallucination_score_i)
    hallucination_score 越低 → 答案越忠实 → Faithfulness 越高

指标二：Fallback 触发精确率（基于 Dataset B）
    定义：对 100 条标注样本（正样本=忠实答案，负样本=注入错误的答案），
          使用现有幻觉检测逻辑判断是否触发 fallback（score ≥ threshold），
          与 should_fallback 标签对比，计算 Precision、Recall、F1。
    重点关注 Recall（漏报幻觉的代价高于误报）

输入：
    Dataset A：data/eval/rag_retrieval.jsonl
    Dataset B：data/eval/hallucination_test.jsonl
输出：data/eval/results/faithfulness_result.json
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from silver_pilot.agent.nodes.medical_agent import _check_faithfulness, _generate_answer
from silver_pilot.agent.state import create_initial_state
from silver_pilot.config import config
from silver_pilot.rag.retriever import PipelineConfig, RAGPipeline
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "eval", "eval_faithfulness")

# ================= 路径配置 =================
DATASET_A_PATH = Path("data/eval/rag_retrieval.jsonl")
DATASET_B_PATH = Path("data/eval/hallucination_test.jsonl")
OUTPUT_PATH = Path("data/eval/results/faithfulness_result.json")

# ================= 评估配置 =================
# 与生产代码中 HALLUCINATION_THRESHOLD 保持一致（见 agent/state.py）
FALLBACK_THRESHOLD: float = config.HALLUCINATION_THRESHOLD


# ────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────

@dataclass
class FaithfulnessEvalResult:
    """Faithfulness 指标评估结果（基于 Dataset A）。"""

    total_queries: int = 0
    faithfulness: float = 0.0             # 主指标：1 - mean(hallucination_score)
    mean_hallucination_score: float = 0.0 # 辅助：直接展示幻觉分数均值
    fallback_triggered_count: int = 0     # 实际触发 fallback 的次数
    fallback_rate: float = 0.0            # fallback 触发率（=被系统拒答的比例）
    score_distribution: dict = field(default_factory=dict)  # 分数段分布


@dataclass
class FallbackPrecisionResult:
    """Fallback 触发精确率评估结果（基于 Dataset B）。"""

    total_samples: int = 0
    positive_count: int = 0              # should_fallback=False 的数量
    negative_count: int = 0              # should_fallback=True 的数量
    # 混淆矩阵
    true_positive: int = 0               # 幻觉答案被正确触发 fallback
    false_positive: int = 0              # 忠实答案被错误触发 fallback
    true_negative: int = 0               # 忠实答案被正确放行
    false_negative: int = 0              # 幻觉答案被错误放行（最危险）
    # 核心指标
    precision: float = 0.0               # TP / (TP + FP)
    recall: float = 0.0                  # TP / (TP + FN)，重点指标
    f1: float = 0.0
    accuracy: float = 0.0
    # 错误分析
    false_negatives: list[dict] = field(default_factory=list)  # 漏报的幻觉案例


@dataclass
class FaithfulnessFullResult:
    """完整评估结果（包含两个子指标）。"""

    faithfulness_on_dataset_a: FaithfulnessEvalResult = field(
        default_factory=FaithfulnessEvalResult
    )
    fallback_precision_on_dataset_b: FallbackPrecisionResult = field(
        default_factory=FallbackPrecisionResult
    )
    fallback_threshold_used: float = FALLBACK_THRESHOLD


# ────────────────────────────────────────────────────────────
# 指标一：Faithfulness（Dataset A）
# ────────────────────────────────────────────────────────────

def _eval_faithfulness(
    dataset_a_path: Path,
    pipeline: RAGPipeline,
) -> FaithfulnessEvalResult:
    """
    对 Dataset A 中的每条 query 运行完整的检索-生成-幻觉检测流程，
    汇总 hallucination_score 计算 Faithfulness。

    Args:
        dataset_a_path: Dataset A 路径
        pipeline:       已初始化的 RAGPipeline

    Returns:
        FaithfulnessEvalResult
    """
    records: list[dict] = []
    with open(dataset_a_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"=== Faithfulness 评估（Dataset A，{len(records)} 条）===")

    # 空状态（不需要用户画像，只评估检索+生成质量）
    state = create_initial_state()

    scores: list[float] = []
    fallback_triggered = 0

    for idx, record in enumerate(records):
        query = record["query"]
        query_id = record.get("id", f"q_{idx}")
        logger.info(f"  [{idx + 1}/{len(records)}] {query_id}: {query[:40]}...")

        try:
            # 检索
            retrieval = pipeline.retrieve(user_query=query)
            rag_context = retrieval.context_text
            if not rag_context:
                logger.warning(f"    RAG 无结果，跳过")
                continue

            # 生成答案
            answer = _generate_answer(query, rag_context, state)
            if not answer:
                continue

            # 幻觉检测（复用生产代码，保证评估与运行时行为完全一致）
            score = _check_faithfulness(rag_context, answer, state)
            scores.append(score)

            # 统计是否触发 fallback
            if score >= FALLBACK_THRESHOLD:
                fallback_triggered += 1

            logger.info(f"    hallucination_score={score:.4f}")

        except Exception as e:
            logger.error(f"    处理失败: {e}")
            continue

    if not scores:
        logger.error("Faithfulness 评估无有效结果")
        return FaithfulnessEvalResult()

    mean_score = sum(scores) / len(scores)
    faithfulness = 1.0 - mean_score

    # 分数段分布（便于分析幻觉分数的集中区间）
    distribution = {
        "0.0-0.1": sum(1 for s in scores if s < 0.1),
        "0.1-0.2": sum(1 for s in scores if 0.1 <= s < 0.2),
        "0.2-0.3": sum(1 for s in scores if 0.2 <= s < 0.3),
        "0.3-0.5": sum(1 for s in scores if 0.3 <= s < 0.5),
        "0.5-1.0": sum(1 for s in scores if s >= 0.5),
    }

    result = FaithfulnessEvalResult(
        total_queries=len(scores),
        faithfulness=round(faithfulness, 4),
        mean_hallucination_score=round(mean_score, 4),
        fallback_triggered_count=fallback_triggered,
        fallback_rate=round(fallback_triggered / len(scores), 4),
        score_distribution=distribution,
    )

    logger.info(f"Faithfulness:          {result.faithfulness:.4f}")
    logger.info(f"Mean hallucination:    {result.mean_hallucination_score:.4f}")
    logger.info(f"Fallback 触发率:       {result.fallback_rate:.4f} ({fallback_triggered}/{len(scores)})")
    return result


# ────────────────────────────────────────────────────────────
# 指标二：Fallback 触发精确率（Dataset B）
# ────────────────────────────────────────────────────────────

def _eval_fallback_precision(dataset_b_path: Path) -> FallbackPrecisionResult:
    """
    对 Dataset B 中的每条记录运行幻觉检测，与 should_fallback 标签对比，
    计算 Precision、Recall、F1。

    Dataset B 中的 generated_answer 和 rag_context 均已预存，
    此函数只需调用 _check_faithfulness 重新打分并二值化。

    Args:
        dataset_b_path: Dataset B 路径

    Returns:
        FallbackPrecisionResult
    """
    records: list[dict] = []
    with open(dataset_b_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"=== Fallback 触发精确率评估（Dataset B，{len(records)} 条）===")

    state = create_initial_state()

    tp = fp = tn = fn = 0
    false_negative_cases: list[dict] = []  # 漏报的幻觉答案（最危险）

    pos_count = sum(1 for r in records if not r["should_fallback"])
    neg_count = sum(1 for r in records if r["should_fallback"])
    logger.info(f"  正样本（忠实答案）: {pos_count} 条 | 负样本（幻觉答案）: {neg_count} 条")

    for idx, record in enumerate(records):
        record_id = record.get("id", f"hal_{idx}")
        query = record["query"]
        rag_context = record["rag_context"]
        generated_answer = record["generated_answer"]
        should_fallback = record["should_fallback"]

        logger.debug(f"  [{idx + 1}/{len(records)}] {record_id} | should_fallback={should_fallback}")

        try:
            # 复用生产代码中的幻觉检测函数（保证评估与运行时完全一致）
            score = _check_faithfulness(rag_context, generated_answer, state)
            # 按生产阈值二值化
            predicted_fallback = score >= FALLBACK_THRESHOLD

            # 更新混淆矩阵
            if should_fallback and predicted_fallback:
                tp += 1   # 幻觉被正确检出
            elif not should_fallback and predicted_fallback:
                fp += 1   # 忠实答案被误报为幻觉
            elif not should_fallback and not predicted_fallback:
                tn += 1   # 忠实答案被正确放行
            else:
                fn += 1   # 幻觉答案被漏报（最危险）
                false_negative_cases.append({
                    "id": record_id,
                    "query": query[:60],
                    "injection_type": record.get("injection_type", ""),
                    "hallucination_score": round(score, 4),
                    "threshold": FALLBACK_THRESHOLD,
                })

        except Exception as e:
            logger.error(f"  {record_id} 处理失败: {e}")
            continue

    # ── 计算指标 ──
    # Precision：触发 fallback 的案例中，有多少是真正的幻觉
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall：所有幻觉答案中，有多少被成功检出（重点指标，漏报代价高）
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1：综合指标
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(records) if records else 0.0

    result = FallbackPrecisionResult(
        total_samples=len(records),
        positive_count=pos_count,
        negative_count=neg_count,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        false_negatives=false_negative_cases,
    )

    logger.info("混淆矩阵（Fallback 检测）：")
    logger.info(f"  TP（幻觉被正确检出）: {tp}")
    logger.info(f"  FP（忠实被误报为幻觉）: {fp}")
    logger.info(f"  TN（忠实被正确放行）: {tn}")
    logger.info(f"  FN（幻觉被漏报）: {fn}  ← 最危险")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}  ← 重点指标")
    logger.info(f"F1:        {f1:.4f}")
    logger.info(f"Accuracy:  {accuracy:.4f}")

    return result


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

def run_eval(
    dataset_a_path: Path = DATASET_A_PATH,
    dataset_b_path: Path = DATASET_B_PATH,
    output_path: Path = OUTPUT_PATH,
) -> FaithfulnessFullResult:
    """
    执行完整的 Faithfulness + Fallback 精确率评估。

    Args:
        dataset_a_path: Dataset A 路径（Faithfulness 评估用）
        dataset_b_path: Dataset B 路径（Fallback 精确率评估用）
        output_path:    结果输出路径

    Returns:
        FaithfulnessFullResult
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 初始化 RAGPipeline（Faithfulness 评估需要真实检索）──
    logger.info("初始化 RAGPipeline...")
    pipeline = RAGPipeline(PipelineConfig())
    pipeline.initialize()

    # ── 指标一：Faithfulness ──
    faithfulness_result = _eval_faithfulness(dataset_a_path, pipeline)

    # ── 指标二：Fallback 触发精确率 ──
    fallback_result = _eval_fallback_precision(dataset_b_path)

    # ── 合并结果 ──
    full_result = FaithfulnessFullResult(
        faithfulness_on_dataset_a=faithfulness_result,
        fallback_precision_on_dataset_b=fallback_result,
        fallback_threshold_used=FALLBACK_THRESHOLD,
    )

    # ── 写入结果文件 ──
    import dataclasses
    data = dataclasses.asdict(full_result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"完整评估结果已保存至: {output_path}")

    # ── 最终汇总输出 ──
    logger.info("=" * 50)
    logger.info("▶ Faithfulness（Dataset A）")
    logger.info(f"  Faithfulness Score: {faithfulness_result.faithfulness:.4f}")
    logger.info(f"  Fallback 触发率:   {faithfulness_result.fallback_rate:.4f}")
    logger.info("▶ Fallback 触发精确率（Dataset B）")
    logger.info(f"  Precision: {fallback_result.precision:.4f}")
    logger.info(f"  Recall:    {fallback_result.recall:.4f}")
    logger.info(f"  F1:        {fallback_result.f1:.4f}")
    logger.info("=" * 50)

    return full_result


if __name__ == "__main__":
    run_eval()
