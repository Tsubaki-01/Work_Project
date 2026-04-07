"""
模块名称：eval_intent
功能描述：评估 Supervisor 意图分类的准确率（Accuracy）及各类别 F1。

评估方式：
    直接调用 supervisor.py 中的 _classify_and_dispatch()，
    构造最小化 AgentState 传入，不走完整 LangGraph 图（避免启动开销）。

指标：
    - 整体 Accuracy（单意图 query 准确判断意图的比例）
    - 各类别 Precision / Recall / F1（Macro-F1 作为汇总指标）
    - EMERGENCY 类 Recall（单独报告，漏报代价极高）
    - 复合意图识别率（is_composite=True 的 query，判断是否识别出多个意图）

输入：data/eval/intent_classification.jsonl（Dataset C，人工构造）
输出：
    data/eval/results/intent_result.json        最终聚合结果
    data/eval/results/intent_result_raw.jsonl   逐条中间记录（支持断点续跑）

Dataset C 格式参考（每行一条）：
{
  "id": "intent_001",
  "input": "阿司匹林一天吃几次",
  "label": "MEDICAL_QUERY",          // 单意图：字符串
  "is_composite": false,
  "difficulty": "easy"
}
{
  "id": "intent_101",
  "input": "帮我查一下阿司匹林怎么吃，顺便明天7点提醒我吃药",
  "label": ["MEDICAL_QUERY", "DEVICE_CONTROL"],  // 复合意图：列表
  "is_composite": true,
  "difficulty": "hard"
}
"""

import concurrent.futures
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from silver_pilot.agent.nodes.supervisor import INTENT_TO_AGENT, _classify_and_dispatch
from silver_pilot.agent.state import create_initial_state
from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "eval", "eval_intent")

# ================= 路径配置 =================
DATASET_PATH = Path("data/eval/intent_classification.jsonl")
OUTPUT_PATH = Path("data/eval/results/intent_result.json")
RAW_RECORD_PATH = Path("data/eval/results/intent_result_raw.jsonl")  # 逐条中间记录

# ================= 超时与重试配置 =================
TIMEOUT_SECONDS = 90  # 单条调用超时阈值（秒）
MAX_RETRIES = 3  # 最大重试次数

# ================= 意图类型定义 =================
ALL_INTENT_TYPES: list[str] = ["MEDICAL_QUERY", "DEVICE_CONTROL", "CHITCHAT", "EMERGENCY"]

# agent 名 → 意图类型的反向映射
AGENT_TO_INTENT: dict[str, str] = {v: k for k, v in INTENT_TO_AGENT.items()}


# ────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────


@dataclass
class SingleIntentResult:
    """单条 query 的分类评估结果。"""

    query_id: str
    input_text: str
    label: list[str]  # 真实标签（单意图也用列表存储，统一处理）
    predicted: list[str]  # 预测结果（从 pending_intents 中提取）
    is_composite: bool
    correct: bool  # 单意图：完全匹配；复合意图：标签集合完全匹配
    partial_match: bool  # 复合意图：预测集合与标签有交集
    raw_output: dict  # _classify_and_dispatch 的原始返回（调试用）


@dataclass
class IntentEvalResult:
    """意图分类完整评估结果。"""

    total: int = 0
    accuracy: float = 0.0

    # ── 单意图分类细粒度指标 ──
    macro_f1: float = 0.0
    per_class_metrics: dict = field(default_factory=dict)

    # ── EMERGENCY 专项 ──
    emergency_recall: float = 0.0
    emergency_total: int = 0

    # ── 复合意图 ──
    composite_accuracy: float = 0.0
    composite_partial_rate: float = 0.0
    composite_total: int = 0

    # ── 分层统计 ──
    by_difficulty: dict = field(default_factory=dict)

    # ── 错误案例（TOP-20）──
    error_cases: list[dict] = field(default_factory=list)


# ────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────


def _build_state_for_query(input_text: str) -> dict:
    """构造最小化 AgentState，只注入用户消息。"""
    state = create_initial_state()
    state["messages"] = [HumanMessage(content=input_text)]
    return state


def _extract_predicted_intents(dispatch_output: dict) -> list[str]:
    """从 _classify_and_dispatch 返回值中提取预测的意图类型列表。"""
    pending_intents: list[dict] = dispatch_output.get("pending_intents", [])
    current_agent: str = dispatch_output.get("current_agent", "")

    if pending_intents:
        return [intent.get("type", "CHITCHAT") for intent in pending_intents]

    if current_agent and current_agent not in ("done", "parallel"):
        intent_type = AGENT_TO_INTENT.get(current_agent, "CHITCHAT")
        return [intent_type]

    return ["CHITCHAT"]


def _load_existing_records(raw_path: Path) -> dict[str, dict]:
    """
    读取已有的逐条中间记录，返回 {query_id: record} 字典。
    用于断点续跑时跳过已完成条目。
    """
    existing: dict[str, dict] = {}
    if not raw_path.exists():
        return existing
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    existing[rec["query_id"]] = rec
                except json.JSONDecodeError:
                    pass
    return existing


def _append_record(raw_path: Path, record: dict) -> None:
    """将单条结果追加写入中间 JSONL 文件。"""
    with open(raw_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _classify_with_timeout(input_text: str) -> dict:
    """
    调用 _classify_and_dispatch，超过 TIMEOUT_SECONDS 秒则抛出 TimeoutError。
    使用线程池实现超时控制（兼容同步代码）。
    """

    def _call() -> dict[str, Any]:
        state = _build_state_for_query(input_text)
        return _classify_and_dispatch(state)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        try:
            return future.result(timeout=TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"分类调用超过 {TIMEOUT_SECONDS}s 未返回")


def _classify_with_retry(input_text: str, query_id: str) -> tuple[list[str], dict]:
    """
    带超时重试的分类调用。
    超时或异常时自动重试，最多 MAX_RETRIES 次。

    Returns:
        (predicted_intents, raw_output_dict)
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            dispatch_output = _classify_with_timeout(input_text)
            predicted = _extract_predicted_intents(dispatch_output)
            raw_output = {
                "current_agent": dispatch_output.get("current_agent"),
                "risk_level": dispatch_output.get("risk_level"),
                "pending_intents_types": [
                    i.get("type") for i in dispatch_output.get("pending_intents", [])
                ],
            }
            if attempt > 1:
                logger.info(f"  [重试成功] {query_id} 第 {attempt} 次成功")
            return predicted, raw_output

        except TimeoutError as e:
            last_error = e
            logger.warning(
                f"  [超时] {query_id} 第 {attempt}/{MAX_RETRIES} 次 — "
                f"超过 {TIMEOUT_SECONDS}s，{'重试...' if attempt < MAX_RETRIES else '放弃'}"
            )
        except Exception as e:
            last_error = e
            logger.error(
                f"  [异常] {query_id} 第 {attempt}/{MAX_RETRIES} 次 — "
                f"{e}，{'重试...' if attempt < MAX_RETRIES else '放弃'}"
            )

    # 全部重试失败，降级处理
    logger.error(f"  [最终失败] {query_id} 全部 {MAX_RETRIES} 次均失败: {last_error}")
    return ["CHITCHAT"], {}


# ────────────────────────────────────────────────────────────
# 核心评估：逐条执行 + 逐条记录
# ────────────────────────────────────────────────────────────


def run_eval(
    dataset_path: Path = DATASET_PATH,
    output_path: Path = OUTPUT_PATH,
    raw_record_path: Path = RAW_RECORD_PATH,
) -> IntentEvalResult:
    """
    执行意图分类评估。

    流程：
        1. 加载 Dataset C
        2. 读取已有中间记录（断点续跑）
        3. 逐条调用分类器（含超时重试），每条完成后立即写入中间 JSONL
        4. 全部完成后读取中间 JSONL，聚合计算指标
        5. 写入最终 JSON 结果

    Args:
        dataset_path:     Dataset C 路径
        output_path:      最终聚合结果输出路径
        raw_record_path:  逐条中间记录路径

    Returns:
        IntentEvalResult
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset C 不存在: {dataset_path}，请先人工构建意图标注集")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_record_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 读取数据集 ──
    records: list[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"加载 Dataset C：{len(records)} 条 query")

    # ── 读取已有中间记录（断点续跑）──
    existing_records = _load_existing_records(raw_record_path)
    if existing_records:
        logger.info(f"发现已有中间记录：{len(existing_records)} 条，将跳过已完成条目")

    # ── 逐条执行 + 逐条写入 ──
    for idx, record in enumerate(records):
        query_id = record.get("id", f"intent_{idx}")
        input_text = record["input"]
        raw_label = record["label"]
        is_composite = record.get("is_composite", False)
        difficulty = record.get("difficulty", "medium")

        label_list: list[str] = raw_label if isinstance(raw_label, list) else [raw_label]

        # 已存在则跳过
        if query_id in existing_records:
            logger.info(f"[{idx + 1}/{len(records)}] {query_id}: 已有记录，跳过")
            continue

        logger.info(
            f"[{idx + 1}/{len(records)}] {query_id}: {input_text[:40]}..."
            f"  (timeout={TIMEOUT_SECONDS}s, max_retry={MAX_RETRIES})"
        )

        # 调用分类器（含超时重试）
        predicted, raw_output = _classify_with_retry(input_text, query_id)

        # 判断正确性
        label_set = set(label_list)
        predicted_set = set(predicted)
        correct = label_set == predicted_set
        partial_match = bool(label_set & predicted_set)

        status = "✓" if correct else ("~" if partial_match else "✗")
        logger.info(f"  {status} 预测={predicted} | 标签={label_list} | correct={correct}")

        # 构造本条记录并立即追加写入中间文件
        row = {
            "query_id": query_id,
            "input_text": input_text,
            "label": label_list,
            "predicted": predicted,
            "is_composite": is_composite,
            "difficulty": difficulty,
            "correct": correct,
            "partial_match": partial_match,
            "raw_output": raw_output,
        }
        _append_record(raw_record_path, row)

    logger.info(f"逐条评估完毕，中间记录已全部写入: {raw_record_path}")

    # ── 读取中间记录，聚合计算 ──
    return _compute_and_save(raw_record_path, output_path, records)


# ────────────────────────────────────────────────────────────
# 读取中间记录 → 聚合计算
# ────────────────────────────────────────────────────────────


def _compute_and_save(
    raw_record_path: Path,
    output_path: Path,
    records: list[dict],
) -> IntentEvalResult:
    """
    读取逐条中间 JSONL，重建 SingleIntentResult 列表，聚合计算并保存结果。

    Args:
        raw_record_path: 逐条中间记录路径
        output_path:     最终聚合结果输出路径
        records:         原始数据集（用于 difficulty 等元信息）

    Returns:
        IntentEvalResult
    """
    logger.info(f"从中间记录重新加载数据: {raw_record_path}")

    all_results: list[SingleIntentResult] = []
    with open(raw_record_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            all_results.append(
                SingleIntentResult(
                    query_id=row["query_id"],
                    input_text=row["input_text"],
                    label=row["label"],
                    predicted=row["predicted"],
                    is_composite=row["is_composite"],
                    correct=row["correct"],
                    partial_match=row["partial_match"],
                    raw_output=row.get("raw_output", {}),
                )
            )

    if not all_results:
        logger.error("中间记录为空，无法计算指标")
        return IntentEvalResult()

    logger.info(f"共加载 {len(all_results)} 条中间记录，开始聚合计算...")

    eval_result = _aggregate_results(all_results, records)

    # 写入最终结果
    import dataclasses

    data = dataclasses.asdict(eval_result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"评估结果已保存至: {output_path}")
    return eval_result


# ────────────────────────────────────────────────────────────
# 聚合计算（不变）
# ────────────────────────────────────────────────────────────


def _aggregate_results(
    all_results: list[SingleIntentResult],
    records: list[dict],
) -> IntentEvalResult:
    """
    从逐条结果聚合计算所有指标。

    Args:
        all_results: 逐条评估结果
        records:     原始数据集记录（用于提取 difficulty 等元信息）

    Returns:
        IntentEvalResult
    """
    total = len(all_results)

    # ── 整体 Accuracy ──
    accuracy = sum(r.correct for r in all_results) / total

    # ── 单意图：各类别 Precision / Recall / F1 ──
    single_results = [r for r in all_results if not r.is_composite]

    tp_map: dict[str, int] = defaultdict(int)
    fp_map: dict[str, int] = defaultdict(int)
    fn_map: dict[str, int] = defaultdict(int)

    for r in single_results:
        label = r.label[0] if r.label else "CHITCHAT"
        predicted = r.predicted[0] if r.predicted else "CHITCHAT"

        if label == predicted:
            tp_map[label] += 1
        else:
            fn_map[label] += 1
            fp_map[predicted] += 1

    per_class_metrics: dict[str, dict] = {}
    f1_values: list[float] = []

    for intent_type in ALL_INTENT_TYPES:
        tp = tp_map[intent_type]
        fp = fp_map[intent_type]
        fn = fn_map[intent_type]
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[intent_type] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            f1_values.append(f1)

        logger.info(f"  [{intent_type}] P={precision:.4f} R={recall:.4f} F1={f1:.4f} N={support}")

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    # ── EMERGENCY 专项指标 ──
    emergency_results = [r for r in all_results if "EMERGENCY" in r.label]
    emergency_total = len(emergency_results)
    emergency_correct = sum(1 for r in emergency_results if "EMERGENCY" in r.predicted)
    emergency_recall = emergency_correct / emergency_total if emergency_total > 0 else 0.0

    # ── 复合意图指标 ──
    composite_results = [r for r in all_results if r.is_composite]
    composite_total = len(composite_results)
    composite_accuracy = (
        sum(r.correct for r in composite_results) / composite_total if composite_total > 0 else 0.0
    )
    composite_partial_rate = (
        sum(r.partial_match for r in composite_results) / composite_total
        if composite_total > 0
        else 0.0
    )

    # ── 按 difficulty 分层统计 ──
    difficulty_groups: dict[str, list[SingleIntentResult]] = defaultdict(list)
    id_to_difficulty = {r.get("id", ""): r.get("difficulty", "medium") for r in records}
    for r in all_results:
        diff = id_to_difficulty.get(r.query_id, "medium")
        difficulty_groups[diff].append(r)

    by_difficulty: dict[str, dict] = {}
    for diff, group in difficulty_groups.items():
        n = len(group)
        by_difficulty[diff] = {
            "count": n,
            "accuracy": round(sum(r.correct for r in group) / n, 4),
        }

    # ── 错误案例（最多20条）──
    error_cases = [
        {
            "query_id": r.query_id,
            "input": r.input_text[:60],
            "label": r.label,
            "predicted": r.predicted,
            "is_composite": r.is_composite,
        }
        for r in all_results
        if not r.correct
    ][:20]

    # ── 汇总输出 ──
    logger.info("=" * 50)
    logger.info(
        f"整体 Accuracy:      {accuracy:.4f} ({sum(r.correct for r in all_results)}/{total})"
    )
    logger.info(f"Macro-F1:           {macro_f1:.4f}")
    logger.info(
        f"EMERGENCY Recall:   {emergency_recall:.4f} ({emergency_correct}/{emergency_total}) ← 重点"
    )
    logger.info(f"复合意图 Accuracy:  {composite_accuracy:.4f} ({composite_total} 条)")
    logger.info(f"错误案例数:         {len(error_cases)}")

    return IntentEvalResult(
        total=total,
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        per_class_metrics=per_class_metrics,
        emergency_recall=round(emergency_recall, 4),
        emergency_total=emergency_total,
        composite_accuracy=round(composite_accuracy, 4),
        composite_partial_rate=round(composite_partial_rate, 4),
        composite_total=composite_total,
        by_difficulty=by_difficulty,
        error_cases=error_cases,
    )


if __name__ == "__main__":
    run_eval()
