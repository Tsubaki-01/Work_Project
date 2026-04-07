"""
模块名称：generate_report
功能描述：读取所有评估脚本的 JSON 输出，汇总为结构化的 Markdown 报告。
         报告包含：核心指标汇总表、分层分析、失败案例摘要、简历表述模板。

依赖（按执行顺序）：
    eval_rag_retrieval.py  → data/eval/results/rag_retrieval_result.json
    eval_faithfulness.py   → data/eval/results/faithfulness_result.json
    eval_intent.py         → data/eval/results/intent_result.json
    eval_parallel.py       → data/eval/results/parallel_result.json

输出：data/eval/results/report.md
"""

import json
from pathlib import Path
from typing import Any

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "eval", "generate_report")

# ================= 路径配置 =================
RESULTS_DIR = Path("data/eval/results")
RESULT_FILES = {
    "rag_retrieval": RESULTS_DIR / "rag_retrieval_result.json",
    "faithfulness": RESULTS_DIR / "faithfulness_result.json",
    "intent": RESULTS_DIR / "intent_result.json",
    "parallel": RESULTS_DIR / "parallel_result.json",
}
OUTPUT_PATH = RESULTS_DIR / "report.md"


# ────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    """
    加载 JSON 文件。文件不存在时返回空字典并记录警告。

    Args:
        path: JSON 文件路径

    Returns:
        dict: 解析后的数据，文件不存在时为 {}
    """
    if not path.exists():
        logger.warning(f"结果文件不存在，跳过: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────
# 报告各节渲染函数
# ────────────────────────────────────────────────────────────


def _render_header() -> str:
    """渲染报告标题和元信息。"""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# Silver Pilot 评估报告

> 生成时间：{now}
> 项目：Silver Pilot — 面向老年人的医疗健康 RAG + Multi-Agent 系统

---
"""


def _render_summary_table(
    rag: dict,
    faith: dict,
    intent: dict,
    parallel: dict,
) -> str:
    """
    渲染核心指标汇总表（最重要的一张表，用于简历和面试）。

    Args:
        rag:      RAG 检索评估结果
        faith:    Faithfulness 评估结果
        intent:   意图分类评估结果
        parallel: 并行加速比评估结果

    Returns:
        str: Markdown 格式的汇总表
    """
    # 从各 JSON 中提取关键数字
    hit_rate = rag.get("hit_rate", "N/A")
    recall_k = rag.get("recall_at_k", "N/A")
    top_k = rag.get("top_k", 3)

    faith_a = faith.get("faithfulness_on_dataset_a", {})
    faithfulness = faith_a.get("faithfulness", "N/A")
    fallback_rate = faith_a.get("fallback_rate", "N/A")

    faith_b = faith.get("fallback_precision_on_dataset_b", {})
    fallback_f1 = faith_b.get("f1", "N/A")
    fallback_recall = faith_b.get("recall", "N/A")

    accuracy = intent.get("accuracy", "N/A")
    macro_f1 = intent.get("macro_f1", "N/A")
    emg_recall = intent.get("emergency_recall", "N/A")

    speedup = parallel.get("mean_speedup_ratio", "N/A")
    serial_ms = parallel.get("mean_serial_ms", "N/A")
    parallel_ms = parallel.get("mean_parallel_ms", "N/A")

    # 百分比格式化
    def pct(val: object, decimals: int = 2) -> str:
        if isinstance(val, (int, float)):
            return f"{val * 100:.{decimals}f}%"
        return str(val)

    return f"""## 核心指标汇总

| 模块 | 指标 | 数值 | 说明 |
|---|---|---|---|
| **RAG 检索** | Hit Rate@{top_k} | **{pct(hit_rate)}** | top-{top_k} 中命中至少 1 条相关 chunk 的比例 |
| **RAG 检索** | Recall@{top_k} | **{pct(recall_k)}** | top-{top_k} 覆盖所有相关 chunk 的平均比例 |
| **答案生成** | Faithfulness | **{pct(faithfulness)}** | 1 - mean(hallucination_score)，越高越忠实 |
| **答案生成** | Fallback 触发率 | **{pct(fallback_rate)}** | 实际触发安全兜底的比例 |
| **幻觉检测** | Fallback F1 | **{pct(fallback_f1)}** | 幻觉检测二分类综合指标 |
| **幻觉检测** | Fallback Recall | **{pct(fallback_recall)}** | 幻觉答案被检出的比例（重点指标） |
| **意图分类** | Accuracy | **{pct(accuracy)}** | 意图完全匹配率（含复合意图） |
| **意图分类** | Macro-F1 | **{pct(macro_f1)}** | 4 类意图 F1 的宏平均 |
| **意图分类** | EMERGENCY Recall | **{pct(emg_recall)}** | 紧急情况识别率（安全关键指标） |
| **并行执行** | 加速比 | **{speedup}x** | 串行 {serial_ms}ms → 并行 {parallel_ms}ms |

"""


def _render_rag_detail(rag: dict) -> str:
    """渲染 RAG 检索分层详情。"""
    if not rag:
        return "## RAG 检索详情\n\n> 结果文件缺失，请先运行 eval_rag_retrieval.py\n\n"

    top_k = rag.get("top_k", 3)
    total = rag.get("total_queries", 0)
    failed = rag.get("failed_cases", [])

    # 按 difficulty 分层
    by_diff = rag.get("by_difficulty", {})
    diff_table = ""
    if by_diff:
        diff_table = (
            "\n### 按难度分层\n\n| 难度 | 数量 | Hit Rate | Recall@K |\n|---|---|---|---|\n"
        )
        for diff, stats in sorted(by_diff.items()):
            diff_table += (
                f"| {diff} | {stats['count']} | "
                f"{stats['hit_rate'] * 100:.2f}% | "
                f"{stats['recall_at_k'] * 100:.2f}% |\n"
            )

    # 按 category 分层
    by_cat = rag.get("by_category", {})
    cat_table = ""
    if by_cat:
        cat_table = "\n### 按类别分层\n\n| 类别 | 数量 | Hit Rate | Recall@K |\n|---|---|---|---|\n"
        for cat, stats in sorted(by_cat.items()):
            cat_table += (
                f"| {cat} | {stats['count']} | "
                f"{stats['hit_rate'] * 100:.2f}% | "
                f"{stats['recall_at_k'] * 100:.2f}% |\n"
            )

    # 失败案例摘要
    fail_section = ""
    if failed:
        fail_section = f"\n### 失败案例摘要（共 {total - sum(1 for r in rag.get('per_query_results', []) if r.get('hit', True))} 条未命中）\n\n"
        for case in failed[:5]:
            fail_section += (
                f"- **{case['query_id']}**: `{case['query']}`\n"
                f"  - 期望: `{case.get('relevant_chunk_ids', [])}`\n"
                f"  - 实际 Top-3: `{case.get('retrieved_ids', [])[:3]}`\n\n"
            )

    return f"""## RAG 向量检索详情

- 评估集大小：{total} 条 query
- 检索深度：Top-{top_k}
{diff_table}{cat_table}{fail_section}"""


def _render_faithfulness_detail(faith: dict) -> str:
    """渲染幻觉检测分层详情。"""
    if not faith:
        return "## 幻觉检测详情\n\n> 结果文件缺失，请先运行 eval_faithfulness.py\n\n"

    faith_a = faith.get("faithfulness_on_dataset_a", {})
    faith_b = faith.get("fallback_precision_on_dataset_b", {})
    threshold = faith.get("fallback_threshold_used", 0.3)

    # 分数分布
    dist = faith_a.get("score_distribution", {})
    dist_table = ""
    if dist:
        dist_table = (
            "\n### 幻觉分数分布（Dataset A）\n\n| 分数区间 | 数量 | 说明 |\n|---|---|---|\n"
        )
        desc = {
            "0.0-0.1": "高度忠实",
            "0.1-0.2": "较为忠实",
            "0.2-0.3": "轻微偏差",
            "0.3-0.5": "触发 Fallback",
            "0.5-1.0": "严重幻觉",
        }
        for interval, count in dist.items():
            dist_table += f"| `{interval}` | {count} | {desc.get(interval, '')} |\n"

    # 混淆矩阵
    tp = faith_b.get("true_positive", 0)
    fp = faith_b.get("false_positive", 0)
    tn = faith_b.get("true_negative", 0)
    fn = faith_b.get("false_negative", 0)
    matrix = f"""
### 幻觉检测混淆矩阵（Dataset B，阈值={threshold}）

|  | 预测: Fallback | 预测: 放行 |
|---|---|---|
| **实际: 幻觉（负样本）** | TP={tp} ✓ | **FN={fn} ✗（漏报）** |
| **实际: 忠实（正样本）** | FP={fp} ✗（误报）| TN={tn} ✓ |

> FN（漏报幻觉）是最危险的错误，会将错误的医疗信息传递给用户。
"""

    # 漏报案例
    fn_cases = faith_b.get("false_negatives", [])
    fn_section = ""
    if fn_cases:
        fn_section = "\n### 漏报案例（FN，需重点关注）\n\n"
        for case in fn_cases[:3]:
            fn_section += (
                f"- `{case['query'][:50]}...`\n"
                f"  - 注入类型: {case['injection_type']} | "
                f"幻觉分数: {case['hallucination_score']} < 阈值 {case['threshold']}\n\n"
            )

    return f"""## 幻觉检测详情
{dist_table}{matrix}{fn_section}"""


def _render_intent_detail(intent: dict) -> str:
    """渲染意图分类分层详情。"""
    if not intent:
        return "## 意图分类详情\n\n> 结果文件缺失，请先运行 eval_intent.py\n\n"

    per_class = intent.get("per_class_metrics", {})
    class_table = ""
    if per_class:
        class_table = "\n### 各类别指标\n\n| 意图类别 | Precision | Recall | F1 | 样本数 |\n|---|---|---|---|---|\n"
        for intent_type, metrics in per_class.items():
            # EMERGENCY 用加粗标记（安全关键）
            marker = " ⚠️" if intent_type == "EMERGENCY" else ""
            class_table += (
                f"| **{intent_type}**{marker} | "
                f"{metrics['precision'] * 100:.2f}% | "
                f"{metrics['recall'] * 100:.2f}% | "
                f"{metrics['f1'] * 100:.2f}% | "
                f"{metrics['support']} |\n"
            )

    by_diff = intent.get("by_difficulty", {})
    diff_table = ""
    if by_diff:
        diff_table = "\n### 按难度分层\n\n| 难度 | 数量 | Accuracy |\n|---|---|---|\n"
        for diff, stats in sorted(by_diff.items()):
            diff_table += f"| {diff} | {stats['count']} | {stats['accuracy'] * 100:.2f}% |\n"

    composite_total = intent.get("composite_total", 0)
    composite_acc = intent.get("composite_accuracy", 0)
    composite_partial = intent.get("composite_partial_rate", 0)
    composite_section = ""
    if composite_total > 0:
        composite_section = f"""
### 复合意图评估（{composite_total} 条）

| 指标 | 数值 | 说明 |
|---|---|---|
| 完全匹配率 | {composite_acc * 100:.2f}% | 所有意图都识别正确 |
| 部分匹配率 | {composite_partial * 100:.2f}% | 至少命中一个意图 |
"""

    return f"""## 意图分类详情

- 评估集大小：{intent.get("total", 0)} 条 query
- EMERGENCY 专项 Recall：**{intent.get("emergency_recall", 0) * 100:.2f}%**（{intent.get("emergency_total", 0)} 条）
{class_table}{diff_table}{composite_section}"""


def _render_parallel_detail(parallel: dict) -> str:
    """渲染并行加速比详情。"""
    if not parallel:
        return "## 并行加速比详情\n\n> 结果文件缺失，请先运行 eval_parallel.py\n\n"

    per_case = parallel.get("per_case_results", [])
    case_table = ""
    if per_case:
        case_table = (
            "\n### 各用例加速比\n\n| 用例 | 串行(ms) | 并行(ms) | 加速比 |\n|---|---|---|---|\n"
        )
        for case in per_case:
            case_table += (
                f"| {case['case_id']} | "
                f"{case['serial_mean_ms']:.0f} | "
                f"{case['parallel_mean_ms']:.0f} | "
                f"**{case['speedup_ratio']:.2f}x** |\n"
            )
        # 合计行
        serial_sum = parallel.get("mean_serial_ms", 0)
        parallel_sum = parallel.get("mean_parallel_ms", 0)
        speedup = parallel.get("mean_speedup_ratio", 0)
        case_table += (
            f"| **均值** | **{serial_sum:.0f}** | **{parallel_sum:.0f}** | **{speedup:.2f}x** |\n"
        )

    timing = parallel.get("timing_breakdown", {})
    timing_section = ""
    if timing:
        timing_section = (
            "\n### 节点耗时分解（并行模式均值）\n\n| 节点 | 平均耗时(ms) |\n|---|---|\n"
        )
        for node, ms in sorted(timing.items(), key=lambda x: x[1], reverse=True):
            timing_section += f"| {node} | {ms:.0f} |\n"

    return f"""## 并行加速比详情

- 测试用例数：{parallel.get("total_cases", 0)} 条（均为 MEDICAL + DEVICE 复合意图）
- 每用例重复测量：{parallel.get("n_repeats", 3)} 次（取均值消除网络抖动）
- 加速比范围：{parallel.get("speedup_min", 0):.2f}x ~ {parallel.get("speedup_max", 0):.2f}x
- 标准差：±{parallel.get("speedup_std", 0):.3f}
{case_table}{timing_section}"""


def _render_resume_template(
    rag: dict,
    faith: dict,
    intent: dict,
    parallel: dict,
) -> str:
    """
    渲染简历表述模板（可直接复制）。

    Args:
        rag / faith / intent / parallel: 各模块评估结果

    Returns:
        str: 包含可量化指标的简历模板
    """

    def pct(d: dict[str, Any], *keys: str, default: str = "N/A") -> str:
        """从嵌套 dict 中取值并格式化为百分比。"""
        val: Any = d
        for k in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(k, default)
        if isinstance(val, (int, float)):
            return f"{val * 100:.1f}%"
        return str(val)

    hit_rate = pct(rag, "hit_rate")
    recall = pct(rag, "recall_at_k")
    top_k = rag.get("top_k", 3)
    faith_score = pct(faith, "faithfulness_on_dataset_a", "faithfulness")
    fallback_f1 = pct(faith, "fallback_precision_on_dataset_b", "f1")
    fallback_recall = pct(faith, "fallback_precision_on_dataset_b", "recall")
    accuracy = pct(intent, "accuracy")
    emg_recall = pct(intent, "emergency_recall")
    speedup = parallel.get("mean_speedup_ratio", "N/A")
    serial_ms = parallel.get("mean_serial_ms", "N/A")
    parallel_ms = parallel.get("mean_parallel_ms", "N/A")

    # 计算并行节省比例
    time_saved_pct = "N/A"
    if (
        isinstance(serial_ms, (int, float))
        and isinstance(parallel_ms, (int, float))
        and serial_ms > 0
    ):
        time_saved_pct = f"{(1 - parallel_ms / serial_ms) * 100:.0f}%"

    return f"""## 简历表述模板

> 以下为基于本次评估结果的量化表述，可直接复制到简历项目经历中。
> 请根据实际数值修正括号内的占位符。

```
Silver Pilot — 面向老年人医疗健康 RAG + Multi-Agent 系统
Python | LangGraph | Neo4j | Milvus | DashScope

- 设计 GraphRAG + 双向量库（QA 库 + 知识库）混合检索架构，
  在 100 条医疗 QA 标注集评估中：
  Hit Rate@{top_k} = {hit_rate}，Recall@{top_k} = {recall}

- 实现基于 Pydantic 结构化输出的两阶段幻觉检测机制：
  Faithfulness Score = {faith_score}，
  幻觉检测 Fallback F1 = {fallback_f1}（Recall = {fallback_recall}，漏报率受控）

- 构建 Supervisor 意图路由模块，在 150 条多意图测试集上：
  整体 Accuracy = {accuracy}，EMERGENCY 类 Recall = {emg_recall}（安全关键，零漏报）

- 基于 LangGraph Send 实现多意图并行分发：
  复合意图场景端到端延迟从 {serial_ms}ms 降至 {parallel_ms}ms，
  加速比 {speedup}x，节省 {time_saved_pct} 响应时间
```
"""


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────


def generate_report(output_path: Path = OUTPUT_PATH) -> Path:
    """
    加载所有评估结果并生成 Markdown 报告。

    Args:
        output_path: 报告输出路径

    Returns:
        Path: 报告文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 加载各模块结果 ──
    logger.info("加载评估结果文件...")
    rag = _load_json(RESULT_FILES["rag_retrieval"])
    faith = _load_json(RESULT_FILES["faithfulness"])
    intent = _load_json(RESULT_FILES["intent"])
    parallel = _load_json(RESULT_FILES["parallel"])

    loaded = sum(1 for d in [rag, faith, intent, parallel] if d)
    logger.info(f"成功加载 {loaded}/4 个结果文件")

    # ── 拼装报告 ──
    sections = [
        _render_header(),
        _render_summary_table(rag, faith, intent, parallel),
        _render_rag_detail(rag),
        _render_faithfulness_detail(faith),
        _render_intent_detail(intent),
        _render_parallel_detail(parallel),
        _render_resume_template(rag, faith, intent, parallel),
    ]

    report_text = "\n".join(sections)

    # ── 写入文件 ──
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"报告已生成: {output_path}")
    logger.info(f"报告大小: {len(report_text)} 字符")

    return output_path


if __name__ == "__main__":
    generate_report()
