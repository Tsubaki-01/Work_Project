"""
模块名称：eval_parallel
功能描述：量化 LangGraph Send 机制在复合意图场景下的并行加速比。

实验设计：
    使用同一批复合意图用例（MEDICAL_QUERY + DEVICE_CONTROL），
    分别在以下两种模式下运行，记录端到端 wall-clock 时间：

    串行模式：将复合意图拆分为两次独立的单意图调用（模拟无并行能力）
              总时间 = 第一次图执行耗时 + 第二次图执行耗时

    并行模式：正常走复合意图，Supervisor 返回 list[Send] 并行分发
              总时间 = 单次图执行耗时（两个 agent 并发运行）

    加速比 = 串行总时间 / 并行总时间

注意事项：
    - 使用 skip_rag=True 初始化 Agent，避免依赖数据库（Neo4j/Milvus）
    - 但需要真实 LLM 调用（DashScope），所以需要有效的 DASHSCOPE_API_KEY
    - 重复测量 N_REPEATS 次取均值，减少网络抖动的影响
    - 串行基准：两次调用使用不同 thread_id，避免状态干扰

输出：data/eval/results/parallel_result.json
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from silver_pilot.agent import create_initial_state, initialize_agent
from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "eval", "eval_parallel")

# ================= 路径配置 =================
OUTPUT_PATH = Path("data/eval/results/parallel_result.json")

# ================= 实验配置 =================
# 重复测量次数（取均值平滑网络抖动）
N_REPEATS: int = 3
# 每次测量后的冷却间隔（秒），避免 API 限流
COOLDOWN_SECONDS: float = 2.0

# ── 测试用例：10 条典型的 MEDICAL + DEVICE 复合意图 ──
# 精心设计：医疗问题和设备操作各自明确，确保 Supervisor 能正确识别双意图
COMPOSITE_TEST_CASES: list[dict] = [
    {
        "id": "comp_001",
        "composite_input": "阿司匹林一天吃几次，帮我明天早上7点设个吃药提醒",
        "medical_only": "阿司匹林一天吃几次",
        "device_only": "帮我明天早上7点设个提醒",
    },
    {
        "id": "comp_002",
        "composite_input": "二甲双胍饭前还是饭后吃，顺便帮我查一下今天天气",
        "medical_only": "二甲双胍饭前还是饭后吃",
        "device_only": "帮我查一下今天的天气",
    },
    {
        "id": "comp_003",
        "composite_input": "高血压患者能吃阿司匹林吗，把卧室灯关一下",
        "medical_only": "高血压患者能吃阿司匹林吗",
        "device_only": "把卧室灯关一下",
    },
    {
        "id": "comp_004",
        "composite_input": "华法林和布洛芬能一起吃吗，明天下午3点提醒我去医院",
        "medical_only": "华法林和布洛芬能一起吃吗",
        "device_only": "明天下午3点提醒我去医院",
    },
    {
        "id": "comp_005",
        "composite_input": "糖尿病患者血糖多少算正常，空调温度设到26度",
        "medical_only": "糖尿病患者血糖多少算正常",
        "device_only": "空调温度设到26度",
    },
    {
        "id": "comp_006",
        "composite_input": "降压药有哪些副作用，帮我查明天南京的天气",
        "medical_only": "降压药有哪些副作用",
        "device_only": "帮我查明天南京的天气",
    },
    {
        "id": "comp_007",
        "composite_input": "心脏病患者能不能吃感冒药，把客厅电视关掉",
        "medical_only": "心脏病患者能不能吃感冒药",
        "device_only": "把客厅电视关掉",
    },
    {
        "id": "comp_008",
        "composite_input": "老人骨质疏松应该补充什么，每天早上8点提醒我喝牛奶",
        "medical_only": "老人骨质疏松应该补充什么",
        "device_only": "每天早上8点提醒我喝牛奶",
    },
    {
        "id": "comp_009",
        "composite_input": "氨氯地平和卡托普利能一起吃吗，帮我在日历上记一下后天去复查",
        "medical_only": "氨氯地平和卡托普利能一起吃吗",
        "device_only": "帮我在日历上记一下后天去复查",
    },
    {
        "id": "comp_010",
        "composite_input": "胃溃疡患者能吃阿司匹林吗，把热水器关了",
        "medical_only": "胃溃疡患者能吃阿司匹林吗",
        "device_only": "把热水器关了",
    },
]


# ────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────


@dataclass
class SingleCaseResult:
    """单条测试用例的计时结果。"""

    case_id: str
    # 串行模式：两次单意图调用的总耗时（每次均值 × 2）
    serial_times_ms: list[float] = field(default_factory=list)
    serial_mean_ms: float = 0.0
    # 并行模式：一次复合意图调用的耗时
    parallel_times_ms: list[float] = field(default_factory=list)
    parallel_mean_ms: float = 0.0
    # 加速比
    speedup_ratio: float = 0.0


@dataclass
class ParallelEvalResult:
    """并行加速比完整评估结果。"""

    total_cases: int = 0
    n_repeats: int = N_REPEATS
    # 核心指标
    mean_serial_ms: float = 0.0
    mean_parallel_ms: float = 0.0
    mean_speedup_ratio: float = 0.0
    # 95% 置信区间（简化版：均值 ± 标准差）
    speedup_std: float = 0.0
    speedup_min: float = 0.0
    speedup_max: float = 0.0
    # 逐条结果
    per_case_results: list[dict] = field(default_factory=list)
    # 节点耗时分解（从 LangGraph stream events 提取）
    timing_breakdown: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 计时工具
# ────────────────────────────────────────────────────────────


def _run_single_intent(
    graph: Any,
    input_text: str,
    thread_id: str,
) -> float:
    """
    运行单意图 query，返回 wall-clock 耗时（毫秒）。
    使用独立的 thread_id 避免状态污染。

    Args:
        graph:      已编译的 LangGraph 图
        input_text: 单意图用户输入
        thread_id:  LangGraph checkpointer 线程 ID

    Returns:
        float: 耗时（毫秒）
    """
    state = create_initial_state()
    state["messages"] = [HumanMessage(content=input_text)]
    cfg = {"configurable": {"thread_id": thread_id}}

    t_start = time.perf_counter()
    # 遍历所有 stream 事件（保证图执行完成）
    for _ in graph.stream(state, config=cfg, stream_mode="updates"):
        pass
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return elapsed_ms


def _run_composite_intent(
    graph: Any,
    composite_input: str,
    thread_id: str,
) -> tuple[float, dict[str, float]]:
    """
    运行复合意图 query（走 Send 并行分发），返回耗时和节点计时分解。

    Args:
        graph:          已编译的 LangGraph 图
        composite_input: 包含多意图的用户输入
        thread_id:      LangGraph checkpointer 线程 ID

    Returns:
        tuple[float, dict[str, float]]: (耗时毫秒, 节点名称→耗时 dict)
    """
    state = create_initial_state()
    state["messages"] = [HumanMessage(content=composite_input)]
    cfg = {"configurable": {"thread_id": thread_id}}

    node_timings: dict[str, float] = {}
    prev_tick = time.perf_counter()

    t_start = time.perf_counter()
    for chunk in graph.stream(state, config=cfg, stream_mode="updates"):
        now_tick = time.perf_counter()
        elapsed_chunk = (now_tick - prev_tick) * 1000
        for node_name in chunk:
            node_timings[node_name] = node_timings.get(node_name, 0) + elapsed_chunk
        prev_tick = now_tick

    total_ms = (time.perf_counter() - t_start) * 1000
    return total_ms, node_timings


# ────────────────────────────────────────────────────────────
# 主评估函数
# ────────────────────────────────────────────────────────────


def run_eval(
    output_path: Path = OUTPUT_PATH,
    n_repeats: int = N_REPEATS,
) -> ParallelEvalResult:
    """
    执行并行加速比评估。

    Args:
        output_path: 结果输出路径
        n_repeats:   每条用例的重复测量次数

    Returns:
        ParallelEvalResult
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 初始化 Agent（skip_rag=True 避免依赖 Neo4j/Milvus）──
    logger.info("初始化 Agent（skip_rag=True）...")
    logger.info("注意：需要有效的 DASHSCOPE_API_KEY 进行真实 LLM 调用")
    graph = initialize_agent(skip_rag=True)
    logger.info("Agent 初始化完成")

    per_case_results: list[SingleCaseResult] = []
    all_node_timings: dict[str, list[float]] = {}  # 节点名→各次耗时列表

    # ── 逐条用例评估 ──
    for case in COMPOSITE_TEST_CASES:
        case_id = case["id"]
        composite_input = case["composite_input"]
        medical_only = case["medical_only"]
        device_only = case["device_only"]

        logger.info(f"=== 用例 {case_id}: {composite_input[:50]}...")

        case_result = SingleCaseResult(case_id=case_id)

        # ── 重复 N 次取均值 ──
        for repeat in range(n_repeats):
            logger.info(f"  第 {repeat + 1}/{n_repeats} 次测量...")

            # ── 串行基准：medical_only 和 device_only 各调用一次 ──
            t_medical = _run_single_intent(
                graph,
                medical_only,
                thread_id=f"{case_id}_serial_med_{repeat}",
            )
            time.sleep(COOLDOWN_SECONDS)  # 冷却，避免 LLM API 限流

            t_device = _run_single_intent(
                graph,
                device_only,
                thread_id=f"{case_id}_serial_dev_{repeat}",
            )
            time.sleep(COOLDOWN_SECONDS)

            serial_total = t_medical + t_device
            case_result.serial_times_ms.append(serial_total)
            logger.info(
                f"    串行: medical={t_medical:.0f}ms + device={t_device:.0f}ms = {serial_total:.0f}ms"
            )

            # ── 并行模式：复合意图一次调用 ──
            t_parallel, node_timings = _run_composite_intent(
                graph,
                composite_input,
                thread_id=f"{case_id}_parallel_{repeat}",
            )
            time.sleep(COOLDOWN_SECONDS)

            case_result.parallel_times_ms.append(t_parallel)
            logger.info(f"    并行: {t_parallel:.0f}ms")
            logger.info(f"    加速比（本次）: {serial_total / t_parallel:.2f}x")

            # 累积节点计时
            for node, t in node_timings.items():
                all_node_timings.setdefault(node, []).append(t)

        # ── 计算该用例的均值和加速比 ──
        case_result.serial_mean_ms = round(sum(case_result.serial_times_ms) / n_repeats, 1)
        case_result.parallel_mean_ms = round(sum(case_result.parallel_times_ms) / n_repeats, 1)
        case_result.speedup_ratio = round(
            case_result.serial_mean_ms / case_result.parallel_mean_ms
            if case_result.parallel_mean_ms > 0
            else 0.0,
            3,
        )

        logger.info(
            f"  用例 {case_id} 均值: 串行={case_result.serial_mean_ms:.1f}ms | "
            f"并行={case_result.parallel_mean_ms:.1f}ms | "
            f"加速比={case_result.speedup_ratio:.2f}x"
        )
        per_case_results.append(case_result)

    # ── 汇总全局指标 ──
    import statistics

    n = len(per_case_results)
    speedup_ratios = [r.speedup_ratio for r in per_case_results]
    mean_serial = sum(r.serial_mean_ms for r in per_case_results) / n
    mean_parallel = sum(r.parallel_mean_ms for r in per_case_results) / n
    mean_speedup = sum(speedup_ratios) / n

    speedup_std = statistics.stdev(speedup_ratios) if n > 1 else 0.0

    # 各节点平均耗时（用于分解分析）
    timing_breakdown = {
        node: round(sum(times) / len(times), 1) for node, times in all_node_timings.items()
    }

    eval_result = ParallelEvalResult(
        total_cases=n,
        n_repeats=n_repeats,
        mean_serial_ms=round(mean_serial, 1),
        mean_parallel_ms=round(mean_parallel, 1),
        mean_speedup_ratio=round(mean_speedup, 3),
        speedup_std=round(speedup_std, 3),
        speedup_min=round(min(speedup_ratios), 3),
        speedup_max=round(max(speedup_ratios), 3),
        per_case_results=[
            {
                "case_id": r.case_id,
                "serial_mean_ms": r.serial_mean_ms,
                "parallel_mean_ms": r.parallel_mean_ms,
                "speedup_ratio": r.speedup_ratio,
            }
            for r in per_case_results
        ],
        timing_breakdown=timing_breakdown,
    )

    # ── 输出汇总 ──
    logger.info("=" * 50)
    logger.info(f"评估用例数:     {n}")
    logger.info(f"重复测量次数:   {n_repeats}")
    logger.info(f"串行均值耗时:   {mean_serial:.1f} ms")
    logger.info(f"并行均值耗时:   {mean_parallel:.1f} ms")
    logger.info(f"平均加速比:     {mean_speedup:.2f}x  (std={speedup_std:.3f})")
    logger.info(f"加速比范围:     [{min(speedup_ratios):.2f}x, {max(speedup_ratios):.2f}x]")
    logger.info(
        f"并行节省耗时:   {mean_serial - mean_parallel:.1f} ms ({(1 - mean_parallel / mean_serial) * 100:.1f}%)"
    )
    logger.info("=" * 50)

    # ── 写入结果文件 ──
    import dataclasses

    data = dataclasses.asdict(eval_result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"并行评估结果已保存至: {output_path}")

    return eval_result


if __name__ == "__main__":
    run_eval()
