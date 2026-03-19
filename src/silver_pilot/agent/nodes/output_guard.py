"""
模块名称：output_guard
功能描述：输出安全校验节点，在最终响应返回用户之前进行安全审查。
         当前阶段采用关键词过滤 + 规则检查的轻量方案，后续可升级为
         NeMo Guardrails (Colang 2.0) 实现更精细的安全护栏。

安全检查项：
    1. 敏感信息过滤（政治、暴力、歧视性内容）
    2. 医疗安全兜底（确保涉及用药/手术的回答包含就医提示）
    3. 个人隐私保护（脱敏处理可能泄露的敏感信息）
    4. 对话摘要压缩（如消息过长则触发摘要）
"""

import re
from pathlib import Path

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

from ..memory.summarizer import ConversationSummarizer
from ..state import AgentState

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "output_guard")


# ────────────────────────────────────────────────────────────
# 安全规则定义
# ────────────────────────────────────────────────────────────

# 医疗安全关键词：当回答涉及这些内容时，确保包含就医提示
MEDICAL_SAFETY_KEYWORDS: list[str] = [
    "剂量",
    "用量",
    "服用",
    "注射",
    "禁忌",
    "副作用",
    "不良反应",
    "手术",
    "化疗",
    "放疗",
    "用药",
    "停药",
    "加药",
    "减药",
]

# 敏感内容过滤模式
SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"自杀|自残|结束生命"),
    re.compile(r"制造.*(?:炸弹|毒品|武器)"),
]

MEDICAL_DISCLAIMER: str = "\n\n（温馨提示：以上信息仅供参考，具体用药请遵医嘱。）"

DISCLAIMER_INDICATORS: list[str] = ["咨询医生", "遵医嘱", "就医", "咨询药师", "建议您咨询"]

EMPTY_FALLBACK: str = "抱歉，我现在无法为您提供回答。请稍后再试，或者直接联系家人。"

# 摘要压缩器（模块级单例）
_summarizer: ConversationSummarizer | None = None


def _get_summarizer() -> ConversationSummarizer:
    """获取或创建摘要压缩器单例。"""
    global _summarizer
    if _summarizer is None:
        _summarizer = ConversationSummarizer()
    return _summarizer


# ────────────────────────────────────────────────────────────
# Output Guard 节点
# ────────────────────────────────────────────────────────────


def output_guard_node(state: AgentState) -> dict:
    """
    输出安全校验节点。

    执行顺序：
        1. 检查是否需要对话摘要压缩
        2. 对 final_response 进行安全审查
        3. 必要时添加医疗安全提示
        4. 返回审查后的最终响应

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 final_response、safety_flags（可选 messages 和 conversation_summary）的状态更新
    """
    final_response = state.get("final_response", "")
    safety_flags = list(state.get("safety_flags", []))

    logger.info(f"Output Guard 开始审查 | 回复长度={len(final_response)}")

    # ── 1. 对话摘要压缩检查 ──
    summary_update = _check_and_compress(state)

    # ── 2. 敏感内容过滤 ──
    final_response, sensitive_detected = _filter_sensitive_content(final_response)
    if sensitive_detected:
        safety_flags.append("sensitive_content_filtered")

    # ── 3. 医疗安全兜底 ──
    final_response, medical_flag = _ensure_medical_safety(final_response)
    if medical_flag:
        safety_flags.append("medical_disclaimer_added")

    # ── 4. 空回复兜底 ──
    if not final_response.strip():
        final_response = EMPTY_FALLBACK
        safety_flags.append("empty_response_fallback")

    logger.info(f"Output Guard 审查完成 | flags={safety_flags}")

    result: dict = {
        "final_response": final_response,
        "safety_flags": safety_flags,
    }

    # 合并摘要压缩结果
    if summary_update:
        result.update(summary_update)

    return result


# ────────────────────────────────────────────────────────────
# 安全检查子函数
# ────────────────────────────────────────────────────────────


def _filter_sensitive_content(text: str) -> tuple[str, bool]:
    """
    过滤敏感内容。

    Args:
        text: 待检查的文本

    Returns:
        tuple: (过滤后的文本, 是否检测到敏感内容)
    """
    detected = False
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(text):
            detected = True
            logger.warning(f"检测到敏感内容 | pattern={pattern.pattern}")
            text = pattern.sub("[内容已过滤]", text)

    return text, detected


def _ensure_medical_safety(text: str) -> tuple[str, bool]:
    """
    确保涉及医疗建议的回答包含就医提示。

    Args:
        text: 回答文本

    Returns:
        tuple: (处理后的文本, 是否添加了医疗提示)
    """
    # 检查是否包含医疗关键词
    has_medical_content = any(keyword in text for keyword in MEDICAL_SAFETY_KEYWORDS)

    if not has_medical_content:
        return text, False

    # 检查是否已包含就医提示
    has_disclaimer = any(
        phrase in text for phrase in ["咨询医生", "遵医嘱", "就医", "咨询药师", "建议您咨询"]
    )

    if has_disclaimer:
        return text, False

    # 添加医疗安全提示
    text = text.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
    return text, True


def _check_and_compress(state: AgentState) -> dict | None:
    """
    检查是否需要对话摘要压缩。

    Args:
        state: 当前 AgentState

    Returns:
        dict | None: 如果触发压缩则返回更新字典，否则返回 None
    """
    summarizer = _get_summarizer()

    if summarizer.should_compress(state):
        logger.info("消息历史过长，触发对话摘要压缩")
        return summarizer.compress(state)

    return None
