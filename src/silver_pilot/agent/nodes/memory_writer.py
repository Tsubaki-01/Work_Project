"""
模块名称：memory_writer
功能描述：记忆写入节点。每隔固定轮数调用 LLM 从近期对话中提取用户健康信息
         （疾病、过敏、用药），增量更新到长期画像。


提取时机：
    每 EXTRACT_INTERVAL 轮触发一次提取（默认 6 轮），避免每轮都调 LLM。
    会话结束前（由 output_guard → memory_writer 链路保证）也会执行一次。
"""

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm_parse
from ..memory.user_profile import UserProfileManager
from ..state import AgentState
from .helpers import build_profile_summary, content_to_text

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "agent", "memory_writer")

# ================= 默认配置 =================
EXTRACT_MODEL: str = config.MEMORY_WRITER_AGENT
EXTRACT_INTERVAL: int = config.MEMORY_WRITER_EXTRACT_INTERVAL
"""每隔多少轮消息触发一次 LLM 提取。"""

PROMPT_TEMPLATE: str = "agent/profile_extract"

# ── 模块级单例 ──
_manager: UserProfileManager | None = None


def set_profile_manager(manager: UserProfileManager) -> None:
    """注入 UserProfileManager 实例（系统启动时调用）。"""
    global _manager
    _manager = manager


# ────────────────────────────────────────────────────────────
# Pydantic 结构化输出
# ────────────────────────────────────────────────────────────


class MedicationItem(BaseModel):
    name: str = Field(description="药品名称")
    dosage: str = Field(default="", description="剂量，如 '100mg/日'")


class ProfileExtractOutput(BaseModel):
    """LLM 从对话中提取的新增健康信息。"""

    chronic_diseases: list[str] = Field(default_factory=list, description="新发现的慢性疾病")
    allergies: list[str] = Field(default_factory=list, description="新发现的过敏原")
    current_medications: list[MedicationItem] = Field(
        default_factory=list, description="新发现的用药信息"
    )


# ────────────────────────────────────────────────────────────
# 节点函数
# ────────────────────────────────────────────────────────────


def memory_writer_node(state: AgentState) -> dict:
    """
    记忆写入节点：定期用 LLM 从对话中提取健康信息并持久化。

    触发条件：消息数 ≥ 2 且距上次提取已过 EXTRACT_INTERVAL 轮。
    会话结束时（图拓扑保证走到此节点）也执行一次。

    """
    if _manager is None:
        logger.error("UserProfileManager 未初始化")
        return {}

    user_id = state.get("user_profile", {}).get("user_id", "")
    if not user_id:
        return {}

    total_turns = state.get("total_turns", 0)
    if total_turns < 2:
        return {}

    # 节流：距上次提取不足 EXTRACT_INTERVAL 轮则跳过
    last_extracted_at = state.get("last_profile_extract_at", 0)
    if total_turns - last_extracted_at < EXTRACT_INTERVAL:
        return {}

    logger.info(f"触发画像提取 | user={user_id}")

    # LLM 提取
    extracted = _extract_from_conversation(state.get("messages", []), state.get("user_profile", {}))
    if extracted is None:
        logger.warning("LLM 画像提取失败，跳过")
        return {}

    # 构建增量更新并持久化
    updates = _build_updates(extracted)
    if updates:
        _manager.update_profile(user_id, updates)
        logger.info(f"画像更新 | user={user_id} | fields={list(updates.keys())}")

    return {"last_profile_extract_at": total_turns}


# ────────────────────────────────────────────────────────────
# LLM 提取
# ────────────────────────────────────────────────────────────


def _extract_from_conversation(
    messages: list[AnyMessage], existing_profile: dict
) -> ProfileExtractOutput | None:
    """调用 LLM 从近期对话中提取新增的用户健康信息。"""
    # 只取最近一段对话，控制 context 长度
    recent = messages[-(EXTRACT_INTERVAL * 2) :]
    conversation_text = content_to_text(recent)
    existing_summary = build_profile_summary(existing_profile)

    prompt_messages = prompt_manager.build_prompt(
        PROMPT_TEMPLATE,
        conversation_text=conversation_text,
        existing_profile=existing_summary,
    )
    return call_llm_parse(EXTRACT_MODEL, prompt_messages, ProfileExtractOutput, temperature=0.0)


# ────────────────────────────────────────────────────────────
# 增量更新
# ────────────────────────────────────────────────────────────


def _build_updates(extracted: ProfileExtractOutput) -> dict:
    """将提取结果转为 update_profile 的增量格式。空列表不写入。"""
    updates: dict = {}

    if extracted.chronic_diseases:
        updates["chronic_diseases"] = extracted.chronic_diseases

    if extracted.allergies:
        updates["allergies"] = extracted.allergies

    if extracted.current_medications:
        updates["current_medications"] = [
            {"name": med.name, "dosage": med.dosage} for med in extracted.current_medications
        ]

    return updates
