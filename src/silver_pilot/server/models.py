"""
模块名称：models
功能描述：Server 层的 Pydantic 请求/响应模型定义，为 REST API 和 WebSocket
         通信提供类型安全的数据契约。
"""

import time
from typing import Any

from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────
# Session
# ────────────────────────────────────────────────────────────


class SessionMeta(BaseModel):
    """会话元数据（不含消息体，仅用于列表展示）。"""

    session_id: str
    name: str = "新对话"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    message_count: int = 0
    user_id: str = "default_user"


class SessionCreate(BaseModel):
    """创建会话请求。"""

    name: str = "新对话"
    user_id: str = "default_user"


class MessageRecord(BaseModel):
    """持久化的单条消息。"""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = Field(default_factory=time.time)
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# WebSocket 协议
# ────────────────────────────────────────────────────────────


class WSIncoming(BaseModel):
    """客户端发送的 WebSocket 消息。"""

    type: str  # "message" | "hitl_response"
    content: str = ""
    modality: dict[str, bool] = Field(
        default_factory=lambda: {"text": True, "audio": False, "image": False}
    )
    image_path: str = ""
    audio_path: str = ""
    confirmed: bool | None = None  # HITL 确认


class WSOutgoing(BaseModel):
    """服务端发送的 WebSocket 消息。"""

    type: str  # "node_start" | "node_end" | "hitl_request" | "response" | "error"
    node: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    content: str = ""
    message: str = ""
    duration_ms: float = 0
    event_seq: int = 0
    group_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    debug: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# Profile / Health / Reminders
# ────────────────────────────────────────────────────────────


class HealthOverview(BaseModel):
    """首页健康概览数据。"""

    blood_pressure: str = "138/85"
    blood_pressure_unit: str = "mmHg"
    blood_pressure_trend: str = "up"
    blood_sugar: str = "6.8"
    blood_sugar_unit: str = "mmol/L"
    blood_sugar_trend: str = "normal"
    completed_reminders: int = 3
    total_reminders: int = 5
    next_reminder_time: str = "12:00"
    next_reminder_msg: str = "吃二甲双胍"


class ReminderItem(BaseModel):
    """单条提醒。"""

    id: str
    time: str
    message: str
    repeat: str = "none"
    active: bool = True
    done: bool = False
