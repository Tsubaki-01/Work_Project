"""
模块名称：session_store
功能描述：会话元数据的内存存储。管理 Session 的 CRUD 和消息记录持久化。
         开发阶段使用内存字典，生产可切换 Redis/SQLite。

设计说明：
    - session_id 同时作为 LangGraph 的 thread_id
    - 消息记录独立于 LangGraph Checkpointer 存储（Checkpointer 存的是 AgentState，
      这里存的是给前端展示用的简化消息列表）
"""

from __future__ import annotations

import time
import uuid

from .models import MessageRecord, SessionMeta


class SessionStore:
    """
    会话元数据内存存储。

    使用示例::

        store = SessionStore()
        meta = store.create("药品咨询", user_id="user_001")
        store.add_message(meta.session_id, MessageRecord(role="user", content="你好"))
        sessions = store.list_sessions("user_001")
    """

    def __init__(self) -> None:
        # session_id → SessionMeta
        self._sessions: dict[str, SessionMeta] = {}
        # session_id → list[MessageRecord]
        self._messages: dict[str, list[MessageRecord]] = {}

    def create(self, name: str = "新对话", user_id: str = "default_user") -> SessionMeta:
        """创建新会话，返回元数据。"""
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        meta = SessionMeta(
            session_id=sid,
            name=name,
            created_at=now,
            updated_at=now,
            message_count=0,
            user_id=user_id,
        )
        self._sessions[sid] = meta
        self._messages[sid] = []

        # 自动添加欢迎消息
        self.add_message(
            sid,
            MessageRecord(
                role="assistant",
                content="您好！我是小银，您的健康助手。有什么可以帮您的吗？",
            ),
        )
        return meta

    def get(self, session_id: str) -> SessionMeta | None:
        """获取会话元数据。"""
        return self._sessions.get(session_id)

    def list_sessions(self, user_id: str = "default_user") -> list[SessionMeta]:
        """列出指定用户的所有会话，按更新时间倒序。"""
        sessions = [s for s in self._sessions.values() if s.user_id == user_id]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        """删除会话。"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._messages.pop(session_id, None)
            return True
        return False

    def add_message(self, session_id: str, message: MessageRecord) -> None:
        """向会话追加消息并更新元数据。"""
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(message)

        meta = self._sessions.get(session_id)
        if meta:
            meta.message_count = len(self._messages[session_id])
            meta.updated_at = time.time()
            # 自动用首条用户消息命名
            if meta.name == "新对话" and message.role == "user":
                meta.name = message.content[:15]

    def get_messages(self, session_id: str) -> list[MessageRecord]:
        """获取会话的所有消息。"""
        return self._messages.get(session_id, [])
