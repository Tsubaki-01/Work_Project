"""
模块名称：user_profile
功能描述：长期用户画像管理器，基于 SQLite 实现跨会话持久化的用户健康画像存储。
         记录用户的慢性病、过敏史、当前用药、紧急联系人、交互习惯等信息，
         供 Supervisor 在会话初始化时加载，并在会话结束时由 MemoryWriter 更新。

设计说明：
    - 开发阶段使用 SQLite（零依赖），生产阶段可切换 Redis（API 一致）
    - 用户画像以 JSON 序列化存储，字段灵活可扩展
    - 支持增量更新：仅修改变更的字段，不覆盖整体画像
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "user_profile")

# ================= 默认配置 =================
DEFAULT_DB_PATH: Path = config.DATA_DIR / "agent" / "user_profiles.db"


# ────────────────────────────────────────────────────────────
# 默认用户画像模板
# ────────────────────────────────────────────────────────────

DEFAULT_PROFILE: dict[str, Any] = {
    "user_id": "",
    "chronic_diseases": [],
    "allergies": [],
    "current_medications": [],
    "emergency_contacts": [],
    "preferred_dialect": "",
    "interaction_patterns": {
        "active_hours": "6:00-21:00",
        "avg_session_turns": 0,
    },
    "created_at": "",
    "updated_at": "",
}


# ────────────────────────────────────────────────────────────
# 用户画像管理器
# ────────────────────────────────────────────────────────────


class UserProfileManager:
    """
    长期用户画像管理器（SQLite 后端）。

    提供用户画像的 CRUD 操作，数据以 JSON 格式序列化存储在 SQLite 中。
    支持增量更新，仅修改变更的字段。

    使用示例::

        manager = UserProfileManager()
        profile = manager.get_profile("elderly_001")
        manager.update_profile("elderly_001", {
            "chronic_diseases": ["高血压", "糖尿病"],
        })

    生命周期：
        - Supervisor 初始化时调用 ``get_profile()`` 加载到 AgentState.user_profile
        - MemoryWriter 节点调用 ``update_profile()`` 持久化本次会话新增的信息
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """
        初始化用户画像管理器。

        Args:
            db_path: SQLite 数据库文件路径。为 None 时使用默认路径。
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f"UserProfileManager 初始化完成 | db={self.db_path}")

    def _init_db(self) -> None:
        """初始化数据库表结构。"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id     TEXT PRIMARY KEY,
                    profile     TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
            """)
            conn.commit()

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """
        获取用户画像。

        如果用户不存在，自动创建默认画像并返回。

        Args:
            user_id: 用户唯一标识

        Returns:
            dict: 用户画像字典
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT profile FROM user_profiles WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()

        if row:
            profile = json.loads(row[0])
            logger.debug(f"加载用户画像 | user_id={user_id}")
            return profile

        # 用户不存在，创建默认画像
        logger.info(f"用户不存在，创建默认画像 | user_id={user_id}")
        return self._create_default_profile(user_id)

    def update_profile(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """
        增量更新用户画像。

        仅修改 ``updates`` 中指定的字段，其余字段保持不变。
        对于列表类型字段（如 chronic_diseases），执行合并去重。

        Args:
            user_id: 用户唯一标识
            updates: 需要更新的字段字典

        Returns:
            dict: 更新后的完整画像
        """
        profile = self.get_profile(user_id)
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        # 增量合并逻辑
        for key, value in updates.items():
            if key in ("user_id", "created_at", "updated_at"):
                continue  # 保护字段不允许外部修改

            if isinstance(value, list) and isinstance(profile.get(key), list):
                # 列表字段：合并去重
                existing = profile[key]
                for item in value:
                    if item not in existing:
                        existing.append(item)
                profile[key] = existing
            elif isinstance(value, dict) and isinstance(profile.get(key), dict):
                # 字典字段：深度合并
                profile[key].update(value)
            else:
                # 其余字段：直接覆盖
                profile[key] = value

        profile["updated_at"] = now

        # 持久化
        self._save_profile(user_id, profile)
        logger.info(f"用户画像已更新 | user_id={user_id} | 更新字段={list(updates.keys())}")
        return profile

    def delete_profile(self, user_id: str) -> bool:
        """
        删除用户画像。

        Args:
            user_id: 用户唯一标识

        Returns:
            bool: 是否删除成功
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM user_profiles WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"用户画像已删除 | user_id={user_id}")
        else:
            logger.warning(f"用户画像不存在，删除无效 | user_id={user_id}")
        return deleted

    def _create_default_profile(self, user_id: str) -> dict[str, Any]:
        """创建并持久化默认用户画像。"""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        profile = {**DEFAULT_PROFILE, "user_id": user_id, "created_at": now, "updated_at": now}
        self._save_profile(user_id, profile)
        return profile

    def _save_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        """将画像序列化后写入 SQLite。"""
        profile_json = json.dumps(profile, ensure_ascii=False)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, profile, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id)
                DO UPDATE SET profile = excluded.profile, updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    profile_json,
                    profile.get("created_at", ""),
                    profile.get("updated_at", ""),
                ),
            )
            conn.commit()
