"""
模块名称：config
功能描述：项目全局配置管理模块，负责加载 .env 环境变量、定义项目目录结构及数据库连接参数，
         并通过单例模式提供统一的配置访问入口。
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class Config:
    def __init__(self) -> None:
        # ============= 加载 .env 文件 =============
        # 记录加载.env文件前的环境变量
        pre_env_keys: set[str] = set(os.environ.keys())
        # 加载.env文件
        self.CURRENT_PKG_DIR: Path = Path(__file__).resolve().parent
        self.ROOT_DIR: Path = self.CURRENT_PKG_DIR.parent.parent
        load_dotenv(self.ROOT_DIR / ".env")
        # 记录加载.env文件后的环境变量
        post_env_keys: set[str] = set(os.environ.keys())
        # 找出.env文件中新增的环境变量
        new_env_keys: set[str] = post_env_keys - pre_env_keys

        # 注入所有新增的环境变量为实例属性
        for env_key, env_value in os.environ.items():
            if env_key in new_env_keys:
                setattr(self, env_key, env_value)

        # ============= qwen连接 ===============
        self.QWEN_URL: dict[str, str] = {
            "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "sg": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        }

        # ============= 定义项目目录 ===============

        self.DATA_DIR: Path = self.ROOT_DIR / "data"
        self.FILES_DIR: Path = self.ROOT_DIR / "files"
        self.WORKSPACE_DIR: Path = self.ROOT_DIR / "workspace"
        self.SCRIPTS_DIR: Path = self.ROOT_DIR / "scripts"
        self.TMP_DIR: Path = self.ROOT_DIR / "tmp"
        self.LOG_DIR: Path = self.ROOT_DIR / "logs"

        # ============================
        self.check_dirs()

    def check_dirs(self) -> None:
        """确保关键目录存在，不存在则自动创建"""
        for attr in ["DATA_DIR", "LOG_DIR"]:
            dir_path: Path = getattr(self, attr)
            dir_path.mkdir(exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        """访问未定义属性时返回环境变量"""
        if name in os.environ:
            return os.environ[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")


# 使用 lru_cache 装饰器实现单例模式
# 避免每次导入都重新读取文件，提高性能
@lru_cache
def get_configs() -> Config:
    print("加载配置文件...")
    singleton = Config()
    print("配置文件加载完成")
    return singleton


config: Config = get_configs()


if __name__ == "__main__":
    print(config.ROOT_DIR)
    print(config.DATA_DIR)
    print(config.DASHSCOPE_API_KEY)
