"""
模块名称：log
功能描述：基于 Loguru 的多通道日志工具模块，支持按业务通道（channel）隔离日志文件输出，
         自动管理控制台初始化与文件 Handler 的防重复绑定，提供日志轮转、压缩归档等功能。
"""

# mypy: ignore-errors
import os
import sys
from pathlib import Path

from loguru import logger

from ..config import config

# --- 模块级状态记录，用于防止重复添加 Handler ---
_CONSOLE_INITIALIZED = False
_CONFIGURED_CHANNELS = set()


def get_channel_logger(log_dir: str | Path = config.LOG_DIR, channel_name: str = "default"):
    """
    获取一个支持多目录隔离的 logger 实例。
    纯函数实现，自动处理控制台初始化与文件 Handler 防止重复绑定的问题。
    """
    global _CONSOLE_INITIALIZED

    log_dir = str(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. 配置全局控制台输出 (整个 Python 进程只执行一次)
    # ---------------------------------------------------------
    if not _CONSOLE_INITIALIZED:
        logger.remove()

        # 配置全局默认的 extra 字段，兜底原生 logger 的调用
        logger.configure(extra={"channel": "GLOBAL"})

        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<yellow>[{extra[channel]}]</yellow> | "
            "<level>{message}</level>"
        )
        logger.add(sys.stdout, format=console_format, level="DEBUG")
        _CONSOLE_INITIALIZED = True

    # ---------------------------------------------------------
    # 2. 配置该通道独享的文件输出 (每个 channel 仅执行一次)
    # ---------------------------------------------------------
    if channel_name not in _CONFIGURED_CHANNELS:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
        )

        def specific_filter(record):
            return record["extra"].get("channel") == channel_name

        # --- INFO 日志 ---
        logger.add(
            os.path.join(log_dir, "runtime.log"),
            format=file_format,
            level="INFO",
            filter=specific_filter,
            rotation="5 MB",
            retention="1 week",
            compression="zip",
            enqueue=True,
            encoding="utf-8",
        )

        # --- ERROR 日志 ---
        logger.add(
            os.path.join(log_dir, "error.log"),
            format=file_format,
            level="ERROR",
            filter=specific_filter,
            rotation="5 MB",
            retention="1 week",
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
        )

        # 记录该 channel 已配置，防止后续重复添加文件 handler
        _CONFIGURED_CHANNELS.add(channel_name)

    # ---------------------------------------------------------
    # 3. 返回绑定了 channel 的 logger
    # ---------------------------------------------------------
    return logger.bind(channel=channel_name)


if __name__ == "__main__":
    # --- 场景模拟：像平时写代码一样随意调用 ---

    # 第一次获取 TaskA 的 logger
    log_a = get_channel_logger(log_dir="logs/task_a", channel_name="TaskA")
    log_a.info("任务 A 第一次调用")

    # 第二次在别的地方又获取了 TaskA 的 logger (不会重复写文件)
    log_a_again = get_channel_logger(log_dir="logs/task_a", channel_name="TaskA")
    log_a_again.info("任务 A 第二次调用")

    # 获取 TaskB 的 logger
    log_b = get_channel_logger(log_dir="logs/task_b", channel_name="TaskB")
    log_b.error("我是任务 B 的错误日志")

    # 原生 logger 兜底测试
    logger.warning("这是一条没有 bind 过 channel 的原生全局日志")
