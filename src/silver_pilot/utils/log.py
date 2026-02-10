# mypy: ignore-errors
import os
import sys
from pathlib import Path

from loguru import logger

from silver_pilot.config import config


class LogManager:
    """
    支持多实例、多目录隔离的日志管理器
    """

    # 类变量，确保控制台只初始化一次（避免重复打印）
    _console_initialized = False

    def __init__(self, log_dir: str | Path = config.LOG_DIR, channel_name: str = "default") -> None:
        """
        初始化日志管理器
        :param log_dir: 日志存储目录
        :param channel_name: 唯一的通道名称（用于区分不同的logger实例）
        """
        self.log_dir = log_dir
        self.channel_name = channel_name

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 1. 配置全局控制台输出 (只执行一次)
        self._ensure_global_console()

        # 2. 配置该实例独享的文件输出
        self._configure_file_sinks()

    def _ensure_global_console(self) -> None:
        """配置控制台输出，确保只添加一次，防止重复"""
        if not LogManager._console_initialized:
            # 清除 loguru 默认的 handler
            logger.remove()

            # 定义控制台格式
            console_format = (
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<yellow>{extra[channel]}</yellow> | "  # 显示通道名
                "<level>{message}</level>"
            )

            # 添加控制台 handler (控制台我们希望看到所有日志，所以不加 filter，或只加 level 限制)
            logger.add(sys.stdout, format=console_format, level="DEBUG")

            LogManager._console_initialized = True

    def _configure_file_sinks(self) -> None:
        """配置具体的文件输出，使用 filter 进行隔离"""

        # 定义文件日志格式
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
        )

        # 定义过滤器：只有当日志的 extra["channel"] 等于当前实例名称时，才写入
        # record 是 loguru 传递给 filter 的日志记录对象
        def specific_filter(record):
            return record["extra"].get("channel") == self.channel_name

        # --- INFO 日志 (包含 INFO, WARNING, ERROR...) ---
        logger.add(
            os.path.join(self.log_dir, "runtime.log"),
            format=file_format,
            level="INFO",
            filter=specific_filter,  # 关键：绑定过滤器
            # type: ignore[arg-type]
            rotation="5 MB",
            retention="1 week",
            compression="zip",
            enqueue=True,
            encoding="utf-8",
        )

        # --- ERROR 日志 (仅 ERROR, CRITICAL) ---
        logger.add(
            os.path.join(self.log_dir, "error.log"),
            format=file_format,
            level="ERROR",
            filter=specific_filter,  # 关键：绑定过滤器
            # type: ignore[arg-type]
            rotation="5 MB",
            retention="1 week",
            encoding="utf-8",
            backtrace=True,  # 记录详细回溯
            diagnose=True,  # 记录变量值
        )

    def get_logger(self):  # type: ignore[no-untyped-def]
        """
        返回一个绑定了当前 channel 上下文的 logger
        """
        # 关键：使用 bind 方法，将 channel_name 注入到 extra 字典中
        return logger.bind(channel=self.channel_name)


if __name__ == "__main__":
    # --- 场景模拟 ---

    # 1. 创建 任务A 的日志管理器，存放在 logs/task_a
    manager_a = LogManager(log_dir="logs/task_a", channel_name="TaskA")
    log_a = manager_a.get_logger()

    # 2. 创建 任务B 的日志管理器，存放在 logs/task_b
    manager_b = LogManager(log_dir="logs/task_b", channel_name="TaskB")
    log_b = manager_b.get_logger()

    # --- 开始打印日志 ---

    # 这条只会出现在 logs/task_a/runtime.log
    log_a.info("我是任务 A 的普通日志")

    # 这条只会出现在 logs/task_b/runtime.log
    log_b.info("我是任务 B 的普通日志")

    # 这条只会出现在 logs/task_a/error.log 和 runtime.log
    log_a.error("任务 A 出错了！")

    # 验证：你可以去文件夹里看，A 的文件里绝对没有 B 的日志
    print("日志写入完成，请查看 logs 文件夹。")
