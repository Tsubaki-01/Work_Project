import os
import sys

from loguru import logger

from silver_pilot.config import config


class LogManager:
    def __init__(self, log_dir=config.LOG_DIR) -> None:  # type: ignore[no-untyped-def]
        self.log_dir = log_dir
        self._configure_logger()

    def _configure_logger(self) -> None:
        # 1. 清除默认配置
        logger.remove()

        # 2. 定义统一的日志格式
        # <green> 等标签是 loguru 特有的色彩控制
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # 3. 添加控制台输出
        logger.add(sys.stdout, format=log_format, level="DEBUG")

        # 4. 添加常规日志文件 (INFO及以上)
        logger.add(
            os.path.join(self.log_dir, "runtime.log"),
            format=log_format,
            level="INFO",
            rotation="5 MB",  # 满5MB自动切分
            retention="1 week",  # 保留最近一周
            compression="zip",  # 旧日志压缩保存
            enqueue=True,  # 异步写入，不阻塞主线程
            encoding="utf-8",
        )

        # 5. 添加错误日志文件 (仅 ERROR 及以上)
        logger.add(
            os.path.join(self.log_dir, "error.log"),
            format=log_format,
            level="ERROR",
            rotation="1 week",
            backtrace=True,  # 记录详细回溯
            diagnose=True,  # 记录变量值
        )

    def get_logger(self):  # type: ignore[no-untyped-def]
        return logger


# 实例化
log = LogManager().get_logger()
