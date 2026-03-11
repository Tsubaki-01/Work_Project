"""
模块名称：pipeline_base
功能描述：文档处理流水线的抽象接口定义。
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDocumentPipeline(ABC):
    """
    文档处理流水线抽象接口。
    用于将各种格式的文档（PDF, DOC, DOCX, HTML等）转换为目标格式（如 Markdown）。
    """

    @abstractmethod
    def process(self, file_paths: list[str | Path], output_dir: str | Path) -> list[Path]:
        """
        处理一组文档并输出到指定目录。

        Args:
            file_paths: 待处理文件路径列表
            output_dir: 结果输出目录

        Returns:
            list[Path]: 处理完的最终输出文件路径列表
        """
        pass
