"""
模块名称：unified_chunker
功能描述：提供对外统一的 Chunker 接口。
         支持识别传入的单一文件或文件夹（遍历其中所有文件），
         并根据文件类型（Excel 或 Markdown/PDF/DOC/DOCX/HTML）
         自动调用对应的切片器（ExcelChunker 或 MarkdownChunker），
         最后将切片结果以 JSON 格式保存在指定的 output_dir 中（文件名与传入文件名相同）。
使用说明::
        chunker = UnifiedChunker()
        chunker.process(source_path, output_dir, **kwargs)
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from silver_pilot.config import config
from silver_pilot.tools.document import ExcelParser, MarkdownCleaner, MarkdownConverter
from silver_pilot.utils import get_channel_logger

from .chunker_base import DocumentChunk
from .excel_chunker import ExcelChunker
from .markdown_chunker import MarkdownChunker

# ================= 日志初始化 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "unified_chunker"
logger = get_channel_logger(LOG_FILE_DIR, "unified_chunker")
# =============================================


class UnifiedChunker:
    """
    提供统一的文档切片接口，封装了对不同格式文档的处理逻辑。
    """

    def __init__(self) -> None:
        pass

    def process(self, source_path: str | Path, output_dir: str | Path, **kwargs: Any) -> None:
        """
        处理给定的单个文件或文件夹，将切片结果保存至目标文件夹。
        支持的文件类型：Markdown/PDF/DOC/DOCX/HTML/Excel 文件。

        :param source_path: 要处理的文件或目录路径
        :param output_dir:  结果保存的输出目录
        :param kwargs:  其他参数，根据文件类型传递给具体的 Chunker
        """
        source_path = Path(source_path)
        output_dir = Path(output_dir)

        if not source_path.exists():
            logger.error(f"输入的源路径不存在: {source_path}")
            raise FileNotFoundError(f"输入的源路径不存在: {source_path}")

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"统一切片开始, 目标输出目录: {output_dir}")

        if source_path.is_dir():
            logger.info(f"检查到输入 {source_path} 是一个目录，开始遍历...")
            self._process_directory(source_path, output_dir, **kwargs)
        else:
            logger.info(f"检查到输入 {source_path} 是一个文件，开始处理...")
            self.process_file(source_path, output_dir, **kwargs)

    def _process_directory(self, source_dir: Path, output_dir: Path, **kwargs: Any) -> None:
        """遍历目录并处理支持的文件"""
        count = 0
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                # 过滤常见隐藏文件和不支持的文件格式（提前在这里拦一道可避免不必要的报错日志）
                if file_path.suffix.lower() in [
                    ".md",
                    ".pdf",
                    ".doc",
                    ".docx",
                    ".html",
                    ".xlsx",
                    ".xls",
                ]:
                    self.process_file(file_path, output_dir, **kwargs)
                    count += 1
        logger.info(f"目录{source_dir}处理完成，共处理了 {count} 个文件。")

    def process_file(self, file_path: str | Path, output_dir: str | Path, **kwargs: Any) -> None:
        """
        处理单一文件：识别类型并调用对应的 Chunker，随后存为 JSON。

        :param file_path: 单个文件的路径
        :param output_dir:  结果保存的输出目录
        """
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        ext = file_path.suffix.lower()
        chunks: list[DocumentChunk] = []

        try:
            if ext in [".xlsx", ".xls"]:
                chunks = self._process_excel(file_path, **kwargs)
            elif ext in [".pdf", ".doc", ".docx", ".html", ".md"]:
                chunks = self._process_markdown(file_path, **kwargs)
            else:
                logger.warning(f"跳过不支持的文件格式: {file_path}")
                return

            # 如果成功获取 chunks，则保存为 json
            if chunks:
                self._save_chunks_to_json(chunks, file_path, output_dir)
            else:
                logger.warning(f"该文件未产出任何 chunk: {file_path}")

        except Exception as e:
            logger.exception(f"处理文件 {file_path} 时发生错误: {e}")

    def _process_excel(self, file_path: Path, **kwargs: Any) -> list[DocumentChunk]:
        """调用 ExcelParser 与 ExcelChunker 进行切片"""
        logger.debug(f"正在以 Excel 模式处理: {file_path}")
        parser = ExcelParser()
        # 这里假设未配置分组时，按默认策略全部分入一个组（ExcelChunker 支持）
        chunker = ExcelChunker(**kwargs)

        parsed_rows = parser.parse(file_path)
        chunks: list[DocumentChunk] = []
        for row in parsed_rows:
            chunks.extend(chunker.build(row))

        return chunks

    def _process_markdown(self, file_path: Path, **kwargs: Any) -> list[DocumentChunk]:
        """调用 MarkdownChunker 进行切片，支持 HTML、PDF、DOC/DOCX 文件"""
        logger.debug(f"正在以 Markdown 模式处理: {file_path}")

        converter = MarkdownConverter(**kwargs)

        # 创建一个自动销毁的虚拟临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            if file_path.suffix.lower() in [".pdf", ".doc", ".docx", ".html"]:
                # 执行格式转化与清洗，将临时目录路径传给 output_dir
                markdown_file_path: Path = converter.process(
                    file_paths=[file_path],
                    output_dir=temp_dir,
                    model_version="MinerU-HTML" if file_path.suffix.lower() == ".html" else "vlm",
                )[0]

            elif file_path.suffix.lower() == ".md":
                md_cleaner = MarkdownCleaner(**kwargs)
                markdown_file_path = md_cleaner.clean_file(file_path, temp_dir)
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return []

            chunker = MarkdownChunker(**kwargs)
            chunks = chunker.build_from_file(markdown_file_path, source_file=str(file_path))
            return chunks

    def _save_chunks_to_json(
        self, chunks: list[DocumentChunk], original_file: Path, output_dir: Path
    ) -> None:
        """将 DocumentChunk 列表保存为 JSON，文件名与原文件名相同"""
        json_file_path = output_dir / f"{original_file.stem}.json"

        # 将 DocumentChunk 转为字典
        try:
            chunks_data = [
                {
                    "content": c.content,
                    "group_name": c.group_name,
                    "metadata": c.metadata,
                    "source_file": c.source_file,
                    "sub_index": c.sub_index,
                }
                for c in chunks
            ]

            with json_file_path.open("w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)

            logger.info(f"保存成功: {json_file_path} (共 {len(chunks)} 个块)")
        except Exception as e:
            logger.exception(f"保存 JSON 失败 {json_file_path}: {e}")
            raise
