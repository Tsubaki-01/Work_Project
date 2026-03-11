"""
模块名称：pipeline
功能描述：提供文档处理流水线接口及基于 MinerU 和 MarkdownCleaner 的实现。
"""

from pathlib import Path

from silver_pilot.config import config
from silver_pilot.tools.document.cleaner import CleanOptions, MarkdownCleaner
from silver_pilot.tools.document.converter import MinerUConverter
from silver_pilot.tools.document.core import BaseDocumentPipeline
from silver_pilot.utils import get_channel_logger

logger = get_channel_logger(
    log_dir=config.LOG_DIR / "document_pipeline", channel_name="DocumentPipeline"
)


class MarkdownConverter(BaseDocumentPipeline):
    """
    标准 Markdown 处理流水线实现。

    工作流：
    1. 使用 MinerUConverter 将文件转换为原始 Markdown 格式。
    2. 使用 MarkdownCleaner 深度清洗原始的 Markdown 文件（去除页码、冗余页眉等）。
    """

    def __init__(self, mineru_token: str | None = None, clean_options: CleanOptions | None = None):
        """
        初始化标准流水线。

        Args:
            mineru_token: MinerU API Token。如果为 None，则默认从配置中读取。
            clean_options: Markdown 清洗选项。如果为 None，则使用默认的配置（全面清洗）。
        """
        self.converter = MinerUConverter(token=mineru_token)
        self.cleaner = MarkdownCleaner(options=clean_options)

    def process(
        self,
        file_paths: list[str | Path],
        output_dir: str | Path,
        *,
        is_ocr: bool = True,
        model_version: str = "vlm",
        poll_interval: int = 20,
        timeout: int = 600,
        temp_dir_name: str = "raw_md",
    ) -> list[Path]:
        """
        执行完整文档处理流水线：格式转化 → Markdown清洗。

        Args:
            file_paths: 待处理源文件路径列表
            output_dir: 最终清洗后的 Markdown 文件输出目录
            is_ocr: MinerU 是否启用OCR，默认 True
            model_version: MinerU 模型版本 (如果是HTML则传入 "MinerU-HTML")
            poll_interval: MinerU 轮询间隔(秒)
            timeout: MinerU 最大等待超时时间(秒)
            temp_dir_name: 存放未清洗的原始 MD 文件的中间临时目录名字，默认 "raw_md"

        Returns:
            list[Path]: 最终经过深入清洗后的 Markdown 文件路径列表
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 建立存放初步转换结果的中间目录
        raw_output_dir = out_dir / temp_dir_name
        raw_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"流水线启动: 开始将 {len(file_paths)} 个文件转换并清洗到 {out_dir}")

        # 1. 使用 MinerU 转换格式
        try:
            raw_md_paths = self.converter.process(
                file_paths=file_paths,
                output_dir=raw_output_dir,
                is_ocr=is_ocr,
                model_version=model_version,
                poll_interval=poll_interval,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"MinerU 转换阶段遇到错误: {e}")
            raise

        logger.info(f"原始转化完成, 共生成 {len(raw_md_paths)} 个原生 md 文件, 开始进入清洗阶段...")

        # 2. 使用 MarkdownCleaner 进行格式清洗
        try:
            cleaned_paths = self.cleaner.clean(raw_md_paths, out_dir)
        except Exception as e:
            logger.error(f"MarkdownCleaner 清洗阶段遇到错误: {e}")
            raise

        logger.info(f"流水线执行完毕, 最终产出 {len(cleaned_paths)} 个清洗后的 md 文件")
        return cleaned_paths
