"""
模块名称：excel_chunker
功能描述：Excel 行数据分块器（按主题分组 Chunking）。
         将 ExcelParser 输出的 ExcelPasedRow 按预定义或自动推断的主题分组合并为
         语义聚合的 Document Chunk，用于后续 Embedding 与 Milvus 入库。

使用说明: ExcelChunker 为主要类，用于将 ExcelPasedRow 分组合并为 DocumentChunk。包含以下方法：

- ``__init__(...)``: 初始化分块器，需提供主题分组配置。
- ``build(self, row: ExcelPasedRow) -> list[DocumentChunk]``: 对单条 ExcelPasedRow 进行分块。
- ``build_batch(...) -> Iterator[DocumentChunk]``: 对批量 ExcelPasedRow 进行分块。
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from silver_pilot.config import config
from silver_pilot.tools.document import ExcelPasedRow
from silver_pilot.utils import get_channel_logger

from .chunker_base import (
    MAX_CHUNK_SIZE,
    OVERLAP_LENGTH,
    DocumentChunk,
    TextSplitter,
)

# ================= 日志初始化 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "excel_chunker"
logger = get_channel_logger(LOG_FILE_DIR, "excel_chunker")
# =============================================


# ---------- 数据结构 ----------


@dataclass
class ChunkGroup:
    """
    主题分组定义：指定组名及该组包含哪些内容列。
    分组用于将类似语义的内容列合并为一个组别。
    用于之后milvus的标量检索。

    示例::

        ChunkGroup(name="基本信息", columns=["性状", "主要成份", "适应症", "规格", "贮藏"])
    """

    name: str
    columns: list[str] = field(default_factory=list)


# ---------- 预定义分组模板 ----------


# 药品说明书专用分组
DRUG_INSTRUCTION_GROUPS: list[ChunkGroup] = [
    ChunkGroup(
        name="基本信息",
        columns=["性状", "主要成份", "相关疾病", "适应症", "规格", "贮藏", "有效期"],
    ),
    ChunkGroup(
        name="临床使用",
        columns=[
            "用法用量",
            "不良反应",
            "禁忌",
            "注意事项",
            "孕妇及哺乳期妇女用药",
            "儿童用药",
            "老人用药",
        ],
    ),
    ChunkGroup(
        name="药理信息",
        columns=["药物相互作用", "药理毒理", "药代动力学"],
    ),
]


# ---------- 核心分块器 ----------


class ExcelChunker:
    """
    将 ``ExcelPasedRow`` 按主题分组构建为 ``DocumentChunk`` 列表。

    **工作流程**:

    1. 根据 ``chunk_groups`` 将每行的内容列分配到对应的组中
    2. 同组内的多列文本合并为一段（用 ``section_separator`` 分隔）
    3. 在文本前注入上下文前缀（如 ``[紫杉醇]``）
    4. 如果合并后文本超过 ``max_chunk_length``，执行二次分段

    典型用法::

        chunker = ExcelChunker(
            chunk_groups=DRUG_INSTRUCTION_GROUPS,
            context_prefix_field="通用名称",
        )
        for parsed_row in parser.parse("药品说明书.xlsx"):
            chunks = chunker.build(parsed_row)
            # chunks: list[DocumentChunk]
    """

    def __init__(
        self,
        *,
        chunk_groups: list[ChunkGroup] | None = DRUG_INSTRUCTION_GROUPS,
        context_prefix_field: str | None = None,
        section_separator: str = "\n",
        max_chunk_length: int = MAX_CHUNK_SIZE,
        overlap_length: int = OVERLAP_LENGTH,
    ) -> None:
        """
        :param chunk_groups: 主题分组定义列表。为 None 时自动按所有内容列生成单一分组。
        :param context_prefix_field: 元数据中用作 chunk 文本前缀的字段名（如 "通用名称"）。
        :param section_separator: 同组内不同列文本之间的分隔符。
        :param max_chunk_length: 单个 chunk 的最大字符数，超过则二次分段。
        :param overlap_length: 二次分段时前后 chunk 的重叠字符数。
        """
        self.chunk_groups = chunk_groups
        self.context_prefix_field = context_prefix_field
        self.section_separator = section_separator
        self._splitter = TextSplitter(
            max_chunk_size=max_chunk_length,
            overlap_size=overlap_length,
        )

    def build(self, parsed_row: ExcelPasedRow) -> list[DocumentChunk]:
        """
        将一行解析后的数据构建为 chunk 列表。

        :param parsed_row: ExcelParser 输出的 ExcelPasedRow
        :return: 该行对应的 DocumentChunk 列表
        """
        # 确定上下文前缀
        prefix = self._resolve_prefix(parsed_row)

        # 确定分组
        groups = self._resolve_groups(parsed_row)

        chunks: list[DocumentChunk] = []

        for group in groups:
            # 合并该组中存在于本行的内容列
            sections: list[str] = []
            for col in group.columns:
                text = parsed_row.contents.get(col)
                if text:
                    sections.append(f"**{col}**{text}")

            if not sections:
                continue  # 该组全部为空，跳过

            # 合并文本
            merged_text = self.section_separator.join(sections)

            # 二次分段（如超过长度阈值）
            text_segments = self._splitter.split_if_needed(merged_text)

            for sub_idx, segment in enumerate(text_segments):
                # 确保切分出来的每一个子块都带有前缀
                final_content = f"**{prefix}** {segment}" if prefix else segment
                chunks.append(
                    DocumentChunk(
                        group_name=group.name,
                        content=final_content,
                        metadata=parsed_row.metadata.copy(),
                        source_file=parsed_row.source_file,
                        sub_index=sub_idx,
                    )
                )

        return chunks

    def build_batch(
        self,
        parsed_rows: Iterator[ExcelPasedRow] | Sequence[ExcelPasedRow],
    ) -> Iterator[DocumentChunk]:
        """
        批量构建：对每行调用 ``build``，以迭代器形式产出所有 chunk。

        :param parsed_rows: ExcelPasedRow 的迭代器或序列
        :yields: DocumentChunk
        """
        total_chunks = 0
        total_rows = 0
        for row in parsed_rows:
            row_chunks = self.build(row)
            total_rows += 1
            total_chunks += len(row_chunks)
            yield from row_chunks

        logger.info(f"📦 分块完成 | 总行数: {total_rows}, 总 chunk 数: {total_chunks}")

    # ---------- 内部方法 ----------

    def _resolve_prefix(self, parsed_row: ExcelPasedRow) -> str:
        """从元数据中提取上下文前缀文本。"""
        if self.context_prefix_field is None:
            return ""
        val = parsed_row.metadata.get(self.context_prefix_field)
        if val is None:
            # 也尝试在 contents 中查找
            val = parsed_row.contents.get(self.context_prefix_field)
        return str(val).strip() if val else ""

    def _resolve_groups(self, parsed_row: ExcelPasedRow) -> list[ChunkGroup]:
        """
        确定分组策略：
        - 如果有预定义分组，使用预定义 + 将未分配的列归入 "其他" 组
        - 如果无分组配置，则所有内容列合为一个 "全部内容" 组
        """
        content_keys = set(parsed_row.contents.keys())

        if self.chunk_groups is None:
            # 无分组定义 → 全部内容合为一个组
            return [ChunkGroup(name="全部内容", columns=list(content_keys))]

        groups: list[ChunkGroup] = []
        assigned: set[str] = set()

        for grp in self.chunk_groups:
            # 只保留在本行中实际有内容的列
            cols = [c for c in grp.columns if c in content_keys]
            if cols:
                groups.append(ChunkGroup(name=grp.name, columns=cols))
                assigned.update(cols)

        # 未被分配的列归入 "其他" 组
        remaining = content_keys - assigned
        if remaining:
            groups.append(ChunkGroup(name="其他", columns=sorted(remaining)))

        return groups
