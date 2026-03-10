"""
模块名称：chunk_builder
功能描述：通用的 Excel 行数据分块器（按主题分组 Chunking）。
         将 ExcelParser 输出的 ParsedRow 按预定义或自动推断的主题分组合并为 2~3 个
         语义聚合的 Document Chunk，用于后续 Embedding 与 Milvus 入库。
使用说明: ChunkBuilder 为主要类，用于将 ParsedRow 分组合并为 DocumentChunk。包含以下方法：

- ``__init__(self, chunk_groups: Sequence[ChunkGroup], context_prefix_field: str | None = None)``: 初始化分块器，需提供主题分组配置。
- ``build(self, row: ParsedRow) -> list[DocumentChunk]``: 对单条 ParsedRow 进行分块，返回分块结果列表。
- ``build_batch(self, rows: Iterator[ParsedRow] | Sequence[ParsedRow]) -> Iterator[DocumentChunk]``: 对批量 ParsedRow 进行分块，返回分块结果迭代器。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from silver_pilot.config import config
from silver_pilot.tools.document import ParsedRow
from silver_pilot.utils import get_channel_logger

# ================= 日志初始化 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "chunk_builder"
logger = get_channel_logger(LOG_FILE_DIR, "chunk_builder")
# =============================================

# ================= 常量定义 =================
MAX_CHUNK_SIZE: int = 1800
OVERLAP_LENGTH: int = 180

# ---------- 数据结构 ----------


@dataclass
class DocumentChunk:
    """
    表示一条可用于 Embedding 入库的文档 chunk。

    Attributes:
        chunk_id:     全局唯一 chunk 标识（由调用方或入库模块生成）
        group_name:   所属主题分组名（如 "基本信息"、"临床使用"）
        content:      chunk 的完整文本（已含上下文前缀）
        metadata:     附加的元数据（来源文件、行号、原始元数据等）
        source_file:  原始文件路径
        sheet_name:   工作表名
        row_index:    原始行号
    """

    group_name: str
    content: str
    metadata: dict
    source_file: str = ""
    sheet_name: str = ""
    row_index: int = -1
    chunk_id: int | None = None
    sub_index: int = 0  # 二次分段后的段内序号

    @property
    def metadata_json(self) -> str:
        """元数据序列化为 JSON 字符串（用于写入 Milvus VARCHAR 字段）。"""
        return json.dumps(self.metadata, ensure_ascii=False)


@dataclass
class ChunkGroup:
    """
    主题分组定义：指定组名及该组包含哪些内容列。

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


class ChunkBuilder:
    """
    将 ``ParsedRow`` 按主题分组构建为 ``DocumentChunk`` 列表。

    **工作流程**:

    1. 根据 ``chunk_groups`` 将每行的内容列分配到对应的组中
    2. 同组内的多列文本合并为一段（用 ``section_separator`` 分隔）
    3. 在文本前注入上下文前缀（如 ``[紫杉醇]``）
    4. 如果合并后文本超过 ``max_chunk_length``，执行二次分段

    典型用法::

        builder = ChunkBuilder(
            chunk_groups=DRUG_INSTRUCTION_GROUPS,
            context_prefix_field="通用名称",
        )
        for parsed_row in parser.parse("药品说明书.xlsx"):
            chunks = builder.build(parsed_row)
            # chunks: list[DocumentChunk]
    """

    def __init__(
        self,
        *,
        chunk_groups: list[ChunkGroup] | None = None,
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
        self.max_chunk_length = max_chunk_length
        self.overlap_length = overlap_length

    def build(self, parsed_row: ParsedRow) -> list[DocumentChunk]:
        """
        将一行解析后的数据构建为 chunk 列表。

        :param parsed_row: ExcelParser 输出的 ParsedRow
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
                    sections.append(f"【{col}】{text}")

            if not sections:
                continue  # 该组全部为空，跳过

            # 合并文本
            merged_text = self.section_separator.join(sections)

            # 注入前缀
            if prefix:
                merged_text = f"[{prefix}] {merged_text}"

            # 二次分段（如超过长度阈值）
            text_segments = self._split_if_needed(merged_text)

            for sub_idx, segment in enumerate(text_segments):
                chunks.append(
                    DocumentChunk(
                        group_name=group.name,
                        content=segment,
                        metadata=parsed_row.metadata.copy(),
                        source_file=parsed_row.source_file,
                        sheet_name=parsed_row.sheet_name,
                        row_index=parsed_row.row_index,
                        sub_index=sub_idx,
                    )
                )

        return chunks

    def build_batch(
        self,
        parsed_rows: Iterator[ParsedRow] | Sequence[ParsedRow],
    ) -> Iterator[DocumentChunk]:
        """
        批量构建：对每行调用 ``build``，以迭代器形式产出所有 chunk。

        :param parsed_rows: ParsedRow 的迭代器或序列
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

    def _resolve_prefix(self, parsed_row: ParsedRow) -> str:
        """从元数据中提取上下文前缀文本。"""
        if self.context_prefix_field is None:
            return ""
        val = parsed_row.metadata.get(self.context_prefix_field)
        if val is None:
            # 也尝试在 contents 中查找
            val = parsed_row.contents.get(self.context_prefix_field)
        return str(val).strip() if val else ""

    def _resolve_groups(self, parsed_row: ParsedRow) -> list[ChunkGroup]:
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

    def _split_if_needed(self, text: str) -> list[str]:
        """
        如果文本超过 max_chunk_length，按自然断点分段（优先换行 > 句号 > 强制截断）。

        :return: 分段后的文本列表
        """
        if len(text) <= self.max_chunk_length:
            return [text]

        segments: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_length

            if end >= len(text):
                segments.append(text[start:])
                break

            # 在 [start, end] 范围内寻找最近的断点
            split_pos = self._find_split_point(text, start, end)
            segments.append(text[start:split_pos])

            # 带重叠地向前推进
            start = max(split_pos - self.overlap_length, start + 1)

        return segments

    @staticmethod
    def _find_split_point(text: str, start: int, end: int) -> int:
        """
        在 [start, end] 内寻找最优断点：
        优先级：换行符 > 。> ！> ？> ；> ， > 强制截断
        """
        # 从 end 向前搜索
        delimiters = ["\n", "。", "！", "？", "；", "，"]
        for delim in delimiters:
            pos = text.rfind(delim, start, end)
            if pos > start:
                return pos + len(delim)  # 断点在分隔符之后
        # 找不到自然断点则强制截断
        return end
