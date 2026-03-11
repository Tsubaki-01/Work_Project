"""
模块名称：chunker_base
功能描述：RAG 切片模块的共享数据结构与通用文本分段工具。
         提供 DocumentChunk（文档 chunk 数据类）和 TextSplitter（通用文本分段器），
         供 ExcelChunker、MarkdownChunker 等上层切片器复用。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 常量定义 =================
MAX_CHUNK_SIZE: int = 800
MIN_CHUNK_SIZE: int = 100
OVERLAP_LENGTH: int = 100

# 中文标点断点优先级（从高到低）
SPLIT_DELIMITERS: list[str] = ["\n\n", "\n", "。", "！", "？", "；", "，"]

# ================= 日志初始化 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "chunker_base"
logger = get_channel_logger(LOG_FILE_DIR, "chunker_base")
# =============================================


# ---------- 共享数据结构 ----------


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


# ---------- 通用文本分段器 ----------


class TextSplitter:
    """
    通用文本分段工具：当文本超过最大长度时按自然断点分段。

    支持中文标点断点优先级：换行符 > 。 > ！ > ？ > ； > ，> 强制截断。
    支持 LaTeX 公式保护：避免在 ``$...$`` 或 ``$$...$$`` 内部截断。
    """

    def __init__(
        self,
        *,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        overlap_size: int = OVERLAP_LENGTH,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        # LaTeX 公式区间正则（用于保护公式完整性）
        self._formula_pattern = re.compile(r"\$\$.*?\$\$|\$[^$\n]+?\$", re.DOTALL)

    def split_if_needed(self, text: str) -> list[str]:
        """
        如果文本超过 max_chunk_size，按自然断点分段。

        :return: 分段后的文本列表
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        logger.debug(
            f"文本超过 max_chunk_size ({self.max_chunk_size})，执行二次分段，当前长度: {len(text)}"
        )

        # 获取公式保护区间
        protected_ranges = self._get_formula_ranges(text)

        segments: list[str] = []
        start = 0
        last_split_pos = 0

        while start < len(text):
            end = start + self.max_chunk_size

            if end >= len(text):
                segments.append(text[start:])
                break

            # 确定寻找断点的搜索下限 (min_pos):
            # 1. 优先在切片后半段寻找(避免切出碎片化短 chunk)。
            # 2. 严格要求大于最后一次切分点(last_split_pos)，这防止了因前后 chunk 极端重叠
            #    导致循环选中同一个标点、产生“每次只前进 1 字符”和死循环的问题。
            min_pos = max(start + self.max_chunk_size // 2, last_split_pos + 1)

            if min_pos >= end:
                split_pos = end
            else:
                # 在 [min_pos, end] 范围内寻找最近的断点
                split_pos = self._find_split_point(text, min_pos, end, protected_ranges)

            segments.append(text[start:split_pos])

            # 记录这次的截断点
            last_split_pos = split_pos

            # 带重叠地向前推进：回退 overlap_size。
            # 为了极端健壮性，绝不允许 start 倒退或在原地打转，保证至少前进一定距离。
            start = max(split_pos - self.overlap_size, start + 1)

        logger.debug(f"二次分段完成，共生成 {len(segments)} 段")
        return segments

    def _get_formula_ranges(self, text: str) -> list[tuple[int, int]]:
        """获取所有公式的字符区间，避免在公式内截断。"""
        return [(m.start(), m.end()) for m in self._formula_pattern.finditer(text)]

    def _find_split_point(
        self,
        text: str,
        min_pos: int,
        end: int,
        protected_ranges: list[tuple[int, int]],
    ) -> int:
        """
        在寻找区间 [min_pos, end] 内部寻找最优断点。

        优先级：换行符 > 。 > ！ > ？ > ； > ， > 强制截断。
        同时避免在 LaTeX 公式内部截断。
        """
        for delim in SPLIT_DELIMITERS:
            pos = text.rfind(delim, min_pos, end)
            if pos != -1:
                candidate = pos + len(delim)
                # 检查是否在公式保护区间内
                if not self._is_in_protected_range(candidate, protected_ranges):
                    return candidate
        # 找不到自然断点则强制截断
        return end

    @staticmethod
    def _is_in_protected_range(pos: int, protected_ranges: list[tuple[int, int]]) -> bool:
        """检查位置是否落在公式保护区间内。"""
        return any(s < pos < e for s, e in protected_ranges)
