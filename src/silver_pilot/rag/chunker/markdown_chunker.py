"""
模块名称：markdown_chunker
功能描述：Markdown 文档切片器（混合切片策略）。
         将 MinerU 产出并经 MarkdownCleaner 清洗后的 Markdown 文档，按标题层级切分为
         语义完整的 DocumentChunk，用于后续 Embedding 与 Milvus 入库。

核心策略:
    1. **按标题切分** — 按 H1 / H2 / H3 标题将文档分割为结构化节点
    2. **短 chunk 合并** — 内容过短的节向前合并，避免碎片
    3. **表格独立 chunk** — 线性化后的表格数据作为独立 chunk
    4. **递归字符切分** — 超长节按中文标点断点二次切分（带 overlap）
    5. **上下文前缀** — 每个 chunk 注入文档标题和章节路径

使用说明::

    chunker = MarkdownChunker()
    chunks = chunker.build_from_file("诊疗方案.md")
    # chunks: list[DocumentChunk]
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

from .chunker_base import (
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    OVERLAP_LENGTH,
    DocumentChunk,
    TextSplitter,
)

# ================= 日志初始化 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "markdown_chunker"
logger = get_channel_logger(LOG_FILE_DIR, "markdown_chunker")
# =============================================


# ---------- 正则模式 ----------

# 匹配 Markdown 标题行（H1-H6）
_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)

# 匹配线性化表格行：「字段: 值」格式（支持中英文冒号）
# 一行内包含多个 "字段: 值，" 或 "字段: 值。" 组合的模式
_TABLE_ROW_PATTERN = re.compile(
    r"^[^\n]{0,20}[:：]\s*.+?[,，;；。]",
)

# YAML frontmatter
_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------- 数据结构 ----------


@dataclass
class MarkdownSection:
    """解析后的 Markdown 节点。"""

    level: int  # 标题级别 (1-6)，0 表示文档顶层
    title: str  # 标题文本
    content: str  # 标题下的正文内容（不含子节标题后的内容）
    path: list[str] = field(default_factory=list)  # 层级路径
    is_table: bool = False  # 是否为表格数据


# ---------- 核心切片器 ----------


class MarkdownChunker:
    """
    Markdown 文档切片器：混合切片策略。

    **工作流程**:

    1. 解析 YAML frontmatter 提取文档标题
    2. 按 H1/H2/H3 标题将文档分割为扁平的 section 列表
    3. 识别表格数据区域，标记为独立 section
    4. 合并过短的 section 到前一个 section
    5. 对超长 section 执行递归字符切分（带 overlap）
    6. 为每个 chunk 注入上下文前缀（文档标题 > 章节路径）

    典型用法::

        chunker = MarkdownChunker()
        chunks = chunker.build_from_file("诊疗方案.md")
    """

    def __init__(
        self,
        *,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        overlap_size: int = OVERLAP_LENGTH,
        min_chunk_size: int = MIN_CHUNK_SIZE,
        split_headers: list[int] | None = None,
        context_prefix: bool = True,
    ) -> None:
        """
        :param max_chunk_size: 单个 chunk 的最大字符数，超过则二次分段。
        :param overlap_size: 二次分段时前后 chunk 的重叠字符数。
        :param min_chunk_size: 低于此字符数的 section 将向前合并。
        :param split_headers: 要切分的标题级别列表，默认 [1, 2, 3]。
        :param context_prefix: 是否在 chunk 前注入文档标题和章节路径。
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.split_headers = split_headers or [1, 2, 3]
        self.context_prefix = context_prefix
        self._splitter = TextSplitter(
            max_chunk_size=max_chunk_size,
            overlap_size=overlap_size,
        )
        logger.debug(
            f"MarkdownChunker 初始化完成 | max_chunk={max_chunk_size}, "
            f"overlap={overlap_size}, min_chunk={min_chunk_size}, "
            f"split_headers={self.split_headers}, prefix={self.context_prefix}"
        )

    # ---------- 公开接口 ----------

    def build(self, md_text: str, source_file: str = "") -> list[DocumentChunk]:
        """
        将 Markdown 文本切片为 DocumentChunk 列表。

        :param md_text: 完整的 Markdown 文本
        :param source_file: 来源文件路径（可选，用于元数据）
        :return: DocumentChunk 列表
        """
        # 1. 提取文档标题
        doc_title, body = self._extract_title(md_text)
        logger.debug(f"提取文档标题: {doc_title or '无标题'}")

        # 2. 按标题切分为 section 列表
        sections = self._parse_sections(body)
        logger.debug(f"初始提取 section 数量: {len(sections)}")

        # 3. 在每个 section 内识别并分离表格数据
        sections = self._extract_table_sections(sections)
        table_count = sum(1 for s in sections if s.is_table)
        logger.debug(f"提取表格后 section 数量: {len(sections)} (独立表格数: {table_count})")

        # 4. 合并过短的 section
        sections = self._merge_short_sections(sections)
        logger.debug(f"合并短 section 后最终 section 数量: {len(sections)}")

        # 5. 生成 DocumentChunk
        chunks: list[DocumentChunk] = []

        for section in sections:
            if not section.content.strip():
                continue

            # 构建上下文前缀
            prefix = ""
            if self.context_prefix and (doc_title or section.path):
                path_parts = [doc_title] + section.path if doc_title else section.path
                prefix = "[" + " > ".join(path_parts) + "] "

            # 构建 chunk 文本
            full_text = prefix + section.content.strip()

            # 对表格 chunk 不做二次切分（即使超长也保持完整）
            if section.is_table:
                chunks.append(
                    DocumentChunk(
                        group_name="表格数据" if section.is_table else section.title,
                        content=full_text,
                        metadata={
                            "doc_title": doc_title,
                            "section_path": " > ".join(section.path),
                            "is_table": True,
                        },
                        source_file=source_file,
                    )
                )
                continue

            # 超长 section 二次切分
            try:
                text_segments = self._splitter.split_if_needed(full_text)

                if len(text_segments) > 1:
                    logger.debug(
                        f"超长 section 二次切分 (归属: '{section.title}') | "
                        f"文本总长度: {len(full_text)} -> 切分数: {len(text_segments)}"
                    )
            except Exception as e:
                logger.error(f"❌ 超长 section 二次切分异常 (归属: '{section.title}') | 错误: {e}")
                text_segments = [full_text]  # 降级处理，不中断流程

            for sub_idx, segment in enumerate(text_segments):
                chunks.append(
                    DocumentChunk(
                        group_name=section.title or "正文",
                        content=segment,
                        metadata={
                            "doc_title": doc_title,
                            "section_path": " > ".join(section.path),
                        },
                        source_file=source_file,
                        sub_index=sub_idx,
                    )
                )

        logger.info(
            f"📦 MD 切片完成 | 文档: {doc_title or source_file} | "
            f"总 section 数: {len(sections)}, 总 chunk 数: {len(chunks)}"
        )
        return chunks

    def build_from_file(
        self, file_path: str | Path, source_file: str | Path = ""
    ) -> list[DocumentChunk]:
        """
        从文件读取 Markdown 文本并切片。

        :param file_path: Markdown 文件路径
        :return: DocumentChunk 列表
        """
        file_path = Path(file_path)
        chunk_source_file = Path(source_file or file_path)
        logger.info(f"📄 开始读取文件并构建 Chunk: {file_path}")
        try:
            md_text = file_path.read_text(encoding="utf-8")
            return self.build(md_text, source_file=str(chunk_source_file))
        except Exception as e:
            logger.error(f"❌ 处理文件失败: {file_path} | 错误信息: {e}")
            raise

    # ---------- 内部方法 ----------

    def _extract_title(self, md_text: str) -> tuple[str, str]:
        """
        提取文档标题。

        优先从 YAML frontmatter ``title`` 字段提取；如无，则取第一个 H1 标题。
        返回 (标题, 去除 frontmatter 后的正文)。
        """
        doc_title = ""
        body = md_text

        # 尝试提取 YAML frontmatter
        fm_match = _FRONTMATTER_PATTERN.match(md_text)
        if fm_match:
            fm_content = fm_match.group(1)
            body = md_text[fm_match.end() :]
            # 简易提取 title 字段
            for line in fm_content.split("\n"):
                line = line.strip()
                if line.lower().startswith("title:"):
                    doc_title = line.split(":", 1)[1].strip().strip("'\"")
                    break

        # 如果 frontmatter 无标题，取第一个 H1 标题
        if not doc_title:
            first_h1 = re.search(r"^#\s+(.*?)\s*$", body, re.MULTILINE)
            if first_h1:
                doc_title = first_h1.group(1).strip()

        return doc_title, body

    def _parse_sections(self, body: str) -> list[MarkdownSection]:
        """
        按标题层级将文档分割为扁平的 section 列表。

        每个 section 包含一个标题和该标题下的正文内容（直到下一个同级或更高级标题）。
        """
        # 找到所有标题及其位置
        headers: list[tuple[int, int, str, int]] = []  # (pos, level, title, end_pos)
        for match in _HEADER_PATTERN.finditer(body):
            level = len(match.group(1))
            title = match.group(2).strip()
            if level in self.split_headers:
                headers.append((match.start(), level, title, match.end()))

        if not headers:
            # 无标题 → 整篇作为一个 section
            return [MarkdownSection(level=0, title="正文", content=body.strip(), path=[])]

        sections: list[MarkdownSection] = []

        # 如果文档开头有标题前的内容，作为前言 section
        if headers[0][0] > 0:
            preamble = body[: headers[0][0]].strip()
            if preamble:
                sections.append(MarkdownSection(level=0, title="前言", content=preamble, path=[]))

        # 构建层级路径栈
        path_stack: list[tuple[int, str]] = []  # (level, title)

        for i, (_pos, level, title, end_pos) in enumerate(headers):
            # 获取该标题到下一个标题之间的内容
            if i + 1 < len(headers):
                content = body[end_pos : headers[i + 1][0]]
            else:
                content = body[end_pos:]

            # 维护路径栈：弹出所有 >= 当前 level 的标题
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()

            path_stack.append((level, title))

            # 构建路径（不包含当前标题本身，仅父级路径）
            path = [t for _, t in path_stack[:-1]]

            sections.append(
                MarkdownSection(
                    level=level,
                    title=title,
                    content=content.strip(),
                    path=path + [title],
                )
            )

        return sections

    def _extract_table_sections(self, sections: list[MarkdownSection]) -> list[MarkdownSection]:
        """
        在每个 section 内识别线性化表格数据，将其分离为独立的表格 section。

        线性化表格的特征：连续多行（≥2行）都是「字段: 值」格式的文本。
        """
        result: list[MarkdownSection] = []

        for section in sections:
            if section.is_table or not section.content.strip():
                result.append(section)
                continue

            # 按段落（空行分隔）分割内容
            paragraphs = re.split(r"\n\s*\n", section.content)

            non_table_parts: list[str] = []
            current_table_lines: list[str] = []

            def _flush_table() -> None:
                """将收集到的表格行作为独立 section 存入结果。"""
                if len(current_table_lines) >= 2:
                    table_text = "\n\n".join(current_table_lines)
                    result.append(
                        MarkdownSection(
                            level=section.level,
                            title=section.title,
                            content=table_text,
                            path=section.path,
                            is_table=True,
                        )
                    )
                else:
                    # 不够 2 行，不算表格，归还到普通内容
                    non_table_parts.extend(current_table_lines)
                current_table_lines.clear()

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if self._is_table_paragraph(para):
                    # 当遇到表格段落前，先把之前积累的普通内容存起来
                    if non_table_parts and not current_table_lines:
                        pass  # 继续积累，等表格结束后一起处理
                    current_table_lines.append(para)
                else:
                    # 非表格段落：先 flush 之前的表格
                    if current_table_lines:
                        # 先把之前的普通内容作为 section
                        if non_table_parts:
                            result.append(
                                MarkdownSection(
                                    level=section.level,
                                    title=section.title,
                                    content="\n\n".join(non_table_parts),
                                    path=section.path,
                                )
                            )
                            non_table_parts = []
                        _flush_table()
                    non_table_parts.append(para)

            # 处理末尾残余
            if current_table_lines:
                if non_table_parts:
                    result.append(
                        MarkdownSection(
                            level=section.level,
                            title=section.title,
                            content="\n\n".join(non_table_parts),
                            path=section.path,
                        )
                    )
                    non_table_parts = []
                _flush_table()

            if non_table_parts:
                result.append(
                    MarkdownSection(
                        level=section.level,
                        title=section.title,
                        content="\n\n".join(non_table_parts),
                        path=section.path,
                    )
                )

        return result

    @staticmethod
    def _is_table_paragraph(para: str) -> bool:
        """
        判断一个段落是否为线性化表格行。

        特征：段落内包含多个 "字段: 值" 样式的键值对，以中英文逗号/分号分隔。
        例如::

            分类: 谷薯类，优选食物: 蒸煮烹饪..., 限量食物: 精白米面类...

        至少包含 2 个「键: 值」对才被视为表格行。
        """
        # 统计 "字段:" 或 "字段：" 模式的出现次数
        kv_count = len(re.findall(r"[\u4e00-\u9fff\w]{1,20}[:：]\s*", para))
        return kv_count >= 2

    def _merge_short_sections(self, sections: list[MarkdownSection]) -> list[MarkdownSection]:
        """
        合并过短的 section：内容字符数 < min_chunk_size 的 section 向前合并。

        - 表格 section 不参与合并
        - 合并后不超过 max_chunk_size 则继续合并
        - 如果是第一个 section 且过短，则向后合并到下一个
        """
        if not sections:
            return sections

        merged: list[MarkdownSection] = []

        for section in sections:
            content_len = len(section.content.strip())

            # 表格 section 不参与合并
            if section.is_table:
                merged.append(section)
                continue

            # 内容足够长，直接加入
            if content_len >= self.min_chunk_size:
                merged.append(section)
                continue

            # 短 section：尝试向前合并
            if merged and not merged[-1].is_table:
                prev = merged[-1]
                combined_len = len(prev.content.strip()) + content_len
                if combined_len <= self.max_chunk_size:
                    # 合并内容
                    separator = "\n\n"
                    title_prefix = f"## {section.title}\n" if section.title else ""
                    prev.content = (
                        prev.content.rstrip() + separator + title_prefix + section.content.strip()
                    )
                    continue

            # 无法向前合并（首个 section 或前一个是表格），直接加入
            merged.append(section)

        return merged
