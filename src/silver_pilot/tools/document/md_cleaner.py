"""
模块名称：md_cleaner
功能描述：Markdown 文档深度清洗工具。支持去除目录、页码噪音、冗余页眉页脚、
         行内噪音、引用标记、参考文献、垃圾字符等；同时支持表格线性化、
         标题层级格式化和段落智能合并，支持md格式和html格式的清洗。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# 日志初始化
# ---------------------------------------------------------------------------
logger = get_channel_logger(config.LOG_DIR / "md_cleaner", channel_name="MarkdownCleaner")


# ---------------------------------------------------------------------------
# 清洗选项（可由调用方自行配置）
# ---------------------------------------------------------------------------
@dataclass
class CleanOptions:
    """控制各清洗步骤是否执行的开关集合。"""

    remove_images: bool = True
    """是否移除图片标记（替换为 [图片略]）。"""

    remove_links: bool = True
    """是否将超链接简化为纯文本。"""

    remove_citations: bool = True
    """是否移除 LaTeX / 普通引用标记。"""

    remove_toc: bool = True
    """是否移除目录区域。"""

    remove_page_noise: bool = True
    """是否移除独立的页码行。"""

    remove_redundant_headers: bool = True
    """是否移除全文重复出现的页眉/页脚行。"""

    remove_inline_noise: bool = True
    """是否清除行内页码、噪音后缀。"""

    remove_references: bool = True
    """是否移除参考文献 / 引用列表段落。"""

    linearize_tables: bool = True
    """是否把 Markdown 表格线性化为「字段: 值」形式。"""

    format_headers: bool = True
    """是否将中文章节标题自动添加 Markdown 标题级别。"""

    merge_paragraphs: bool = True
    """是否执行智能段落合并。"""

    clean_html_tags: bool = True
    """是否清洗 HTML 标签（<img>、<a>、<table> 及其他标签）。"""


# ---------------------------------------------------------------------------
# 清洗统计
# ---------------------------------------------------------------------------
@dataclass
class CleanStats:
    """记录每个清洗步骤所删除 / 处理的数量。"""

    removed_images: int = 0
    removed_links: int = 0
    removed_citations: int = 0
    removed_toc_lines: int = 0
    removed_page_noise_lines: int = 0
    removed_redundant_header_lines: int = 0
    removed_inline_noise_count: int = 0
    removed_garbage_lines: int = 0
    removed_reference_lines: int = 0
    tables_linearized: int = 0
    removed_html_tags: int = 0


# ---------------------------------------------------------------------------
# 主清洗类
# ---------------------------------------------------------------------------
class MarkdownCleaner:
    """Markdown 文档深度清洗器。

    主要功能
    --------
    - 提取 / 生成 YAML frontmatter 标题
    - 线性化表格
    - 保护代码块和公式块
    - 去除图片、链接、引用标记、目录、页码、冗余页眉页脚、参考文献
    - 智能合并段落
    - 格式化中文章节标题层级

    Parameters
    ----------
    options : CleanOptions | None
        清洗选项；为 ``None`` 时使用默认全部开启。
    custom_noise_patterns : list[str] | None
        用户自定义的噪音正则列表，将在行内噪音清除阶段额外匹配。
    """

    def __init__(
        self,
        options: CleanOptions | None = None,
        custom_noise_patterns: list[str] | None = None,
    ) -> None:
        self.options: CleanOptions = options or CleanOptions()
        self.stats: CleanStats = CleanStats()

        # ---- 段落结尾标点集 ----
        self.end_punctuations: set[str] = set("。！？；：\u201c\u2018\u2019）)】」』…!?;:\"'")

        # ---- 标题 / 列表行识别模式 ----
        # [Bug 2 修复] 将更精确的 `\d+\.\d+` 放在 `\d+\.` 之前
        # [Bug 1 修复] `\+\+` 改为 `\*\+`
        self.title_patterns: list[re.Pattern[str]] = [
            re.compile(p)
            for p in [
                r"^第[一二三四五六七八九十百]+(章|节|篇|条|问)\b",
                r"^分篇[一二三四五六七八九十百]+\b",
                r"^[一二三四五六七八九十]+、",
                r"^[（(][一二三四五六七八九十]+[)）]",
                r"^\d+\.\d+",  # 更精确的多级编号在前
                r"^\d+\.",  # 单级编号在后
                r"^[\-\*\+]\s+",  # [Bug 1] 修复：`\+` 重复 → `\*`
                r"^#+\s+",
                r"^\d{3}\s+\S+",
            ]
        ]

        # ---- 独立页码行模式 ----
        self.page_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"^\d{1,6}$",
                r"^/\s*\d{1,6}$",
                r"^\d{1,6}\s*/$",
                r"^[—\-–\uff0d]\s*\d{1,6}\s*[—\-–\uff0d]$",
                r"^Page\s+\d+$",
                r"^第\s*\d+\s*页$",
                r"^/\d+$",
            ]
        ]

        # ---- 标题层级识别（用于 format_headers，处理无 # 前缀的纯文本行）----
        # h2: 章 / 篇 / 部 / 部分 / 编 / 中文序号「一、」
        self.h2_pattern: re.Pattern[str] = re.compile(
            r"^("
            r"第[一二三四五六七八九十百零○〇]+[章篇部编]"  # 第一章、第二篇、第三部
            r"|第\d+[章篇部编]"  # 第1章、第2篇
            r"|第[一二三四五六七八九十百零○〇]+部分"  # 第一部分
            r"|第\d+部分"  # 第1部分
            r"|[一二三四五六七八九十]+、"  # 一、 二、
            r")"
        )
        # h3: 节 / 条 / 款 / 分篇 / 带括号的序号
        self.h3_pattern: re.Pattern[str] = re.compile(
            r"^("
            r"第[一二三四五六七八九十百零○〇]+[节条款项]"  # 第一节、第二条、第三款
            r"|第\d+[节条款项]"  # 第1节、第2条
            r"|分篇[一二三四五六七八九十百]+"  # 分篇一
            r"|[（(][一二三四五六七八九十]+[)）]"  # （一）(一)
            r"|[（(]\d+[)）]"  # （1）(1)
            r")"
        )
        # h4: 目 / 多级编号 / 3位数代码 / 带圈序号 / 字母序号
        self.h4_pattern: re.Pattern[str] = re.compile(
            r"^("
            r"第[一二三四五六七八九十百零○〇]+目"  # 第一目
            r"|第\d+目"  # 第1目
            r"|\d{3}\s+\S+"  # 101 标题
            r"|\d+\.\d+\.\d+\s+\S+"  # 1.1.1 标题
            r"|\d+\.\d+\s+\S+"  # 1.1 标题
            r"|[①②③④⑤⑥⑦⑧⑨⑩]"  # 带圈数字
            r"|[a-zA-Z][)）]\s+\S+"  # a) 标题
            r")"
        )

        # ---- 列表 / 表格 ----
        self.list_pattern: re.Pattern[str] = re.compile(r"^(\d+\.|[\-\*\+])\s+")
        self.table_pattern: re.Pattern[str] = re.compile(
            r"((?:^|\n)[ \t]*\|.*\|[ \t]*\n[ \t]*\|(?:[-:| ]+)\|[ \t]*\n"
            r"(?:[ \t]*\|.*\|[ \t]*(?:\n|$))*)"
        )

        # ---- 图片 / 链接（Markdown 格式） ----
        self.img_pattern: re.Pattern[str] = re.compile(r"!\[.*?\]\([^\)]+\)")
        self.link_pattern: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\([^\)]+\)")

        # ---- 图片 / 链接 / 表格（HTML 格式） ----
        self.html_img_pattern: re.Pattern[str] = re.compile(r"<img\b[^>]*?/?>", re.IGNORECASE)
        self.html_link_pattern: re.Pattern[str] = re.compile(
            r"<a\b[^>]*?>(.*?)</a>", re.IGNORECASE | re.DOTALL
        )
        self.html_table_pattern: re.Pattern[str] = re.compile(
            r"<table\b[^>]*?>.*?</table>", re.IGNORECASE | re.DOTALL
        )
        # 通用 HTML 标签（清除残留的 <div>、<span>、<br> 等）
        self.html_general_tag_pattern: re.Pattern[str] = re.compile(
            r"</?(?:div|span|p|br|hr|section|article|header|footer|nav|"
            r"ul|ol|li|dl|dt|dd|blockquote|pre|code|em|strong|b|i|u|s|"
            r"sub|sup|mark|small|del|ins|figure|figcaption|caption|"
            r"thead|tbody|tfoot|col|colgroup|h[1-6])\b[^>]*?/?>",
            re.IGNORECASE,
        )

        # ---- 引用标记 ----
        self.latex_cite_pattern: re.Pattern[str] = re.compile(r"\$\s*\^\{\[[\d,\s\-–]+\]\}\s*\$")
        self.normal_cite_pattern: re.Pattern[str] = re.compile(r"\[[\d,\s\-–]+\]")

        # ---- 目录相关 ----
        self.toc_guide_pattern: re.Pattern[str] = re.compile(r"(?:[.]{3,}|[…]{2,}|[-]{3,})\s*\d+$")
        self.toc_title_pattern: re.Pattern[str] = re.compile(
            r"^(第[一二三四五六七八九十百]+[章节篇].*?|[\d\.]+.*?)\s+\d+$"
        )

        # ---- 参考文献区域识别 ----
        self.reference_header_pattern: re.Pattern[str] = re.compile(
            r"^(#{0,6}\s*)?(参考文献|参\s*考\s*文\s*献|References|Bibliography|"
            r"Works\s+Cited|引用文献|文献索引)\s*$",
            re.IGNORECASE,
        )
        self.reference_item_pattern: re.Pattern[str] = re.compile(
            r"^\s*\[?\d+\]?\s*[A-Za-z\u4e00-\u9fa5]"
        )

        # ---- 保护块存储 & 文档标题 ----
        self.protected_blocks: dict[str, str] = {}
        self.doc_title: str = "未命名文档"

        # ---- 自定义噪音模式 ----
        self.custom_noise_patterns: list[re.Pattern[str]] = [
            re.compile(p) for p in (custom_noise_patterns or [])
        ]

    # ===================================================================
    # 工具方法
    # ===================================================================

    def is_paragraph_end(self, text: str) -> bool:
        """判断文本是否以段落结尾标点结束。

        [Bug 6 修复] 先 strip() 去除行尾空格，避免误判。
        """
        text = text.strip()
        if not text:
            return False
        return text[-1] in self.end_punctuations

    def is_title_or_list(self, text: str) -> bool:
        """判断一行文本是否为标题或列表项。"""
        text = text.strip()
        for pattern in self.title_patterns:
            if pattern.match(text):
                return True
        return False

    # ===================================================================
    # 文本预处理
    # ===================================================================

    def normalize_text(self, text: str) -> str:
        """统一换行符并清除不可见控制字符。"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\f", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text

    def extract_frontmatter(self, text: str) -> str:
        """提取首个 ``# 标题`` 作为文档标题。"""
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i >= 10:
                break
            match = re.match(r"^#\s+(.+)", line)
            if match:
                self.doc_title = match.group(1).strip()
                # lines.pop(i)
                logger.debug("提取文档标题: {}", self.doc_title)
                return "\n".join(lines)

        self.doc_title = "未命名文档"
        return text

    # ===================================================================
    # 表格处理
    # ===================================================================

    def linearize_tables(self, text: str) -> str:
        """将 Markdown 表格转换为「字段: 值」的线性化文本。"""

        def process_table(match: re.Match[str]) -> str:
            table_text: str = match.group(0).strip()
            lines: list[str] = table_text.split("\n")
            if len(lines) < 3:
                return match.group(0)

            headers: list[str] = [col.strip() for col in lines[0].strip("|").split("|")]
            linearized_rows: list[str] = []

            for line in lines[2:]:
                cols: list[str] = [col.strip() for col in line.strip("|").split("|")]
                row_parts: list[str] = []
                for idx in range(min(len(headers), len(cols))):
                    if headers[idx] and cols[idx]:
                        row_parts.append(f"{headers[idx]}: {cols[idx]}")
                if row_parts:
                    linearized_rows.append("，".join(row_parts) + "。")

            self.stats.tables_linearized += 1
            return "\n\n" + "\n".join(linearized_rows) + "\n\n"

        return self.table_pattern.sub(process_table, text)

    # ===================================================================
    # 保护 / 恢复代码块 & 公式块
    # ===================================================================

    def protect_blocks(self, text: str) -> str:
        """用占位符替换代码块和公式块，防止后续处理破坏其内容。

        [Bug 8 说明] 使用 ````(3+) 反引号匹配，可正确处理嵌套代码块。
        """
        self.protected_blocks.clear()

        def _make_placeholder(match: re.Match[str]) -> str:
            key: str = f"__PROTECTED_BLOCK_{len(self.protected_blocks)}__"
            self.protected_blocks[key] = match.group(0)
            return f"\n{key}\n"

        # 匹配 3 个或更多反引号的代码块（支持嵌套）
        text = re.sub(r"(`{3,}).*?\1", _make_placeholder, text, flags=re.DOTALL)
        # 匹配 $$ 公式块
        text = re.sub(r"\$\$.*?\$\$", _make_placeholder, text, flags=re.DOTALL)
        return text

    def restore_blocks(self, text: str) -> str:
        """将保护占位符还原为原始内容。"""
        for key, val in self.protected_blocks.items():
            text = text.replace(key, val)
        return text

    # ===================================================================
    # HTML 内容清洗
    # ===================================================================

    def clean_html_content(self, text: str) -> str:
        """清洗 HTML 格式的图片、链接、表格，并移除残留 HTML 标签。

        处理内容：
        - ``<img>`` 标签 → ``[图片略]``
        - ``<a>`` 标签 → 保留链接文本
        - ``<table>`` 标签 → 线性化为「字段: 值」文本
        - 其他常见 HTML 标签 → 移除标签保留文本内容
        - HTML 注释 ``<!-- ... -->`` → 移除
        - HTML 实体 ``&amp;`` ``&lt;`` 等 → 转为字面字符
        """
        total_removed: int = 0

        # 1. 移除 HTML 注释
        text, c = re.subn(r"<!--.*?-->", "", text, flags=re.DOTALL)
        total_removed += c

        # 2. <img> → [图片略]
        if self.options.remove_images:
            text, c = self.html_img_pattern.subn("[图片略]", text)
            self.stats.removed_images += c
            total_removed += c

        # 3. <a href="...">文本</a> → 文本
        if self.options.remove_links:
            text, c = self.html_link_pattern.subn(r"\1", text)
            self.stats.removed_links += c
            total_removed += c

        # 4. <table> → 线性化
        if self.options.linearize_tables:
            text, c = self.html_table_pattern.subn(self._linearize_html_table, text)
            total_removed += c

        # 5. 移除残留的通用 HTML 标签
        text, c = self.html_general_tag_pattern.subn("", text)
        total_removed += c

        # 6. 转换常见 HTML 实体
        html_entities: dict[str, str] = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&apos;": "'",
            "&nbsp;": " ",
            "&#39;": "'",
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)

        # 7. 处理数值型 HTML 实体 &#NNN; / &#xHHH;
        text = re.sub(
            r"&#(\d+);",
            lambda m: chr(int(m.group(1))),
            text,
        )
        text = re.sub(
            r"&#x([0-9a-fA-F]+);",
            lambda m: chr(int(m.group(1), 16)),
            text,
        )

        self.stats.removed_html_tags += total_removed
        if total_removed:
            logger.debug("清洗 HTML 标签/元素: {} 个", total_removed)
        return text

    def _linearize_html_table(self, match: re.Match[str]) -> str:
        """将单个 HTML ``<table>`` 标签转换为线性化文本。

        支持 ``rowspan`` / ``colspan`` 合并单元格的完整展开。
        算法：
        1. 解析每个 ``<tr>`` 中的 ``<th>``/``<td>``，提取文本、rowspan、colspan
        2. 构建二维网格，自动填充合并区域
        3. 识别表头行（全部为 ``<th>`` 或第一行），生成复合表头
        4. 将数据行线性化为「表头: 值」格式
        """
        table_html: str = match.group(0)

        # ---- 1. 提取所有 <tr> ----
        raw_rows: list[str] = re.findall(
            r"<tr\b[^>]*?>(.*?)</tr>", table_html, re.IGNORECASE | re.DOTALL
        )
        if not raw_rows:
            return ""

        # ---- 2. 解析每个单元格的文本、rowspan、colspan ----
        # cell_info: (text, rowspan, colspan, is_th)
        _CellInfo = tuple[str, int, int, bool]  # noqa: N806

        def _parse_row_cells(row_html: str) -> list[_CellInfo]:
            """解析一行中的所有 <th>/<td> 单元格。"""
            cells: list[_CellInfo] = []
            for cell_match in re.finditer(
                r"<(th|td)\b([^>]*?)>(.*?)</(?:th|td)>",
                row_html,
                re.IGNORECASE | re.DOTALL,
            ):
                tag_name: str = cell_match.group(1).lower()
                attrs: str = cell_match.group(2)
                content: str = cell_match.group(3)

                # 提取 rowspan / colspan
                rs_m = re.search(r'rowspan\s*=\s*["\']?(\d+)', attrs, re.IGNORECASE)
                cs_m = re.search(r'colspan\s*=\s*["\']?(\d+)', attrs, re.IGNORECASE)
                rowspan: int = int(rs_m.group(1)) if rs_m else 1
                colspan: int = int(cs_m.group(1)) if cs_m else 1

                # 清除单元格内残留的 HTML 标签
                text: str = re.sub(r"<[^>]+>", "", content).strip()
                cells.append((text, rowspan, colspan, tag_name == "th"))
            return cells

        parsed_rows = [_parse_row_cells(r) for r in raw_rows]

        # ---- 3. 构建二维网格（展开 rowspan / colspan）----
        total_rows: int = len(parsed_rows)
        # 估算最大列数
        max_cols: int = 0
        for cells in parsed_rows:
            col_sum: int = sum(cs for _, _, cs, _ in cells)
            if col_sum > max_cols:
                max_cols = col_sum
        # rowspan 可能使实际列数更大，动态扩展
        max_cols = max(max_cols, 1)

        # grid[r][c] = (text, is_th)；None 表示未填充
        _GridCell = tuple[str, bool] | None  # noqa: N806
        grid: list[list[_GridCell]] = [
            list[_GridCell]([None] * max_cols) for _ in range(total_rows)
        ]
        for row_idx, cells in enumerate(parsed_rows):
            col_cursor: int = 0
            cell_idx: int = 0
            while cell_idx < len(cells):
                # 跳过已被上方 rowspan 占据的位置
                while col_cursor < max_cols and grid[row_idx][col_cursor] is not None:
                    col_cursor += 1

                # 如果超出当前列数，需要动态扩展
                if col_cursor >= max_cols:
                    extra: int = col_cursor - max_cols + 1
                    max_cols += extra
                    for row in grid:
                        row.extend([None] * extra)

                text, rowspan, colspan, is_th = cells[cell_idx]

                # 填充 rowspan × colspan 区域
                for dr in range(rowspan):
                    target_row: int = row_idx + dr
                    if target_row >= total_rows:
                        break
                    for dc in range(colspan):
                        target_col: int = col_cursor + dc
                        # 动态扩展列
                        while target_col >= len(grid[target_row]):
                            grid[target_row].append(None)
                            if target_col >= max_cols:
                                max_cols = target_col + 1
                        # 只在第一个单元格写入文本，其余标记为空串
                        if dr == 0 and dc == 0:
                            grid[target_row][target_col] = (text, is_th)
                        else:
                            grid[target_row][target_col] = (text, is_th)

                col_cursor += colspan
                cell_idx += 1

        # ---- 4. 通过左上角单元格的 rowspan/colspan 识别表头结构 ----
        # rowspan = 横向表头行数，colspan = 纵向表头列数（行标签列数）
        top_left_cell = parsed_rows[0][0] if parsed_rows[0] else None
        if top_left_cell:
            _, tl_rowspan, tl_colspan, _ = top_left_cell
            header_row_count: int = max(1, tl_rowspan)
            header_col_count: int = max(1, tl_colspan)
        else:
            header_row_count = 1
            header_col_count = 1

        # 安全上界
        if header_row_count >= total_rows:
            header_row_count = 1
        actual_cols: int = max(len(row) for row in grid)
        if header_col_count >= actual_cols:
            header_col_count = 1

        # ---- 5. 构建复合列表头（横向表头）----
        # 对 header_col_count 右侧的每一列，将各表头行的文本用「-」拼接
        # 例如：第一行 "5年内风险(千人)" colspan=3，第二行有3个子表头
        #   → col1: "5年内风险(千人)-未合并心肾疾病，危险因素少"
        col_headers: list[str] = []
        for col_idx in range(actual_cols):
            if col_idx < header_col_count:
                # 行标签列：取左上角区域的表头文本
                parts: list[str] = []
                prev_text: str = ""
                for row_idx in range(header_row_count):
                    cell = grid[row_idx][col_idx]
                    if cell is not None:
                        cell_text = cell[0]
                        if cell_text and cell_text != prev_text:
                            parts.append(cell_text)
                            prev_text = cell_text
                col_headers.append("-".join(parts) if parts else "")
            else:
                # 数据列：将表头行的文本拼接为复合表头
                parts = []
                prev_text = ""
                for row_idx in range(header_row_count):
                    if col_idx < len(grid[row_idx]):
                        cell = grid[row_idx][col_idx]
                        if cell is not None:
                            cell_text = cell[0]
                            if cell_text and cell_text != prev_text:
                                parts.append(cell_text)
                                prev_text = cell_text
                col_headers.append("-".join(parts) if parts else "")

        # ---- 6. 线性化数据行 ----
        linearized_rows: list[str] = []
        for row_idx in range(header_row_count, total_rows):
            row_parts: list[str] = []

            # 6a. 行标签列（纵向表头）
            for col_idx in range(header_col_count):
                cell = grid[row_idx][col_idx]
                header = col_headers[col_idx]
                val = cell[0] if cell is not None else ""
                if header and val:
                    row_parts.append(f"{header}: {val}")

            # 6b. 数据列
            for col_idx in range(header_col_count, min(len(col_headers), len(grid[row_idx]))):
                cell = grid[row_idx][col_idx]
                header = col_headers[col_idx]
                val = cell[0] if cell is not None else ""
                if not header or not val:
                    continue
                row_parts.append(f"{header}: {val}")

            if row_parts:
                linearized_rows.append("，".join(row_parts) + "。")

        self.stats.tables_linearized += 1
        if linearized_rows:
            return "\n\n" + "\n".join(linearized_rows) + "\n\n"
        return ""

    # ===================================================================
    # 图片 / 链接 / 引用 清洗（Markdown 格式）
    # ===================================================================

    def clean_images_and_links(self, text: str) -> str:
        """移除图片标记并将链接简化为纯文本。"""
        if self.options.remove_images:
            text, count = self.img_pattern.subn("[图片略]", text)
            self.stats.removed_images += count
        if self.options.remove_links:
            text, count = self.link_pattern.subn(r"\1", text)
            self.stats.removed_links += count
        return text

    def remove_citations(self, text: str) -> str:
        """移除 LaTeX 上标引用 ``$^{[1]}$`` 和普通引用 ``[1, 2]``。"""
        text, c1 = self.latex_cite_pattern.subn("", text)
        text, c2 = self.normal_cite_pattern.subn("", text)
        self.stats.removed_citations += c1 + c2
        return text

    # ===================================================================
    # 行级清洗
    # ===================================================================

    def remove_toc(self, lines: list[str]) -> list[str]:
        """移除目录区域（从"目录"关键字到正文恢复为止）。"""
        cleaned: list[str] = []
        in_toc_zone: bool = False
        original_count: int = len(lines)

        for line in lines:
            s: str = line.strip()

            # 检测目录起始标记
            if s in ["目录", "Contents", "目 录", "TOC"]:
                in_toc_zone = True
                logger.debug("检测到目录起始标记: '{}'", s)
                continue

            # 匹配目录引导符行（如 "...... 15"）
            if self.toc_guide_pattern.search(s):
                continue

            # 目录区内：跳过省略号 / 引导点行
            if in_toc_zone and re.match(r"^[\.…\s]+$", s):
                continue

            # 目录区内：跳过带页码的标题行
            if in_toc_zone and self.toc_title_pattern.match(s):
                continue

            # 长行视为正文
            if in_toc_zone and len(s) > 30 and not self.toc_guide_pattern.search(s):
                in_toc_zone = False

            cleaned.append(line)

        self.stats.removed_toc_lines += original_count - len(cleaned)
        return cleaned

    def remove_page_noises(self, lines: list[str]) -> list[str]:
        """移除独立的页码行（如 ``- 12 -``、``Page 15``）。"""
        cleaned_lines: list[str] = []
        for line in lines:
            s: str = line.strip()
            if "__PROTECTED_BLOCK_" in s:
                cleaned_lines.append(line)
                continue
            is_noise: bool = False
            if len(s) < 30:
                for p in self.page_patterns:
                    if p.match(s):
                        is_noise = True
                        break
            if not is_noise:
                cleaned_lines.append(line)

        removed: int = len(lines) - len(cleaned_lines)
        self.stats.removed_page_noise_lines += removed
        if removed:
            logger.debug("移除页码噪音行: {} 行", removed)
        return cleaned_lines

    def remove_redundant_headers(self, lines: list[str]) -> tuple[list[str], list[str]]:
        """移除全文中高频出现的短行（通常为页眉 / 页脚）。

        Returns
        -------
        tuple[list[str], list[str]]
            清洗后的行列表 和 被判定为噪音的候选字符串列表。
        """
        line_counts: dict[str, int] = {}
        non_empty_count: int = 0

        for line in lines:
            s: str = line.strip()
            if s and "__PROTECTED_BLOCK_" not in s:
                non_empty_count += 1
                if len(s) < 50:
                    line_counts[s] = line_counts.get(s, 0) + 1

        threshold: float = max(2, non_empty_count * 0.002)
        global_noise_candidates: set[str] = {k for k, v in line_counts.items() if v > threshold}
        sorted_candidates: list[str] = sorted(
            global_noise_candidates,
            key=lambda x: line_counts.get(x, 0),
            reverse=True,
        )

        cleaned_lines: list[str] = []
        for line in lines:
            s = line.strip()
            if not s or "__PROTECTED_BLOCK_" in s:
                cleaned_lines.append(line)
                continue
            if len(s) < 50 and s in global_noise_candidates:
                continue  # 跳过噪音行
            cleaned_lines.append(line)

        removed: int = len(lines) - len(cleaned_lines)
        self.stats.removed_redundant_header_lines += removed
        if removed:
            logger.debug("移除冗余页眉/页脚行: {} 行, 候选: {}", removed, sorted_candidates)
        return cleaned_lines, sorted_candidates

    def remove_inline_noises(
        self,
        lines: list[str],
        global_noise_candidates: list[str] | None = None,
    ) -> list[str]:
        """清除行内的噪音后缀（页码、页眉残留等）。

        [Bug 4 修复] 移除硬编码的 fallback_seeds，改用动态候选 +
        用户自定义模式。
        """
        cleaned_lines: list[str] = []

        # 从全局噪音候选中提取动态种子
        dynamic_noise_seeds: list[str] = []
        if global_noise_candidates:
            dynamic_noise_seeds = [c for c in global_noise_candidates if len(c) > 4][:5]

        # 构建种子匹配模式（数字部分泛化）
        seed_patterns: list[re.Pattern[str]] = []
        for seed in dynamic_noise_seeds:
            pattern_str: str = re.sub(r"\d+", r"\\d+", re.escape(seed))
            seed_patterns.append(re.compile(pattern_str))

        # 加入用户自定义噪音模式
        seed_patterns.extend(self.custom_noise_patterns)

        # 行尾页码后缀模式
        suffix_patterns: list[re.Pattern[str]] = [
            re.compile(r"\s+\d+$"),
            re.compile(r"\s+-\s*\d+\s*-\s*$"),
            re.compile(r"\s+Page\s+\d+\s*$", re.IGNORECASE),
            re.compile(r"\s*/\s*\d+\s*$"),
        ]

        for line in lines:
            s: str = line.rstrip()
            if not s or "__PROTECTED_BLOCK_" in s:
                cleaned_lines.append(s)
                continue

            original: str = s

            # 移除行尾页码后缀
            for p in suffix_patterns:
                s = p.sub("", s)

            # 移除种子噪音
            for p in seed_patterns:
                if p.search(s):
                    s = p.sub("", s).strip()

            if s != original:
                self.stats.removed_inline_noise_count += 1

            cleaned_lines.append(s)

        return cleaned_lines

    def remove_references(self, lines: list[str]) -> list[str]:
        """移除参考文献 / 引用列表区域。

        从「参考文献」或「References」等标题行开始，向下连续匹配
        ``[n] ...`` 格式的条目，直到遇到非引用行为止。
        """
        cleaned: list[str] = []
        in_ref_zone: bool = False
        blank_streak: int = 0
        removed: int = 0

        for line in lines:
            s: str = line.strip()

            # 检测参考文献标题
            if self.reference_header_pattern.match(s):
                in_ref_zone = True
                blank_streak = 0
                removed += 1
                logger.debug("检测到参考文献区域起始: '{}'", s)
                continue

            if in_ref_zone:
                # 空行计数，连续 3 个空行则退出
                if not s:
                    blank_streak += 1
                    if blank_streak >= 3:
                        in_ref_zone = False
                    continue

                blank_streak = 0

                # 匹配引用条目 [n] ... 或 n.
                if self.reference_item_pattern.match(s):
                    removed += 1
                    continue

                # 非引用格式的行 → 退出参考文献区域
                in_ref_zone = False

            cleaned.append(line)

        self.stats.removed_reference_lines += removed
        if removed:
            logger.debug("移除参考文献行: {} 行", removed)
        return cleaned

    def filter_garbage(self, lines: list[str]) -> list[str]:
        """过滤不含任何中英文或数字的纯符号行（保留 Markdown 结构符号行）。"""
        cleaned: list[str] = []
        for line in lines:
            s: str = line.strip()
            if not s or "__PROTECTED_BLOCK_" in s:
                cleaned.append(line)
                continue
            if not re.search(r"[\u4e00-\u9fa5a-zA-Z0-9]", s):
                # 保留 Markdown 结构符号行（标题标记、分隔线等）
                if re.match(r"^[\#\-\*\>\|\s\=]+$", s):
                    cleaned.append(line)
                continue
            cleaned.append(line)

        removed: int = len(lines) - len(cleaned)
        self.stats.removed_garbage_lines += removed
        return cleaned

    # ===================================================================
    # 格式化
    # ===================================================================

    def format_headers(self, lines: list[str]) -> list[str]:
        """为中文章节标题自动添加 Markdown 标题级别前缀（针对无 # 前缀的行）。"""
        new_lines: list[str] = []
        for line in lines:
            s: str = line.strip()
            if not s or "__PROTECTED_BLOCK_" in s or s.startswith("#") or len(s) > 50:
                new_lines.append(line)
                continue
            if self.h2_pattern.match(s):
                new_lines.append("## " + s)
            elif self.h3_pattern.match(s):
                new_lines.append("### " + s)
            elif self.h4_pattern.match(s):
                new_lines.append("#### " + s)
            else:
                new_lines.append(line)
        return new_lines

    def normalize_heading_levels(self, lines: list[str]) -> list[str]:
        """根据编号层级重新分配 Markdown 标题级别。

        PDF 提取的文档通常将所有标题设为 ``#``（h1）。此方法通过分析
        编号模式（``1``、``2.1``、``2.1.1``）和中文章节标题来推断正确的
        层级，让后续文档可按标题层级追溯。

        层级映射规则：
        - 文档标题（已由 ``extract_frontmatter`` 提取）→ ``#``（h1）
        - ``# N 标题``（单级编号）→ ``##``（h2）
        - ``# N.M 标题``（两级编号）→ ``###``（h3）
        - ``# N.M.K 标题``（三级及以上编号）→ ``####``（h4）
        - ``# 第X章`` → ``##``（h2）
        - ``# 第X节`` / ``# 分篇X`` → ``###``（h3）
        - 其他非编号的 ``#`` 标题 → 保持不变
        """
        # 编号层级检测模式
        numbered_heading_pattern: re.Pattern[str] = re.compile(r"^(#+)\s+((\d+(?:\.\d+)*)\s+.+)$")
        # h2: 章 / 篇 / 部 / 部分 / 编 / 中文序号
        chinese_h2_pattern: re.Pattern[str] = re.compile(
            r"^(#+)\s+(("
            r"第[一二三四五六七八九十百零○〇]+[章篇部编]"
            r"|第\d+[章篇部编]"
            r"|第[一二三四五六七八九十百零○〇]+部分"
            r"|第\d+部分"
            r"|[一二三四五六七八九十]+、"
            r").*)$"
        )
        # h3: 节 / 条 / 款 / 项 / 分篇 / 带括号序号
        chinese_h3_pattern: re.Pattern[str] = re.compile(
            r"^(#+)\s+(("
            r"第[一二三四五六七八九十百零○〇]+[节条款项]"
            r"|第\d+[节条款项]"
            r"|分篇[一二三四五六七八九十百]+"
            r"|[（(][一二三四五六七八九十]+[)）]"
            r"|[（(]\d+[)）]"
            r").*)$"
        )

        new_lines: list[str] = []
        for line in lines:
            s: str = line.strip()

            # 跳过空行、保护块、非标题行
            if not s or "__PROTECTED_BLOCK_" in s or not s.startswith("#"):
                new_lines.append(line)
                continue

            # ---- 数字编号标题 ----
            m = numbered_heading_pattern.match(s)
            if m:
                content: str = m.group(2)  # "2.1 对症治疗药物"
                number_part: str = m.group(3)  # "2.1"
                depth: int = number_part.count(".") + 1  # 编号层级深度
                # 层级映射：depth 1 → h2, depth 2 → h3, depth 3+ → h4
                level: int = min(depth + 1, 4)
                new_lines.append("#" * level + " " + content)
                continue

            # ---- 中文 h2 标题（章/篇/部/编/部分/序号） ----
            m = chinese_h2_pattern.match(s)
            if m:
                content = m.group(2)
                new_lines.append("## " + content)
                continue

            # ---- 中文 h3 标题（节/条/款/项/分篇/括号序号） ----
            m = chinese_h3_pattern.match(s)
            if m:
                content = m.group(2)
                new_lines.append("### " + content)
                continue

            # ---- 其他 # 标题保持原样 ----
            new_lines.append(line)

        return new_lines

    def smart_merge_paragraphs(self, lines: list[str]) -> list[str]:
        """智能合并被错误断行的段落。

        规则：
        - 标题行、列表项独占一行并前后留空行
        - 以段落结尾标点结束的行不与下行合并
        - 中文之间直接拼接，ASCII 之间加空格

        [Bug 7 修复] 列表项判断统一使用 ``list_pattern``，不再通过
        ``is_title_or_list`` 重复匹配。
        """
        merged_lines: list[str] = []
        buffer: str = ""

        for line in lines:
            s: str = line.strip()
            is_list_item: bool = bool(self.list_pattern.match(s))

            # 如果 buffer 本身是列表项，先输出
            if buffer and bool(self.list_pattern.match(buffer)):
                merged_lines.append(buffer)
                merged_lines.append("")
                buffer = ""

            # 空行：输出 buffer
            if not s:
                if buffer:
                    merged_lines.append(buffer)
                    merged_lines.append("")
                    buffer = ""
                continue

            # 保护块：直接输出
            if "__PROTECTED_BLOCK_" in s:
                if buffer:
                    merged_lines.append(buffer)
                    buffer = ""
                merged_lines.append(s)
                merged_lines.append("")
                continue

            # [Bug 7 修复] 标题 / 列表项：独占一行，前后留空
            is_title: bool = self.is_title_or_list(s)
            if is_title or is_list_item:
                if buffer:
                    merged_lines.append(buffer)
                    merged_lines.append("")
                    buffer = ""
                if merged_lines and merged_lines[-1] != "":
                    merged_lines.append("")
                merged_lines.append(s)
                merged_lines.append("")
                continue

            # 首行：存入 buffer
            if not buffer:
                buffer = s
                continue

            # 含有表格分隔符：不合并
            if "|" in buffer or "|" in s:
                merged_lines.append(buffer)
                merged_lines.append("")
                buffer = s
                continue

            # 段落结尾标点：不合并
            if self.is_paragraph_end(buffer):
                merged_lines.append(buffer)
                merged_lines.append("")
                buffer = s
                continue

            # 拼接规则：ASCII 之间加空格，中文之间直接拼接
            if buffer[-1].isascii() and s[0].isascii():
                buffer = buffer + " " + s
            else:
                buffer = buffer + s

        if buffer:
            merged_lines.append(buffer)

        return merged_lines

    # ===================================================================
    # 主清洗流程
    # ===================================================================

    def clean_text(self, text: str | None) -> str:
        """执行完整的 Markdown 清洗流程。

        Parameters
        ----------
        text : str | None
            待清洗的 Markdown 文本。

        Returns
        -------
        str
            清洗后带 YAML frontmatter 的纯净文本。

        Raises
        ------
        TypeError
            当 ``text`` 既不是 ``str`` 也不是 ``None`` 时。
        """
        # 输入校验
        if text is None:
            logger.warning("输入文本为 None，返回空文档。")
            return "---\ntitle: 未命名文档\n---\n"
        if not isinstance(text, str):
            raise TypeError(f"期望 str 类型，收到 {type(text).__name__}")
        if not text.strip():
            logger.warning("输入文本为空白字符串，返回空文档。")
            return "---\ntitle: 未命名文档\n---\n"

        # 重置统计
        self.stats = CleanStats()
        logger.info("开始 Markdown 清洗...")

        # 1. 提取标题 & 预处理
        text = self.extract_frontmatter(text)
        text = self.normalize_text(text)

        # 2. 表格线性化（Markdown 格式）
        if self.options.linearize_tables:
            text = self.linearize_tables(text)

        # 3. 保护代码块和公式块
        text = self.protect_blocks(text)

        # 4. HTML 内容清洗（<img>/<a>/<table> 及其他标签）
        if self.options.clean_html_tags:
            text = self.clean_html_content(text)

        # 5. Markdown 图片 / 链接
        text = self.clean_images_and_links(text)

        # 5. 参考文献移除（必须在引用标记清除之前，否则 [n] 被删后无法识别条目）
        if self.options.remove_references:
            ref_lines: list[str] = text.split("\n")
            ref_lines = self.remove_references(ref_lines)
            text = "\n".join(ref_lines)

        # 6. 引用标记清除
        if self.options.remove_citations:
            text = self.remove_citations(text)

        # 7. 逐行处理
        lines: list[str] = text.split("\n")

        if self.options.remove_toc:
            lines = self.remove_toc(lines)

        if self.options.remove_page_noise:
            lines = self.remove_page_noises(lines)

        global_noise_candidates: list[str] = []
        if self.options.remove_redundant_headers:
            lines, global_noise_candidates = self.remove_redundant_headers(lines)

        if self.options.remove_inline_noise:
            lines = self.remove_inline_noises(lines, global_noise_candidates)

        lines = self.filter_garbage(lines)

        if self.options.format_headers:
            lines = self.format_headers(lines)
            lines = self.normalize_heading_levels(lines)

        if self.options.merge_paragraphs:
            lines = self.smart_merge_paragraphs(lines)

        # 7. 组装输出
        result_text: str = "\n".join(lines)
        result_text = self.restore_blocks(result_text)
        result_text = re.sub(r"\n{3,}", "\n\n", result_text)

        yaml_frontmatter: str = f"---\ntitle: {self.doc_title}\n---\n\n"

        logger.info(
            "清洗完成 | 图片:{} 链接:{} 引用:{} 目录行:{} 页码行:{} "
            "页眉行:{} 行内噪音:{} 垃圾行:{} 参考文献行:{} 表格:{} HTML标签:{}",
            self.stats.removed_images,
            self.stats.removed_links,
            self.stats.removed_citations,
            self.stats.removed_toc_lines,
            self.stats.removed_page_noise_lines,
            self.stats.removed_redundant_header_lines,
            self.stats.removed_inline_noise_count,
            self.stats.removed_garbage_lines,
            self.stats.removed_reference_lines,
            self.stats.tables_linearized,
            self.stats.removed_html_tags,
        )

        return yaml_frontmatter + result_text.strip() + "\n"

    def clean(self, file_path: Path | str, output_path: Path | str) -> str:
        """清洗 Markdown 文件。

        Parameters
        ----------
        file_path : Path | str
            待清洗的 Markdown 文件路径。
        output_path : Path | str
            清洗后文件的保存路径。

        Returns
        -------
        str
            清洗后的纯净文本。
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with open(file_path, encoding="utf-8") as f:
            text: str = f.read()
        result_text = self.clean_text(text)
        if isinstance(output_path, str):
            output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)
        return result_text
