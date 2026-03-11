"""
模块名称：excel_parser
功能描述：通用 Excel 文件解析器，提供自动列角色推断与结构化的逐行数据迭代能力。
         可处理药品说明书等已知数据源（通过预定义配置），
         也可自动分析用户上传的任意 Excel 文件。
使用说明: ExcelParser 为主要类，用于解析 Excel 文件。包含以下方法：

- ``__init__(self, config: ExcelConfig)``: 初始化解析器，需提供 Excel 配置。
- ``parse(self, file_path: Path, sheet_name: str | int | None ) -> Iterator[ExcelPasedRow]``: 解析 Excel 文件，返回逐行解析结果。

"""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志初始化 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "excel_parser"
logger = get_channel_logger(LOG_FILE_DIR, "excel_parser")
# =============================================


# ---------- 数据类型定义 ----------


class ColumnRole(str, Enum):
    """
    列角色枚举：描述一列在 RAG 流水线中扮演的角色。
    用于识别哪些列是元数据，哪些是主要内容，哪些是噪音列即需要跳过的列
    """

    METADATA = "metadata"
    """元数据列，如 ID、名称、分类等短文本，不参与 Embedding，但附加到 chunk 作为过滤/溯源信息。"""

    CONTENT = "content"
    """内容列，如适应症、不良反应等长文本，将被构建为 chunk 并 Embedding。"""

    SKIP = "skip"
    """跳过列，如空列、无意义列、爬虫残留等。"""


class ColumnProfile:
    """单列的统计画像，用于自动推断角色。"""

    def __init__(
        self,
        name: str,
        fill_rate: float,
        avg_length: float,
        unique_ratio: float,
        max_length: int,
    ) -> None:
        self.name = name
        self.fill_rate = fill_rate
        self.avg_length = avg_length
        self.unique_ratio = unique_ratio
        self.max_length = max_length

    def __repr__(self) -> str:
        return (
            f"ColumnProfile(name={self.name!r}, fill={self.fill_rate:.0%}, "
            f"avg_len={self.avg_length:.0f}, unique={self.unique_ratio:.0%})"
        )


class ColumnConfig:
    """
    列角色配置：支持手动精确指定或留空交由自动推断。

    示例 (药品说明书)::

        cfg = ColumnConfig(
            skip=["r3", "标题链接"],
            metadata=["标题", "编号", "通用名称", "批准文号", "药品分类", "生产企业"],
            content=["适应症", "不良反应", "用法用量", "禁忌", "注意事项"],
            context_prefix_field="通用名称",
        )
    """

    def __init__(
        self,
        *,
        skip: list[str] | None = None,
        metadata: list[str] | None = None,
        content: list[str] | None = None,
        context_prefix_field: str | None = None,
    ) -> None:
        self.skip: set[str] = set(skip or [])
        self.metadata: set[str] = set(metadata or [])
        self.content: set[str] = set(content or [])
        self.context_prefix_field: str | None = context_prefix_field

    def get_role(self, column_name: str) -> ColumnRole | None:
        """如果该列在配置中有明确指定，则返回角色；否则返回 None（交给自动推断）。"""
        if column_name in self.skip:
            return ColumnRole.SKIP
        if column_name in self.metadata:
            return ColumnRole.METADATA
        if column_name in self.content:
            return ColumnRole.CONTENT
        return None


# =============================================
# 药品说明书的列配置
# =============================================
DRUG_CONFIG = ColumnConfig(
    skip=["r3", "标题链接"],
    metadata=[
        "标题",
        "编号",
        "通用名称",
        "商品名称",
        "汉语拼音",
        "批准文号",
        "药品分类",
        "生产企业",
        "药品性质",
        "规格",
        "有效期",
    ],
    content=[
        "相关疾病",
        "性状",
        "主要成份",
        "适应症",
        "不良反应",
        "用法用量",
        "禁忌",
        "注意事项",
        "孕妇及哺乳期妇女用药",
        "儿童用药",
        "老人用药",
        "药物相互作用",
        "药理毒理",
        "药代动力学",
        "贮藏",
    ],
    context_prefix_field="通用名称",
)


# ---------- 列分析器 ----------


class ColumnAnalyzer:
    """
    自动分析 DataFrame 中每列的数据特征，推断其在 RAG 流水线中应扮演的角色。

    推断规则（按优先级执行）：
    1. 填充率 < 30%             → SKIP
    2. 平均长度 > 80            → CONTENT
    3. 平均长度 < 30 且唯一值 < 20 → METADATA（枚举/标签）
    4. 唯一值占比 > 90% 且平均长度 < 50 → METADATA（ID 类）
    5. 其余                     → METADATA（默认）
    """

    # ---------- 可调阈值 ----------
    FILL_RATE_THRESHOLD: float = 0.30
    CONTENT_AVG_LEN_THRESHOLD: float = 80.0
    METADATA_SHORT_LEN_THRESHOLD: float = 30.0
    METADATA_ENUM_UNIQUE_LIMIT: int = 20
    METADATA_ID_UNIQUE_RATIO: float = 0.90
    METADATA_ID_MAX_AVG_LEN: float = 50.0

    @classmethod
    def profile_columns(cls, df: pd.DataFrame) -> dict[str, ColumnProfile]:
        """
        计算每列的统计画像。

        :param df: 待分析的 DataFrame（应已移除表头行）
        :return: {列名: ColumnProfile}
        """
        profiles: dict[str, ColumnProfile] = {}
        total_rows = len(df)
        if total_rows == 0:
            return profiles

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            fill_rate = len(non_null) / total_rows

            str_series = non_null.astype(str)
            lengths = str_series.str.len()
            avg_length = float(lengths.mean()) if len(lengths) > 0 else 0.0
            max_length = int(lengths.max()) if len(lengths) > 0 else 0

            n_unique = non_null.nunique()
            unique_ratio = n_unique / len(non_null) if len(non_null) > 0 else 0.0

            profiles[col] = ColumnProfile(
                name=col,
                fill_rate=fill_rate,
                avg_length=avg_length,
                unique_ratio=unique_ratio,
                max_length=max_length,
            )
        return profiles

    @classmethod
    def infer_roles(
        cls,
        df: pd.DataFrame,
        column_config: ColumnConfig | None = DRUG_CONFIG,
    ) -> dict[str, ColumnRole]:
        """
        为每列分配角色：先查配置，未命中则走自动推断。

        :param df: 待分析的 DataFrame
        :param column_config: 可选的手动配置（优先级高于自动推断）
        :return: {列名: ColumnRole}
        """
        profiles = cls.profile_columns(df)
        roles: dict[str, ColumnRole] = {}

        for col, prof in profiles.items():
            # 1) 配置优先
            if column_config is not None:
                manual_role = column_config.get_role(col)
                if manual_role is not None:
                    roles[col] = manual_role
                    continue

            # 2) 自动推断
            if prof.fill_rate < cls.FILL_RATE_THRESHOLD:
                roles[col] = ColumnRole.SKIP
            elif prof.avg_length > cls.CONTENT_AVG_LEN_THRESHOLD:
                roles[col] = ColumnRole.CONTENT
            elif (
                prof.avg_length < cls.METADATA_SHORT_LEN_THRESHOLD
                and prof.unique_ratio < cls.METADATA_ENUM_UNIQUE_LIMIT / max(len(df), 1)
            ):
                roles[col] = ColumnRole.METADATA
            elif (
                prof.unique_ratio > cls.METADATA_ID_UNIQUE_RATIO
                and prof.avg_length < cls.METADATA_ID_MAX_AVG_LEN
            ):
                roles[col] = ColumnRole.METADATA
            else:
                # 默认当元数据处理（保守策略）
                roles[col] = ColumnRole.METADATA

        logger.info(
            f"📊 列角色推断完成 | CONTENT={sum(1 for r in roles.values() if r == ColumnRole.CONTENT)}, "
            f"METADATA={sum(1 for r in roles.values() if r == ColumnRole.METADATA)}, "
            f"SKIP={sum(1 for r in roles.values() if r == ColumnRole.SKIP)}"
        )
        return roles


# ---------- 解析后的数据结构 ----------


class ExcelPasedRow:
    """
    表示一行 Excel 数据经解析后的结构化结果。

    Attributes:
        row_index:  原始行号（从 0 开始，不含表头）
        metadata:   元数据键值对 {列名: 值}
        contents:   内容键值对 {列名: 文本内容}
        source_file: 来源文件路径
        sheet_name:  工作表名称
    """

    __slots__ = ("row_index", "metadata", "contents", "source_file", "sheet_name")

    def __init__(
        self,
        row_index: int,
        metadata: dict[str, Any],
        contents: dict[str, str],
        source_file: str,
        sheet_name: str,
    ) -> None:
        self.row_index = row_index
        self.metadata = metadata
        self.contents = contents
        self.source_file = source_file
        self.sheet_name = sheet_name

    def __repr__(self) -> str:
        meta_keys = list(self.metadata.keys())
        content_keys = list(self.contents.keys())
        return f"ExcelPasedRow(row={self.row_index}, meta_keys={meta_keys}, content_keys={content_keys})"


# ---------- 核心解析器 ----------


class ExcelParser:
    """
    通用 Excel 解析器。

    - 读取 Excel 文件（支持 .xlsx / .xls）
    - 自动或按配置识别每列的角色
    - 逐行输出结构化的 ``ExcelPasedRow``

    典型用法::

        parser = ExcelParser(column_config=ColumnConfig(
            skip=["r3"],
            context_prefix_field="通用名称",
        ))
        for row in parser.parse("药品说明书.xlsx"):
            print(row.metadata, row.contents.keys())
    """

    def __init__(self, column_config: ColumnConfig | None = None) -> None:
        """
        :param column_config: 列角色配置。传入则优先使用配置；未指定的列走自动推断。
        """
        self.column_config = column_config

    def parse(
        self,
        file_path: str | Path,
        *,
        sheet_name: str | int | None = None,
    ) -> Iterator[ExcelPasedRow]:
        """
        解析 Excel 文件，逐行生成 ``ExcelPasedRow``。

        :param file_path: Excel 文件路径
        :param sheet_name: 指定工作表名称/索引。为 None 时解析所有 sheet。
        :yields: ExcelPasedRow
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Excel 文件不存在: {file_path}")

        logger.info(f"📂 开始解析 Excel: {file_path.name}")

        # 读取所有 sheet 或指定 sheet
        sheets: dict[str, pd.DataFrame]
        if sheet_name is not None:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            name = sheet_name if isinstance(sheet_name, str) else f"Sheet{sheet_name}"
            sheets = {name: df}
        else:
            sheets = pd.read_excel(file_path, sheet_name=None)

        source = str(file_path)

        for sname, df in sheets.items():
            logger.info(f"📄 解析工作表: {sname} ({len(df)} 行, {len(df.columns)} 列)")
            yield from self._parse_sheet(df, source, str(sname))

    def _parse_sheet(
        self,
        df: pd.DataFrame,
        source_file: str,
        sheet_name: str,
    ) -> Iterator[ExcelPasedRow]:
        """解析单个 sheet，逐行产出 ExcelPasedRow。"""
        # 推断列角色
        roles = ColumnAnalyzer.infer_roles(df, self.column_config)

        metadata_cols = [c for c, r in roles.items() if r == ColumnRole.METADATA]
        content_cols = [c for c, r in roles.items() if r == ColumnRole.CONTENT]
        skip_cols = [c for c, r in roles.items() if r == ColumnRole.SKIP]

        logger.info(
            f"  ✅ 元数据列 ({len(metadata_cols)}): {metadata_cols}\n"
            f"  ✅ 内容列   ({len(content_cols)}): {content_cols}\n"
            f"  ⏭️ 跳过列   ({len(skip_cols)}): {skip_cols}"
        )

        for idx, row in df.iterrows():
            # 收集元数据
            metadata: dict[str, Any] = {}
            for col in metadata_cols:
                val = row[col]
                if pd.notna(val):
                    metadata[col] = val

            # 收集内容
            contents: dict[str, str] = {}
            for col in content_cols:
                val = row[col]
                if pd.notna(val):
                    text = str(val).strip()
                    if text:
                        contents[col] = text

            # 跳过完全空的行（既无元数据也无内容）
            if not metadata and not contents:
                continue

            yield ExcelPasedRow(
                row_index=int(idx),
                metadata=metadata,
                contents=contents,
                source_file=source_file,
                sheet_name=sheet_name,
            )

    def get_column_analysis_report(
        self,
        file_path: str | Path,
        sheet_name: str | int = 0,
    ) -> str:
        """
        生成列分析报告（用于调试/检查自动推断结果）。

        :param file_path: Excel 文件路径
        :param sheet_name: 工作表名或索引
        :return: 可打印的报告文本
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        profiles = ColumnAnalyzer.profile_columns(df)
        roles = ColumnAnalyzer.infer_roles(df, self.column_config)

        lines: list[str] = [
            f"=== 列分析报告: {Path(file_path).name} ===",
            f"总行数: {len(df)}, 总列数: {len(df.columns)}",
            "",
            f"{'列名':<20} {'角色':<10} {'填充率':>6} {'平均长度':>8} {'最大长度':>8} {'唯一率':>6}",
            "-" * 70,
        ]
        for col in df.columns:
            prof = profiles[col]
            role = roles[col]
            lines.append(
                f"{prof.name:<20} {role.value:<10} "
                f"{prof.fill_rate:>5.0%} {prof.avg_length:>8.0f} "
                f"{prof.max_length:>8d} {prof.unique_ratio:>5.0%}"
            )
        return "\n".join(lines)
