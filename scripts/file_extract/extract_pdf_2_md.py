"""
模块名称：extract_pdf_2_md
功能描述：批量扫描指定文件夹中的 PDF 文件，调用 MinerU 服务进行 PDF→Markdown 转换，
         输出目录为 PDF 文件所在的原文件夹。
"""

from pathlib import Path

from silver_pilot.tools.document import MinerUConverter

# ========== 配置：在此处添加要处理的文件夹路径 ==========

FOLDERS_TO_PROCESS: list[str | Path] = []

# ==================================================


def collect_pdfs(folders: list[str | Path]) -> dict[Path, list[Path]]:
    """
    扫描文件夹列表，收集所有 PDF 文件并按所属文件夹分组。

    Args:
        folders: 要扫描的文件夹路径列表

    Returns:
        dict[Path, list[Path]]: 以文件夹路径为键、该文件夹下的 PDF 文件列表为值
    """
    grouped: dict[Path, list[Path]] = {}
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"[警告] 文件夹不存在，已跳过: {folder_path}")
            continue
        if not folder_path.is_dir():
            print(f"[警告] 路径不是文件夹，已跳过: {folder_path}")
            continue

        pdf_files = sorted(folder_path.glob("*.pdf"))
        if not pdf_files:
            print(f"[信息] 文件夹中未找到 PDF 文件: {folder_path}")
            continue

        grouped[folder_path] = pdf_files
        print(f"[信息] {folder_path} -> 找到 {len(pdf_files)} 个 PDF 文件")

    return grouped


def main() -> None:
    if not FOLDERS_TO_PROCESS:
        print("[错误] FOLDERS_TO_PROCESS 为空，请在脚本顶部添加要处理的文件夹路径。")
        return

    # 1. 收集所有 PDF 文件，按文件夹分组
    grouped_pdfs = collect_pdfs(FOLDERS_TO_PROCESS)
    if not grouped_pdfs:
        print("[错误] 未找到任何 PDF 文件，程序退出。")
        return

    total_files = sum(len(files) for files in grouped_pdfs.values())
    print(f"\n共找到 {total_files} 个 PDF 文件，分布在 {len(grouped_pdfs)} 个文件夹中。\n")

    # 2. 逐个文件夹调用 MinerU 处理，output_dir 为 PDF 文件的原文件夹
    service = MinerUConverter()
    for folder, pdf_files in grouped_pdfs.items():
        print(f"{'=' * 60}")
        print(f"正在处理文件夹: {folder}")
        print(f"PDF 文件数量: {len(pdf_files)}")
        for f in pdf_files:
            print(f"  - {f.name}")
        print(f"输出目录: {folder}")
        print(f"{'=' * 60}")

        try:
            result_dirs = service.process(
                file_paths=pdf_files,
                output_dir=folder,
            )
            print(f"[成功] 文件夹处理完成: {folder}")
            for d in result_dirs:
                print(f"  结果目录: {d}")
        except Exception as e:
            print(f"[错误] 处理文件夹 {folder} 时出错: {e}")
            continue

        print()

    print("所有文件夹处理完毕。")


if __name__ == "__main__":
    main()
