import glob
import json
import os
import re

from tqdm import tqdm

from silver_pilot.config import config

# ================= 配置区域 =================

INPUT_DIR = config.DATA_DIR / "raw" / "KG" / "md_new"
OUTPUT_DIR = config.DATA_DIR / "processed" / "KG" / "pdf_chunks"

CHUNK_SIZE = 800
OVERLAP = 50
# ===========================================


def smart_split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    智能切分：
    用于处理单个标题下过长的文本内容。
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # 1. 初步确定结束位置
        end = start + chunk_size

        # 如果已经超出文本长度，直接取到最后，并修正起始位置以保证 chunk_size
        if end >= len(text):
            chunks.append(text[-chunk_size:])  # 强制取最后 chunk_size 长度
            break

        # 2. 寻找最佳切分点 (避免切断单词)
        # 在 end 附近向前寻找最近的分隔符 (换行 > 句号 > 空格)
        # 我们只在 end 之前 50 个字符范围内找，找不到就硬切
        search_range = 50
        best_end = end

        # 截取边界附近的文本
        buffer = text[end - search_range : end]

        # 优先级 1: 换行符
        newline_idx = buffer.rfind("\n")
        if newline_idx != -1:
            best_end = end - search_range + newline_idx + 1  # +1 是为了保留换行符
        else:
            # 优先级 2: 句号 (中文或英文)
            period_idx = max(buffer.rfind("。"), buffer.rfind("."))
            if period_idx != -1:
                best_end = end - search_range + period_idx + 1
            else:
                # 优先级 3: 空格
                space_idx = buffer.rfind(" ")
                if space_idx != -1:
                    best_end = end - search_range + space_idx + 1

        # 3. 提取 Chunk
        chunk = text[start:best_end]
        chunks.append(chunk)

        # 4. 计算下一个 Start (回退 Overlap)
        # 下一个块的起始点 = 当前结束点 - 重叠量
        start = best_end - overlap

    return chunks


def parse_markdown_with_path(
    md_content: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> list[dict]:
    """
    解析 Markdown 的标题层级，生成带上下文路径的 chunks。
    """
    lines = md_content.split("\n")
    headers_stack: list[str] = []  # 标题栈，例如 ['一级标题', '二级标题']
    current_section_content: list[str] = []
    final_chunks = []

    def flush_buffer() -> None:
        """将当前缓存的文本处理并加入结果集"""
        if not current_section_content:
            return

        full_text = "\n".join(current_section_content).strip()
        if not full_text:
            return

        # 1. 构造当前路径字符串
        # 如果栈为空，说明是文件开头的无标题引言，标记为 Root
        path_str = " > ".join(headers_stack) if headers_stack else "Document Root"

        # 2. 检查长度并切分
        # 如果该段落很长，用 smart_split_text 切成多段，但它们共享同一个 path
        sub_chunks = smart_split_text(full_text, chunk_size, overlap)

        for sub_c in sub_chunks:
            final_chunks.append({"header_path": path_str, "content": sub_c})

    for line in lines:
        # 正则匹配标题行 (例如: ## 不良反应)
        header_match = re.match(r"^(#+)\s+(.*)", line)

        if header_match:
            # 遇到新标题前，先保存上一段内容
            flush_buffer()
            current_section_content = []

            # 更新标题栈
            level = len(header_match.group(1))  # '#' 的数量
            title = header_match.group(2).strip()

            # 逻辑：如果新标题级别 N <= 当前栈深度，说明层级结束，弹出旧标题
            # 例如：在 ### 之后遇到 ##，说明 ### 结束了
            while len(headers_stack) >= level:
                headers_stack.pop()

            headers_stack.append(title)
        else:
            # 非标题行，归入当前段落
            current_section_content.append(line)

    # 循环结束后，别忘了处理最后一段内容
    flush_buffer()

    return final_chunks


def main() -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 支持 input_dir 为字符串或 Path 对象
    md_files = glob.glob(os.path.join(str(INPUT_DIR), "*.md"))
    print(f"找到 {len(md_files)} 个 MD 文件，开始语义结构化切分...\n")

    for file_path in tqdm(md_files, desc="处理文件"):
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # 调用新的带有路径解析的切分函数
        structured_chunks = parse_markdown_with_path(content, CHUNK_SIZE, OVERLAP)

        # 添加源文件名信息
        for chunk in structured_chunks:
            chunk["source_file"] = base_name

        # ================= 输出为 JSON =================
        # 方案：每个原始 MD 文件生成一个对应的 JSON 文件，里面包含该文件的所有 chunks 列表
        output_filename = f"{base_name}_chunks.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with open(output_path, "w", encoding="utf-8") as out_f:
            # ensure_ascii=False 保证中文正常显示
            # indent=2 为了方便人类阅读查看，生产环境可去掉以节省空间
            json.dump(structured_chunks, out_f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"请确保输入目录存在: {INPUT_DIR}")
    else:
        main()
