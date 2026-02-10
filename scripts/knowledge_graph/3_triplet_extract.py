import glob
import json
import os
import re
import time
from typing import Any

import dashscope
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager

# ================= 配置区域 =================
# 请替换为你的 DashScope API Key
DASHSCOPE_API_KEY = config.DASHSCOPE_API_KEY

# 输入和输出目录
INPUT_DIR = config.DATA_DIR / "processed/KG/pdf_chunks"
OUTPUT_DIR = config.DATA_DIR / "processed/KG/triplets"

# 模型名称
MODEL_NAME = "qwen3-max-2026-01-23"

# ===========================================

# 设置 API Key
dashscope.api_key = config.DASHSCOPE_API_KEY


# ================= Pydantic Schema 定义 =================
class KGTriplet(BaseModel):
    start: str = Field(description="头实体")
    start_label: str = Field(description="头实体类型")
    type_: str = Field(description="关系类型")
    end: str = Field(description="尾实体")
    end_label: str = Field(description="尾实体类型")


class ExtractionOutput(BaseModel):
    step_by_step_analysis: str = Field(description="思维链推理过程")
    triples: list[KGTriplet] = Field(description="三元组列表")


# ================= Prompt 模板路径 =================

prompt_path = config.SCRIPTS_DIR / "knowledge_graph/prompt_triplet_extract.yaml"

# ================= 核心处理函数 =================


def clean_json_string(json_str: str) -> str:
    """
    清洗 LLM 返回的字符串，去除 Markdown 标记
    """
    # 去除 ```json 和 ```
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, json_str, re.DOTALL)
    if match:
        return match.group(1)
    return json_str.strip()


def process_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    调用 DashScope 处理单个 Chunk
    """
    header_path = chunk.get("header_path", "")
    chunk_text = chunk.get("content", "")

    messages = prompt_manager.render(prompt_path, header_path=header_path, chunk_text=chunk_text)

    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        response = client.chat.completions.parse(
            model=MODEL_NAME,
            messages=messages,  # type: ignore[arg-type]
            response_format=ExtractionOutput,
            temperature=0.1,
        )
        data = response.choices[0].message.parsed
        # 【修复点】将 Pydantic 对象转换为字典
        if data:
            return data.model_dump()
        else:
            # 防御性编程：如果解析失败返回 None
            return {"error": "PARSE_FAILED", "raw": response.choices[0].message.content}

    except Exception as e:
        print(f"未知错误: {e}")
        return {"error": "UNKNOWN_ERROR", "msg": str(e)}


# ================= 主程序 =================


def main() -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(INPUT_DIR):
        print(f"输入目录不存在: {INPUT_DIR}")
        return

    json_files = glob.glob(os.path.join(INPUT_DIR, "*_chunks.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件待处理...")

    for json_file in tqdm(json_files, desc="处理进度"):
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_filename = base_name + "_triplets.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # 如果输出文件已存在，跳过（断点续传）
        if os.path.exists(output_path):
            print(f"跳过已存在文件: {output_filename}")
            continue
        print(f"正在处理: {json_file}")

        with open(json_file, encoding="utf-8") as f:
            chunks = json.load(f)

        extracted_results = []

        # 使用 tqdm 显示进度条
        for chunk in tqdm(chunks, desc="抽取中"):
            result = process_chunk(chunk)

            # 将原始信息和抽取结果合并，方便追溯
            record = {"source_chunk": chunk, "extraction": result}
            extracted_results.append(record)

            # 简单的流控，避免触发 API 限流 (根据你的账号等级调整)
            time.sleep(0.5)

        # 保存结果
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(extracted_results, out_f, ensure_ascii=False, indent=2)

        print(f"✅ 完成，结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
