import json
import os
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

from silver_pilot.config import config
from silver_pilot.utils import LogManager

# ================= 配置区域 =================
# 1. Neo4j 数据库连接
URI = config.NEO4J_URI  # 请修改为你的 Neo4j 地址
AUTH = (config.NEO4J_USER, config.NEO4J_PASSWORD)  # 请修改为你的用户名和密码

# 2. 文件路径
TRIPLETS_FILE_PATH = config.DATA_DIR / "processed/KG/triplets"
ALIGNMENT_FILE = (
    config.DATA_DIR / "processed/KG/entity_alignment/entity_alignment_results.csv"
)  # 你的 CSV 对齐结果文件

# 3. 日志配置
LOG_FILE_DIR = config.DATA_DIR / "processed/KG/import_to_neo4j"
log_manager = LogManager(LOG_FILE_DIR, "KnowledgeGraphImporter")
logger = log_manager.get_logger()
# ===========================================


class KnowledgeGraphImporter:
    def __init__(self, uri: str, auth: tuple[str, str]):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self) -> None:
        self.driver.close()

    def normalize_label(self, label: str) -> str:
        """清洗Label格式，去除冒号，并处理非法输入"""
        # 1. 处理空值 (NaN, None)
        if pd.isna(label) or label == "":
            return "Unknown"

        # 2. 强制转为字符串 (防止 CSV 里某些 label 被误读为数字)
        label_str = str(label)

        # 3. 去除冒号和首尾空格
        return label_str.replace(":", "").strip()

    def load_alignment_map(self, csv_path: str | Path) -> dict[tuple[str, str], dict[str, str]]:
        """
        加载对齐结果 CSV，构建映射字典
        Key: (extracted_name, label)
        Value: { 'target_name': 标准名, 'action': 动作 }
        """
        if not os.path.exists(csv_path):
            logger.error(f"找不到对齐文件: {csv_path}")
            return {}

        df = pd.read_csv(csv_path)
        mapping = {}

        # 确保 label 列也是清洗过的
        df["label"] = df["label"].apply(self.normalize_label)

        for _, row in df.iterrows():
            extracted_name = row["extracted_name"]
            label = row["label"]
            action = row["action"]
            matched_name = row["matched_name"]

            # 复合键：因为同一个名字在不同Label下可能有不同含义
            key = (extracted_name, label)

            if "Link Existing" in action:
                # 如果是链接已有，目标名字使用匹配到的标准名
                target_name = matched_name
            else:
                # 如果是新建，目标名字保持原样
                target_name = extracted_name

            mapping[key] = {"target_name": target_name, "action": action}

        logger.info(f"已加载 {len(mapping)} 条实体映射规则。")
        return mapping

    def prepare_data(
        self, json_path: str | Path, mapping: dict[tuple[str, str], dict[str, str]]
    ) -> list[dict]:
        """
        读取 JSON 并利用映射字典清洗数据
        """
        if not os.path.exists(json_path):
            logger.error(f"找不到三元组文件: {json_path}")
            return []

        cleaned_triplets = []

        # 兼容单文件或文件夹读取
        files = []
        if os.path.isdir(json_path):
            files = [
                os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith(".json")
            ]
        else:
            files = [json_path]  # type: ignore[list-item]

        for file_path in files:
            with open(file_path, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]

                    for item in data:
                        if "extraction" not in item:
                            continue

                        source_file = item.get("source_chunk", {}).get("source_file", "Unknown")

                        for t in item["extraction"].get("triples", []):
                            # 1. 获取原始信息
                            start_raw = t["start"]
                            start_label = self.normalize_label(t["start_label"])
                            end_raw = t["end"]
                            end_label = self.normalize_label(t["end_label"])
                            rel_type = t["type_"]

                            # 2. 查找映射 (如果没有映射，默认使用原名，视为新建)
                            start_info = mapping.get(
                                (start_raw, start_label),
                                {"target_name": start_raw, "action": "Create New"},
                            )
                            end_info = mapping.get(
                                (end_raw, end_label),
                                {"target_name": end_raw, "action": "Create New"},
                            )

                            # 3. 构建清洗后的数据
                            cleaned_triplets.append(
                                {
                                    "start_name": start_info["target_name"],
                                    "start_label": start_label,
                                    "start_is_new": "Create New" in start_info["action"],
                                    "end_name": end_info["target_name"],
                                    "end_label": end_label,
                                    "end_is_new": "Create New" in end_info["action"],
                                    "rel_type": rel_type,
                                    "source": source_file,
                                }
                            )

                except Exception as e:
                    logger.warning(f"处理文件 {file_path} 时出错: {e}")

        logger.info(f"共准备了 {len(cleaned_triplets)} 条待写入的三元组。")
        return cleaned_triplets

    def batch_import(self, triplets: list[dict]) -> None:
        """
        按关系类型分组，批量写入 Neo4j
        """
        if not triplets:
            return

        # 按关系类型分组 (Cypher 无法动态传关系类型，必须拼在语句里)
        # 格式：(rel_type, start_label, end_label)
        grouped = {}  # type: ignore[var-annotated]
        for t in triplets:
            key = (t["rel_type"], t["start_label"], t["end_label"])

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(t)

        total_batches = len(grouped)
        current_batch = 0

        # 解包 key
        for (rel_type, start_label, end_label), batch in grouped.items():
            current_batch += 1

            logger.info(
                f"[{current_batch}/{total_batches}] "
                + f"正在写入: ({start_label})-[{rel_type}]->({end_label}) 共 {len(batch)} 条..."
            )

            # 动态构建 Cypher
            # 注意：使用了 APOC 库的 merge.node 也可以，但原生 Cypher 更通用
            # 逻辑：
            # 1. MERGE 头实体 (如果是新实体，加 source 属性)
            # 2. MERGE 尾实体
            # 3. MERGE 关系 (防止重复)

            try:
                with self.driver.session() as session:
                    query = f"""
                    UNWIND $batch AS row

                    // 1. 处理头实体
                    MERGE (s:`{start_label}` {{name: row.start_name}})
                    ON CREATE SET s.source = 'PDF_Extraction', s.created_at = timestamp()

                    // 2. 处理尾实体
                    MERGE (e:`{end_label}` {{name: row.end_name}})
                    ON CREATE SET e.source = 'PDF_Extraction', e.created_at = timestamp()

                    // 3. 处理关系
                    MERGE (s)-[r:`{rel_type}`]->(e)
                    ON CREATE SET r.source = row.source, r.confidence = 'High', r.created_at = timestamp()
                    """

                    session.run(query, batch=batch)

            except Exception as e:
                logger.error(f"❌ 写入 [{rel_type}] 批次失败: {e}")


if __name__ == "__main__":
    importer = KnowledgeGraphImporter(URI, AUTH)

    try:
        # 1. 加载映射字典
        mapping_dict = importer.load_alignment_map(ALIGNMENT_FILE)

        # 2. 准备数据 (应用映射)
        triplets_data = importer.prepare_data(TRIPLETS_FILE_PATH, mapping_dict)

        # 3. 执行写入
        importer.batch_import(triplets_data)

    finally:
        importer.close()
        logger.info("任务结束。")
