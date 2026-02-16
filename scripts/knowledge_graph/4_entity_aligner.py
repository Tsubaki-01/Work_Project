import json
import os
import sys
from collections import defaultdict

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from silver_pilot.config import config
from silver_pilot.utils.log import LogManager

# ================= 配置区域 =================
# 1. 路径配置
CMEKG_CSV_PATH = config.DATA_DIR / "raw/datasets/full_export.csv"  # 你的CMeKG全量节点文件
EXTRACTED_JSON_PATH = config.DATA_DIR / "processed/KG/triplets"
OUTPUT_CSV_PATH = config.DATA_DIR / "processed/KG/entity_alignment/entity_alignment_results.csv"
LOG_FILE_DIR = config.DATA_DIR / "processed/KG/entity_alignment"

# 2. 模型配置
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 或 'shibing624/text2vec-base-chinese'
BATCH_SIZE = 64  # 计算向量时的批次大小，显存/内存不够可调小

# 3. 匹配逻辑
SIMILARITY_THRESHOLD = 0.85  # 相似度阈值

# 4. 日志配置
log_manager = LogManager(LOG_FILE_DIR, "EntityAligner")
logger = log_manager.get_logger()
# ===========================================


class EntityAligner:
    def __init__(self, model_name: str, threshold: float):
        self.threshold = threshold
        logger.info(f"正在加载语义模型: {model_name} ...")
        # device='cpu' 或 'cuda'，库会自动选择
        self.model = SentenceTransformer(model_name)
        logger.info("模型加载完成。")

    def normalize_label(self, label: str) -> str:
        """清洗Label格式，去除冒号，并处理非法输入"""
        # 1. 处理空值 (NaN, None)
        if pd.isna(label) or label == "":
            return "Unknown"

        # 2. 强制转为字符串 (防止 CSV 里某些 label 被误读为数字)
        label_str = str(label)

        # 3. 去除冒号和首尾空格
        return label_str.replace(":", "").strip()

    def load_cmekg_data(self, csv_path: str) -> dict[str, pd.DataFrame]:
        """
        分块加载大型CSV，并按Label分组存储 (已增加空值容错处理)
        """
        logger.info(f"开始加载 CMeKG 数据: {csv_path}")
        if not os.path.exists(csv_path):
            logger.error(f"文件未找到: {csv_path}")
            sys.exit(1)

        grouped_data = defaultdict(list)
        try:
            chunk_count = 0
            # encoding='utf-8-sig' 可以自动处理 BOM 头，推荐加上
            for chunk in pd.read_csv(
                csv_path, chunksize=50000, encoding="utf-8-sig", on_bad_lines="skip"
            ):
                # 1. 动态寻找列名 (容错处理)
                cols = chunk.columns
                name_col = next((c for c in cols if "name" in c.lower()), "name")
                label_col = next((c for c in cols if "label" in c.lower()), "label")
                id_col = next((c for c in cols if "id" in c.lower()), "id")

                # ================= 关键修复开始 =================
                # 2. 预处理：填充空值并强转为字符串，避免 AttributeError
                chunk[label_col] = chunk[label_col].fillna("Unknown").astype(str)
                # ================= 关键修复结束 =================

                # 3. 应用清洗函数
                chunk["clean_label"] = chunk[label_col].apply(self.normalize_label)

                # 4. 筛选有效数据 (过滤掉 Unknown 的行，避免无效计算)
                # 仅保留需要的列
                subset = chunk[[id_col, name_col, "clean_label"]].rename(
                    columns={id_col: "id", name_col: "name", "clean_label": "label"}
                )

                # 过滤掉标签解析失败的数据
                subset = subset[subset["label"] != "Unknown"]

                # 5. 分组存放
                for label, group in subset.groupby("label"):
                    grouped_data[label].append(group)

                chunk_count += 1
                if chunk_count % 10 == 0:
                    logger.info(f"已处理 {chunk_count * 5}万 行数据...")

            # 合并每个Label下的DataFrame
            final_groups = {}
            total_nodes = 0
            for label, df_list in grouped_data.items():
                if df_list:  # 确保非空
                    final_groups[label] = pd.concat(df_list, ignore_index=True)
                    total_nodes += len(final_groups[label])

            logger.info(
                f"CMeKG 数据加载完毕。共 {total_nodes} 个有效节点，涉及 {len(final_groups)} 个标签。"
            )
            return final_groups

        except Exception as e:
            # 打印完整的错误堆栈，方便调试
            import traceback

            logger.error(f"读取 CSV 失败: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)

    def load_extracted_data(self, json_path: str) -> dict[str, list[str]]:
        """
        加载JSON数据（支持单文件或文件夹），提取唯一实体并按Label分组
        """
        logger.info(f"开始加载提取数据: {json_path}")

        # 1. 确定要处理的文件列表
        files_to_process = []

        if os.path.isfile(json_path):
            # 情况A: 传入的是单个文件
            files_to_process.append(json_path)
        elif os.path.isdir(json_path):
            # 情况B: 传入的是文件夹 -> 读取下面所有文件
            for file_name in os.listdir(json_path):
                full_path = os.path.join(json_path, file_name)
                # 确保是文件而不是子文件夹
                if os.path.isfile(full_path):
                    files_to_process.append(full_path)
        else:
            logger.error(f"路径不存在: {json_path}")
            return {}

        extracted_map = defaultdict(set)
        valid_files_count = 0
        total_triples = 0

        # 2. 遍历读取
        for file_path in files_to_process:
            try:
                with open(file_path, encoding="utf-8") as f:
                    # 尝试解析 JSON
                    data = json.load(f)

                    # 兼容数据可能是 list 或 dict 的情况
                    if isinstance(data, dict):
                        data = [data]  # 统一转为 list 处理

                    # 提取逻辑
                    for item in data:
                        if "extraction" in item and "triples" in item["extraction"]:
                            for t in item["extraction"]["triples"]:
                                s_label = self.normalize_label(t.get("start_label", "Unknown"))
                                e_label = self.normalize_label(t.get("end_label", "Unknown"))

                                extracted_map[s_label].add(t["start"])
                                extracted_map[e_label].add(t["end"])
                                total_triples += 1

                    valid_files_count += 1

            except json.JSONDecodeError:
                logger.warning(f"⚠️ 跳过非JSON文件或格式错误: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"❌ 读取文件出错 {os.path.basename(file_path)}: {e}")

        # 3. 结果转换
        final_map = {k: list(v) for k, v in extracted_map.items()}
        logger.info(f"数据加载完毕。成功读取文件: {valid_files_count}/{len(files_to_process)} 个。")
        logger.info(f"共提取三元组 {total_triples} 个，涉及标签: {list(final_map.keys())}")

        return final_map

    def align_vectors_faiss(self, extracted_names: list[str], cmekg_df: pd.DataFrame) -> list[dict]:
        """
        核心方法：使用 Faiss 在特定 Label 分组内进行检索
        """
        # 1. 准备 CMeKG 向量 (Database)
        db_names = cmekg_df["name"].tolist()
        # 注意：这里如果 db_names 也是百万级，model.encode 建议也加 batch_size
        logger.info(f"  正在计算 {len(db_names)} 个数据库实体的向量...")
        db_embeddings = self.model.encode(
            db_names, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True
        )

        # 2. 准备 提取实体 向量 (Query)
        query_embeddings = self.model.encode(
            extracted_names, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True
        )

        # 3. 归一化 (L2 Normalization) -> 使得内积等于余弦相似度
        faiss.normalize_L2(db_embeddings)
        faiss.normalize_L2(query_embeddings)

        # 4. 构建 Faiss 索引
        dim = db_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product
        index.add(db_embeddings)

        # 5. 检索 (Top 1)
        # D: Distances (Similarity Scores), I: Indices
        D, I = index.search(query_embeddings, 1)

        results = []
        for i, name in enumerate(extracted_names):
            score = float(D[i][0])
            match_idx = int(I[i][0])

            match_record = cmekg_df.iloc[match_idx]
            match_name = match_record["name"]
            match_id = match_record["id"]

            action = "🆕 Create New"
            if score >= self.threshold:
                action = "🔗 Link Existing"

            results.append(
                {
                    "extracted_name": name,
                    "matched_name": match_name,
                    "matched_id": match_id,
                    "score": round(score, 4),
                    "action": action,
                }
            )

        return results

    def run(self) -> None:
        # 1. 加载数据
        cmekg_groups = self.load_cmekg_data(CMEKG_CSV_PATH)
        extracted_groups = self.load_extracted_data(EXTRACTED_JSON_PATH)

        all_results = []

        # 2. 遍历提取出来的每一个 Label 组
        for label, extracted_names in extracted_groups.items():
            logger.info(f"正在处理标签组: [{label}] (共 {len(extracted_names)} 个实体)...")

            if label in cmekg_groups:
                # 命中：该标签在数据库中存在，进行 Faiss 检索
                cmekg_df = cmekg_groups[label]
                logger.info(f"  -> 数据库中 [{label}] 共有 {len(cmekg_df)} 个候选实体。")

                group_results = self.align_vectors_faiss(extracted_names, cmekg_df)

                # 追加 Label 信息
                for res in group_results:
                    res["label"] = label
                all_results.extend(group_results)

            else:
                # 未命中：数据库里根本没这个标签，直接全判定为新实体
                logger.warning(f"  -> 数据库中不存在标签 [{label}]，该组所有实体标记为新实体。")
                for name in extracted_names:
                    all_results.append(
                        {
                            "extracted_name": name,
                            "matched_name": None,
                            "matched_id": None,
                            "score": 0.0,
                            "action": "🆕 Create New",
                            "label": label,
                        }
                    )

        # 3. 导出结果
        if all_results:
            df_out = pd.DataFrame(all_results)
            # 调整列顺序
            cols = ["label", "extracted_name", "action", "score", "matched_name", "matched_id"]
            df_out = df_out[cols].sort_values(by=["label", "score"], ascending=[True, False])

            df_out.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"🎉 处理完成！结果已保存至: {OUTPUT_CSV_PATH}")

            # 简单的统计日志
            link_count = len(df_out[df_out["action"].str.contains("Link")])
            new_count = len(df_out) - link_count
            logger.info(f"统计: 🔗 链接已有节点: {link_count} | 🆕 新建节点: {new_count}")
        else:
            logger.warning("没有产生任何结果。")


if __name__ == "__main__":
    # 实例化并运行
    # 请确保此时目录下有 cmekg_nodes.csv 和 你的json文件
    aligner = EntityAligner(model_name=MODEL_NAME, threshold=SIMILARITY_THRESHOLD)
    aligner.run()
