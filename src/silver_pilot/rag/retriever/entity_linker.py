from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from silver_pilot.config import config
from silver_pilot.dao import Neo4jManager
from silver_pilot.utils import get_channel_logger

from .models import EntityLabel, ExtractedEntity, LinkedEntity

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "entity_linker")

# ================= 默认配置 =================
DEFAULT_MODEL = config.ENTITY_LINK_MODEL
DEFAULT_THRESHOLD = float(config.ENTITY_LINK_THRESHOLD)
# 默认索引保存路径
DEFAULT_INDEX_DIR = Path(config.DATA_DIR) / "entity_indices"


class EntityLinker:
    """
    运行时实体链接器（支持磁盘索引加载）。

    **生命周期**:
    【在线运行时】
    1. linker = EntityLinker()
    2. linker.load_index()  # 秒级加载磁盘索引
    3. linked = linker.link(extracted_entities)

    【离线定时构建】
    1. linker = EntityLinker()
    2. linker.build_and_save_index() # 耗时操作，存入磁盘
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.threshold = threshold

        logger.info(f"正在加载语义模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("语义模型加载完成")

        # 按 label 分组的索引：{label: {"index": FaissIndex, "names": list[str]}}
        self._indices: dict[str, dict[str, Any]] = {}
        self._index_built = False

    # ──────────────────────────────────────────────────
    # 离线：构建并保存索引（定时任务或手动执行）
    # ──────────────────────────────────────────────────

    def build_and_save_index(
        self, save_dir: Path = DEFAULT_INDEX_DIR, neo4j_manager: Neo4jManager | None = None
    ) -> None:
        """
        [离线方法] 从 Neo4j 拉取数据，计算向量，并持久化到本地磁盘。
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        if neo4j_manager is None:
            neo4j_manager = Neo4jManager()

        logger.info(f"开始离线构建实体链接索引，保存路径: {save_dir} ...")

        schema = neo4j_manager.extract_schema()
        labels = schema.get("labels", [])
        logger.info(f"图谱中共有 {len(labels)} 种节点标签")

        total_nodes = 0
        for label in labels:
            try:
                names = self._fetch_names_by_label(neo4j_manager, label)
                if not names:
                    continue

                # 编码为向量
                logger.info(f"  [{label}] 共 {len(names)} 条实体，正在编码为向量...")
                embeddings = self.model.encode(names, batch_size=64, show_progress_bar=True)
                embeddings = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(embeddings)

                # 构建 Faiss 索引
                logger.info(f"  [{label}] 正在构建 Faiss 索引...")
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)

                # --- 持久化到磁盘 ---
                logger.info(f"  [{label}] 正在持久化到磁盘...")
                index_path = save_dir / f"{label}.index"
                names_path = save_dir / f"{label}_names.json"

                faiss.write_index(index, str(index_path))
                with open(names_path, "w", encoding="utf-8") as f:
                    json.dump(names, f, ensure_ascii=False)

                total_nodes += len(names)
                logger.info(f"  [{label}] {len(names)} 个实体已索引并保存")

            except Exception as e:
                logger.warning(f"构建 [{label}] 索引失败: {e}")

        logger.info(f"离线索引构建全部完成 | 总节点数={total_nodes}")

        if neo4j_manager is not None:
            neo4j_manager.close()

    @staticmethod
    def _fetch_names_by_label(manager: Neo4jManager, label: str) -> list[str]:
        """分批次从 Neo4j 中获取指定 label 下所有节点的 name 字段。"""
        names = []
        batch_size = 100000  # 每次拉取 10 万条
        skip = 0

        try:
            with manager.driver.session() as session:
                while True:
                    query = (
                        f"MATCH (n:`{label}`) WHERE n.name IS NOT NULL "
                        f"RETURN n.name AS name "
                        f"SKIP {skip} LIMIT {batch_size}"
                    )
                    result = session.run(query)

                    batch_names = [record["name"] for record in result if record["name"]]
                    if not batch_names:
                        break  # 拉取不到数据说明到底了，退出循环

                    names.extend(batch_names)
                    skip += batch_size

        except Exception as e:
            logger.error(f"分批查询 [{label}] 节点名称失败: {e}")

        return names

    # ──────────────────────────────────────────────────
    # 在线：从磁盘加载索引（系统启动时调用一次）
    # ──────────────────────────────────────────────────

    def load_index(self, load_dir: Path = DEFAULT_INDEX_DIR) -> None:
        """
        [在线方法] 运行时从磁盘直接加载 Faiss 索引和实体名称映射。
        """
        if not load_dir.exists():
            logger.error(f"索引目录 {load_dir} 不存在，请先执行离线构建 build_and_save_index()")
            return

        logger.info(f"开始从 {load_dir} 加载实体链接索引...")

        # 遍历目录找所有的 .json 文件推断存在的 label
        total_nodes = 0
        loaded_labels = 0

        for names_file in load_dir.glob("*_names.json"):
            label = names_file.name.replace("_names.json", "")
            index_file = load_dir / f"{label}.index"

            if not index_file.exists():
                logger.warning(
                    f"找到 [{label}] 的 names 文件，但缺失对应的 .index 文件，跳过加载。"
                )
                continue

            try:
                # 1. 加载 Names
                with open(names_file, "r", encoding="utf-8") as f:
                    names = json.load(f)

                # 2. 加载 Faiss 索引
                index = faiss.read_index(str(index_file))

                self._indices[label] = {"index": index, "names": names}
                total_nodes += len(names)
                loaded_labels += 1
                logger.debug(f"  [{label}] 加载成功，包含 {len(names)} 个实体")

            except Exception as e:
                logger.error(f"加载 [{label}] 索引失败: {e}")

        self._index_built = True
        logger.info(f"实体链接索引加载完成 | 标签数={loaded_labels} | 总节点数={total_nodes}")

    # ──────────────────────────────────────────────────
    # 运行时链接
    # ──────────────────────────────────────────────────

    def link(self, entities: list[ExtractedEntity]) -> list[LinkedEntity]:
        """
        将 NER 抽取的实体列表链接到 Neo4j 标准节点。

        Args:
            entities: NER 阶段输出的实体列表

        Returns:
            list[LinkedEntity]: 链接结果列表（与输入顺序一致）
        """
        if not self._index_built:
            logger.warning("索引尚未构建，所有实体标记为未链接")
            return [
                LinkedEntity(original_name=e.name, label=e.label, is_linked=False) for e in entities
            ]

        results: list[LinkedEntity] = []

        for entity in entities:
            linked = self._link_single(entity)
            results.append(linked)

        linked_count = sum(1 for r in results if r.is_linked)
        logger.info(f"实体链接完成 | 总数={len(results)} | 成功链接={linked_count}")
        return results

    def _link_single(self, entity: ExtractedEntity) -> LinkedEntity:
        """对单个实体执行链接。"""
        label_str = entity.label.value

        # 1. 精确匹配优先：如果实体名直接存在于索引中
        if label_str in self._indices:
            names = self._indices[label_str]["names"]
            if entity.name in names:
                return LinkedEntity(
                    original_name=entity.name,
                    label=entity.label,
                    neo4j_name=entity.name,
                    similarity_score=1.0,
                    is_linked=True,
                )

        # 2. 语义匹配：在对应 label 的索引中检索
        if label_str in self._indices:
            match = self._search_in_label(entity.name, label_str)
            if match:
                return match

        # 3. 跨 label 兜底：在所有索引中尝试（处理 label 不准确的情况）
        best_match: LinkedEntity | None = None
        best_score = 0.0

        for candidate_label in self._indices:
            if candidate_label == label_str:
                continue
            match = self._search_in_label(entity.name, candidate_label)
            if match and match.similarity_score > best_score:
                best_match = match
                best_score = match.similarity_score

        if best_match and best_score >= self.threshold:
            logger.debug(
                f"跨标签链接: '{entity.name}' ({label_str}) → "
                f"'{best_match.neo4j_name}' (实际标签: {best_match.label.value})"
            )
            return best_match

        # 4. 未命中：标记为未链接
        return LinkedEntity(
            original_name=entity.name,
            label=entity.label,
            is_linked=False,
        )

    def _search_in_label(self, query_name: str, label: str) -> LinkedEntity | None:
        """在指定 label 的 Faiss 索引中检索最相似的实体。"""
        index_data = self._indices.get(label)
        if not index_data:
            return None

        index: faiss.IndexFlatIP = index_data["index"]
        names: list[str] = index_data["names"]

        # 编码查询
        query_vec = self.model.encode([query_name], show_progress_bar=False)
        query_vec = np.array(query_vec, dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # 检索 Top-1
        distances, indices = index.search(query_vec, 1)
        score = float(distances[0][0])
        match_idx = int(indices[0][0])

        if score >= self.threshold:
            try:
                entity_label = EntityLabel(label)
            except ValueError:
                entity_label = EntityLabel.UNKNOWN

            return LinkedEntity(
                original_name=query_name,
                label=entity_label,
                neo4j_name=names[match_idx],
                similarity_score=score,
                is_linked=True,
            )

        return None
