import re
from typing import Any

import pandas as pd
from neo4j import GraphDatabase

# ================= 导入 =================
from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志与配置初始化 =================
LOG_FILE_DIR = config.LOG_DIR / "neo4j_logs"
logger = get_channel_logger(LOG_FILE_DIR, "Neo4jManager")
# ===================================================


class Neo4jManager:
    """
    Neo4j 知识图谱 ETL 与 CRUD 核心管理器。
    用于在项目软件中实现对图数据库中节点、关系的增删改查操作。
    """

    def __init__(self, uri: str | None = None, auth: tuple[str, str] | None = None) -> None:
        """
        初始化 Neo4j 驱动连接。
        默认使用 silver_pilot.config 中的全局配置。

        Args:
            uri (Optional[str]): Neo4j 数据库连接 URI。为空则读取 config.NEO4J_URI。
            auth (Optional[tuple[str, str]]): 用户名和密码。为空则读取 config.NEO4J_USER 等。
        """
        # 严格使用配置项作为 fallback
        final_uri = uri or config.NEO4J_URI
        final_auth = auth or (config.NEO4J_USER, config.NEO4J_PASSWORD)

        self.driver = GraphDatabase.driver(final_uri, auth=final_auth)
        logger.info("Neo4j 驱动初始化成功。")

    def close(self) -> None:
        """关闭数据库连接"""
        self.driver.close()
        logger.info("Neo4j 连接已关闭。")

    @staticmethod
    def normalize_label(label: Any) -> str:
        """
        清洗和规范化 Neo4j 节点标签 (Label)。
        增强版逻辑：处理空值类型、字符串字面量空值、特殊字符清理、空白符替换及长度截断。

        Args:
            label (Any): 原始标签名

        Returns:
            str: 清洗后的标准标签名，默认 fallback 为 "Unknown"
        """
        # 1. 基础空值及类型防御判断
        if pd.isna(label) or label is None:
            return "Unknown"

        # 2. 强制转换为字符串并去除首尾空白
        label_str = str(label).strip()

        # 3. 处理字符串形式的常见无效/空值表示 (忽略大小写)
        if label_str.lower() in ("", "nan", "null", "none", "undefined", "unknown"):
            return "Unknown"

        # 4. 语义分隔符处理
        # 将空格、连字符(-)、斜杠(/、\) 统一替换为下划线，保留词汇间的界限
        # 例如: "Disease Type" -> "Disease_Type", "Covid-19" -> "Covid_19"
        label_str = re.sub(r"[\s\-/\\]+", "_", label_str)

        # 5. 移除非法字符 (过滤掉冒号、引号、括号等 Cypher 敏感字符)
        # 正则含义: 保留 字母、数字、下划线 (\w) 以及 中文字符 (\u4e00-\u9fa5)
        label_str = re.sub(r"[^\w\u4e00-\u9fa5]", "", label_str)

        # 6. 整理下划线
        # 将多个连续下划线合并为一个，并去除首尾可能残留的下划线
        label_str = re.sub(r"_+", "_", label_str).strip("_")

        # 7. 二次防空验证 (防止原始字符串全是非法字符，被正则清空)
        if not label_str:
            return "Unknown"

        # 8. 长度截断保护 (Neo4j 最佳实践中，Label 不宜过长，此处设置硬性上限为 50 字符)
        max_length = 50
        if len(label_str) > max_length:
            label_str = label_str[:max_length]
            # 截断后可能尾部又出现了下划线，再次清理
            label_str = label_str.strip("_")

        return label_str

    # ==========================================
    #                 CREATE / LOAD (增)
    # ==========================================

    def batch_import_triplets(self, triplets: list[dict[str, Any]]) -> None:
        """
        批量导入三元组数据 (ETL Load 核心逻辑)。
        按关系类型及节点标签进行分组，利用 MERGE 语法避免重复创建。

        Args:
            triplets (list[dict[str, Any]]): 包含清洗后三元组信息的字典列表。
                预期格式:
                {
                    "start_name": str, "start_label": str,
                    "end_name": str, "end_label": str,
                    "rel_type": str, "source": str
                }
        """
        if not triplets:
            logger.warning("传入的三元组列表为空，取消导入操作。")
            return

        # 分组逻辑：(rel_type, start_label, end_label)
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for t in triplets:
            start_label = self.normalize_label(t.get("start_label", "Unknown"))
            end_label = self.normalize_label(t.get("end_label", "Unknown"))
            rel_type = str(t.get("rel_type", "RELATED_TO")).strip()

            key = (rel_type, start_label, end_label)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(t)

        total_batches = len(grouped)
        current_batch = 0

        for (rel_type, start_label, end_label), batch in grouped.items():
            current_batch += 1
            logger.info(
                f"[{current_batch}/{total_batches}] "
                f"正在写入: ({start_label})-[{rel_type}]->({end_label}) 共 {len(batch)} 条..."
            )

            try:
                with self.driver.session() as session:
                    query = f"""
                    UNWIND $batch AS row

                    // 1. 合并头实体
                    MERGE (s:`{start_label}` {{name: row.start_name}})
                    ON CREATE SET s.source = row.source, s.created_at = timestamp()

                    // 2. 合并尾实体
                    MERGE (e:`{end_label}` {{name: row.end_name}})
                    ON CREATE SET e.source = row.source, e.created_at = timestamp()

                    // 3. 合并关系
                    MERGE (s)-[r:`{rel_type}`]->(e)
                    ON CREATE SET r.source = row.source, r.confidence = 'High', r.created_at = timestamp()
                    """
                    session.run(query, batch=batch)
            except Exception as e:
                logger.error(f"❌ 写入 [{rel_type}] 批次时发生异常: {e}")

    # ==========================================
    #                 READ / EXTRACT (查)
    # ==========================================

    def get_node_by_name(self, label: str, name: str) -> dict[str, Any] | None:
        """
        根据标签和名称精确查询节点属性。

        Args:
            label (str): 节点标签
            name (str): 实体名称

        Returns:
            Optional[dict[str, Any]]: 节点的属性字典，如果不存在则返回 None
        """
        clean_label = self.normalize_label(label)
        query = f"MATCH (n:`{clean_label}` {{name: $name}}) RETURN properties(n) AS props"

        try:
            with self.driver.session() as session:
                result = session.run(query, name=name)
                record = result.single()
                if record:
                    return dict(record["props"])
        except Exception as e:
            logger.error(f"❌ 查询节点 [{clean_label}: {name}] 失败: {e}")
        return None

    def extract_schema(self) -> dict[str, list[str]]:
        """
        动态提取图数据库的 Schema (标签和关系类型)。

        Returns:
            dict[str, list[str]]: 包含 "labels" 和 "types" 的字典
        """
        schema: dict[str, list[str]] = {"labels": [], "types": []}

        try:
            with self.driver.session() as session:
                labels_res = session.run("CALL db.labels()")
                schema["labels"] = [record[0] for record in labels_res]

                types_res = session.run("CALL db.relationshipTypes()")
                schema["types"] = [record[0] for record in types_res]
        except Exception as e:
            logger.error(f"❌ 提取 Schema 失败: {e}")

        return schema

    # ==========================================
    #                 UPDATE (改)
    # ==========================================

    def update_node_properties(self, label: str, name: str, properties: dict[str, Any]) -> bool:
        """
        更新指定节点的属性。

        Args:
            label (str): 节点标签
            name (str): 实体名称
            properties (dict[str, Any]): 需要更新的属性字典

        Returns:
            bool: 是否更新成功
        """
        clean_label = self.normalize_label(label)

        query = f"""
        MATCH (n:`{clean_label}` {{name: $name}})
        SET n += $properties, n.updated_at = timestamp()
        RETURN count(n) AS updated_count
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, name=name, properties=properties)
                record = result.single()
                if record and record["updated_count"] > 0:
                    logger.info(f"✅ 成功更新节点 [{clean_label}: {name}] 的属性。")
                    return True
        except Exception as e:
            logger.error(f"❌ 更新节点 [{clean_label}: {name}] 失败: {e}")

        logger.warning(f"⚠️ 更新未生效，未找到节点 [{clean_label}: {name}]。")
        return False

    # ==========================================
    #                 DELETE (删)
    # ==========================================

    def delete_node_and_relationships(self, label: str, name: str) -> bool:
        """
        级联删除指定的节点及其所有关联关系。

        Args:
            label (str): 节点标签
            name (str): 实体名称

        Returns:
            bool: 是否删除成功
        """
        clean_label = self.normalize_label(label)
        query = f"""
        MATCH (n:`{clean_label}` {{name: $name}})
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, name=name)
                record = result.single()
                if record and record["deleted_count"] > 0:
                    logger.info(f"✅ 成功删除节点 [{clean_label}: {name}] 及其关联关系。")
                    return True
        except Exception as e:
            logger.error(f"❌ 删除节点 [{clean_label}: {name}] 失败: {e}")

        logger.warning(f"⚠️ 删除未生效，未找到节点 [{clean_label}: {name}]。")
        return False


if __name__ == "__main__":
    from unittest.mock import MagicMock, patch

    print("=" * 50)
    print("1. 测试 normalize_label (静态方法，无数据库交互)")
    print("=" * 50)

    test_cases = [
        ("正常标签", "Disease"),
        ("带前后空格", "  Symptom  "),
        ("防注入", "A:B'C\"D"),
        ("语义保留", "Medical Test-Type 1"),
        ("伪装的空值", "  null "),
        ("Pandas 缺失值", pd.NA),
    ]

    for desc, raw_label in test_cases:
        cleaned = Neo4jManager.normalize_label(raw_label)
        print(f"[{desc:^15}] 原始值: {repr(raw_label):<25} -> 清洗后: {repr(cleaned)}")

    print("\n" + "=" * 50)
    print("2. 测试 Neo4j CRUD 逻辑")
    print("=" * 50)

    # 使用 patch 拦截 GraphDatabase.driver，防止真实连接数据库
    with patch("neo4j.GraphDatabase.driver") as MockDriver:
        # 创建一个假的 session 和事务返回结果
        mock_session = MagicMock()
        MockDriver.return_value.session.return_value.__enter__.return_value = mock_session

        # 实例化 Manager (此时内部的 self.driver 是一个 Mock 对象)
        # 传入虚拟的 uri 防止读取真实 config 去尝试连接
        manager = Neo4jManager(uri="bolt://fake-mock-db:7687", auth=("fake", "fake"))

        # --- 1. 测试增 (Mock) ---
        mock_triplets = [
            {
                "start_name": "阿司匹林",
                "start_label": "Drug:!@",
                "end_name": "发热",
                "end_label": "Symptom",
                "rel_type": "TREATS",
                "source": "test",
            }
        ]
        print("执行插入 (Mock)...")
        manager.batch_import_triplets(mock_triplets)
        # 验证 mock_session.run 是否被调用，这说明我们的代码逻辑顺利走到了发送 Cypher 这一步
        if mock_session.run.called:
            print("✅ 拦截成功！Cypher 写入语句已生成并发送给 Mock Driver，未触碰真实 DB。")

        # --- 2. 测试查 (Mock) ---
        print("\n执行查询 (Mock)...")
        # 模拟数据库返回了一个包含 props 的记录
        mock_record = {"props": {"name": "阿司匹林", "price": 10.5}}
        mock_session.run.return_value.single.return_value = mock_record

        node_info = manager.get_node_by_name("Drug", "阿司匹林")
        print(f"✅ 模拟查到的节点属性: {node_info}")

        # --- 3. 测试改 (Mock) ---
        print("\n执行更新 (Mock)...")
        # 模拟数据库返回受影响的行数为 1
        mock_session.run.return_value.single.return_value = {"updated_count": 1}
        update_status = manager.update_node_properties("Drug", "阿司匹林", {"price": 15.5})
        print(f"✅ 模拟更新状态: {update_status}")

        # --- 4. 测试删 (Mock) ---
        print("\n执行删除 (Mock)...")
        # 模拟数据库返回删除的行数为 1
        mock_session.run.return_value.single.return_value = {"deleted_count": 1}
        delete_status = manager.delete_node_and_relationships("Drug", "阿司匹林")
        print(f"✅ 模拟删除状态: {delete_status}")

        manager.close()
        print("\n🎉 Mock 测试全部完成，您的真实数据库毫发无损！")
