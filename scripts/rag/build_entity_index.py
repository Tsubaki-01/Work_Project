"""
独立脚本：离线构建并更新 Neo4j 实体链接 Faiss 索引。
建议通过 cron 定时任务（如每日凌晨）执行，或者在图谱数据大批量更新后手动执行。
"""

import sys
import time

from silver_pilot.config import config
from silver_pilot.rag.retriever.entity_linker import EntityLinker

print("=== 开始执行离线实体索引构建任务 ===")
start_time = time.time()

try:
    # 1. 初始化 Linker (仅加载 SentenceTransformer 模型)
    print("正在初始化 EntityLinker...")
    linker = EntityLinker()

    # 2. 拉取图谱数据，构建 Faiss 索引并持久化到磁盘
    # 默认使用 EntityLinker 中定义的 DEFAULT_INDEX_DIR
    linker.build_and_save_index()

    cost_time = time.time() - start_time
    print(f"=== 离线实体索引构建任务成功完成！总耗时: {cost_time:.2f} 秒 ===")

except Exception as e:
    print(f"构建索引过程中发生严重异常: {e}")
    sys.exit(1)
