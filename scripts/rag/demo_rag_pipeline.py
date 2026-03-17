"""
模块名称：demo_rag_pipeline
功能描述：RAG 检索流水线的端到端使用示例。
         展示如何初始化流水线、执行查询，以及如何访问各阶段中间结果。

使用方式：
    python scripts/rag/demo_rag_pipeline.py
"""

from silver_pilot.rag.retriever import PipelineConfig, RAGPipeline


def main() -> None:
    # ============================================================
    # 1. 配置流水线参数
    # ============================================================
    pipeline_config = PipelineConfig(
        # 查询处理：使用 qwen3.5-flash 做重写和 NER
        query_model="qwen3.5-flash",
        # 实体链接：语义匹配阈值
        linking_threshold=0.85,
        # 向量检索：使用本地 embedding 模型
        embedding_backend="local",
        vector_top_k=5,
        qa_enabled=True,
        kb_enabled=True,
        # 重排序：使用 Qwen API 或本地模型
        reranker_backend="local",
        rerank_top_k=5,
        # 上下文构建：直接拼接模式（快速）
        context_mode="direct",
        max_context_chars=3000,
    )

    # ============================================================
    # 2. 初始化流水线（启动时调用一次）
    # ============================================================
    pipeline = RAGPipeline(pipeline_config)
    pipeline.initialize()  # 加载模型、构建 Faiss 索引等

    # ============================================================
    # 3. 执行查询
    # ============================================================
    test_queries = [
        "阿司匹林和华法林能一起吃吗？",
        "我血糖高，最近眼睛看不清楚了，该怎么办？",
        "帮我看看这个药一天吃几次",
    ]

    for idx, query in enumerate(test_queries):
        print("\n" + "=" * 60)
        print(f"用户查询: {query}")
        print("=" * 60)

        # 执行检索
        result = pipeline.retrieve(
            user_query=query,
            image_context="药品名称: 阿司匹林肠溶片，规格: 100mg，用法用量: 一日一次"
            if idx == 2
            else "",
        )

        # 查看查询处理结果
        pq = result.processed_query
        if pq:
            print(f"\n重写后查询: {pq.rewritten_query}")
            if pq.sub_queries:
                print(f"子查询: {pq.sub_queries}")
            if pq.entities:
                print(f"抽取实体: {[(e.name, e.label.value) for e in pq.entities]}")

        # 查看实体链接结果
        if result.linked_entities:
            for le in result.linked_entities:
                status = (
                    f"✅ → {le.neo4j_name} ({le.similarity_score:.2f})"
                    if le.is_linked
                    else "❌ 未链接"
                )
                print(f"实体链接: {le.original_name} {status}")

        # 查看检索统计
        print(f"\n检索统计: {result.retrieval_stats}")

        # 查看最终 context
        print(f"\n最终上下文 ({len(result.context_text)} 字符):")
        print("-" * 40)
        print(
            result.context_text[:500] + "..."
            if len(result.context_text) > 500
            else result.context_text
        )

    # ============================================================
    # 4. 清理资源
    # ============================================================
    pipeline.close()


if __name__ == "__main__":
    main()
