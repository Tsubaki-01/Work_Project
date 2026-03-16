"""
模块名称：reranker
功能描述：检索结果重排序器，对多路召回的候选文档统一评分排序。
         支持两种后端：本地 BGE-Reranker 模型和 Qwen Rerank API。
"""

from abc import ABC, abstractmethod
from typing import Any

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

from .models import RetrievalResult

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "rag_retriever", "reranker")

# ================= 默认配置 =================
DEFAULT_RERANK_TOP_K = config.RERANK_TOP_K or 5
DEFAULT_RERANK_MODE = config.RERANK_MODE or "qwen"

# ────────────────────────────────────────────────────────────
# 抽象基类
# ────────────────────────────────────────────────────────────


class BaseReranker(ABC):
    """重排序器的基类，统一 rerank 接口。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> list[RetrievalResult]:
        """
        对检索结果重新评分并排序。

        Args:
            query: 用户原始查询（或重写后的查询）
            results: 多路召回合并后的候选结果列表
            top_k: 返回的最大结果数

        Returns:
            list[RetrievalResult]: 按 rerank_score 降序排列的结果
        """
        ...


# ────────────────────────────────────────────────────────────
# BGE-Reranker 本地模型实现
# ────────────────────────────────────────────────────────────


class BGEReranker(BaseReranker):
    """
    使用 BAAI/bge-reranker-v2-m3 本地模型进行重排序。

    所需依赖: pip install FlagEmbedding

    使用示例::

        reranker = BGEReranker()
        ranked = reranker.rerank("阿司匹林的禁忌", candidates)
    """

    DEFAULT_MODEL = config.RERANK_LOCAL_MODEL or "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        use_fp16: bool = True,
        device: str | None = None,
    ) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError("请先安装依赖: pip install FlagEmbedding") from None

        logger.info(f"正在加载 Reranker 模型: {model_name_or_path}")
        self.model = FlagReranker(model_name_or_path, use_fp16=use_fp16, device=device)
        logger.info("Reranker 模型加载完成")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        # 构建 query-document pairs
        pairs = [[query, r.content] for r in results]

        try:
            # FlagReranker.compute_score 返回 float 或 list[float]
            scores = self.model.compute_score(pairs, normalize=True)

            # 统一为列表
            if isinstance(scores, (float, int)):
                scores = [scores]

            # 写入 rerank_score
            for result, score in zip(results, scores, strict=True):
                result.rerank_score = float(score)

        except Exception as e:
            logger.error(f"Reranker 计算失败: {e}，保留原始排序")
            return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

        # 按 rerank_score 降序排列，取 Top-K

        ranked = sorted(
            results,
            key=lambda r: r.rerank_score if r.rerank_score is not None else -float("inf"),
            reverse=True,
        )[:top_k]

        logger.info(
            f"重排序完成 | 输入={len(results)} | 输出={len(ranked)} | "
            f"最高分={ranked[0].rerank_score:.4f} | 最低分={ranked[-1].rerank_score:.4f}"
        )
        return ranked


# ────────────────────────────────────────────────────────────
# Qwen Rerank API 实现
# ────────────────────────────────────────────────────────────


class QwenReranker(BaseReranker):
    """
    使用 Qwen Rerank API 进行重排序（无需本地 GPU）。

    使用示例::

        reranker = QwenReranker()
        ranked = reranker.rerank("阿司匹林的禁忌", candidates)
    """

    DEFAULT_MODEL = config.RERANK_CLOUD_MODEL or "qwen3-rerank"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        logger.info(f"QwenReranker 初始化完成 | model={self.model}")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        try:
            import dashscope
            from dashscope import TextReRank

            dashscope.api_key = self.api_key

            # 构造文档列表
            documents = [r.content for r in results]

            response = TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=False,
            )

            if response.status_code != 200:
                logger.error(f"Qwen Rerank 调用失败: {response.message}")
                return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

            # 解析结果
            ranked_results: list[RetrievalResult] = []
            for item in response.output.results:
                idx = item.index
                score = item.relevance_score
                results[idx].rerank_score = float(score)
                ranked_results.append(results[idx])

            ranked_results.sort(key=lambda r: r.final_score, reverse=True)

            logger.info(f"Qwen 重排序完成 | 输入={len(results)} | 输出={len(ranked_results)}")
            return ranked_results

        except ImportError:
            logger.error("请先安装 dashscope: pip install dashscope")
            return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Qwen Rerank 失败: {e}，回退到原始排序")
            return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


# ────────────────────────────────────────────────────────────
# 工厂函数
# ────────────────────────────────────────────────────────────


def create_reranker(backend: str = DEFAULT_RERANK_MODE, **kwargs: Any) -> BaseReranker:
    """
    根据 backend 名称创建 Reranker 实例。

    Args:
        backend: "local" (BGE-Reranker) 或 "qwen" (Qwen API)

    Returns:
        BaseReranker 实例
    """
    backend = backend.lower()
    if backend == "local":
        return BGEReranker(**kwargs)
    elif backend == "qwen":
        return QwenReranker(**kwargs)
    else:
        raise ValueError(f"不支持的 reranker backend: '{backend}'，可选值: local / qwen")
