"""
模块名称：embedder
功能描述：文本向量化模块，支持 Qwen Embedding API 和本地 BGE-M3 模型两种后端。
         上层调用方只需关心 encode() 接口，无需感知底层实现。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志与配置初始化 =================
LOG_FILE_DIR = config.LOG_DIR / "embedder"
logger = get_channel_logger(LOG_FILE_DIR, "embedder")

# ================= 需要的配置 =================
DASHSCOPE_API_KEY = config.DASHSCOPE_API_KEY
QWEN_URL = config.QWEN_URL

# ────────────────────────────────────────────────────────────
# 抽象基类
# ────────────────────────────────────────────────────────────


class BaseEmbedder(ABC):
    """所有 Embedder 的基类，统一 encode 接口。"""

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        将文本列表转为向量列表。
        :param texts: 待编码的文本列表
        :return:      对应的向量列表，顺序与输入一致
        """
        ...

    def encode_one(self, text: str) -> list[float]:
        """单条文本编码的便捷方法。"""
        return self.encode([text])[0]


# ────────────────────────────────────────────────────────────
# Qwen Embedding API 实现
# ────────────────────────────────────────────────────────────


class QwenEmbedder(BaseEmbedder):
    """
    通过 OpenAI 兼容接口调用阿里云 Qwen Embedding 模型。
    默认模型：text-embedding-v3（dim=1024，支持自定义 dim）

    所需依赖：pip install openai
    """

    DEFAULT_MODEL = "text-embedding-v3"
    DEFAULT_BATCH_SIZE = 25  # 阿里云单次最多 25 条
    DEFAULT_DIM = 1024
    QUERY_PREFIX = "Retrieve relevant passages that answer the question: "

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        dimension: int = DEFAULT_DIM,
        region: str = "cn",
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请先安装依赖：pip install openai") from None

        # api_key 优先使用传入值，其次从 config 读取
        resolved_key = api_key or DASHSCOPE_API_KEY
        if not resolved_key:
            raise ValueError(
                "Qwen API Key 未配置，请传入 api_key 参数或在 config 中设置 DASHSCOPE_API_KEY"
            )

        self.model = model
        self.batch_size = batch_size
        self.dimension = dimension
        self.client = OpenAI(api_key=resolved_key, base_url=QWEN_URL[region])

        logger.info(
            f"QwenEmbedder 初始化完成 | model={self.model} | dim={self.dimension} | batch_size={self.batch_size}"
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("encode() 收到空列表，直接返回")
            return []

        # 清洗：去除换行，空串替换为空格（API 不接受空字符串）
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]

        vectors: list[list[float]] = []
        total_batches = (len(cleaned) + self.batch_size - 1) // self.batch_size

        logger.info(f"开始编码 | 总条数={len(cleaned)} | 分批数={total_batches}")

        for i in range(0, len(cleaned), self.batch_size):
            batch = cleaned[i : i + self.batch_size]
            batch_idx = i // self.batch_size + 1
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimension,
                )
                batch_vectors = [item.embedding for item in response.data]
                vectors.extend(batch_vectors)
                logger.debug(f"批次 {batch_idx}/{total_batches} 完成 | 本批条数={len(batch)}")
            except Exception as e:
                logger.error(f"批次 {batch_idx}/{total_batches} 请求失败 | error={e}")
                raise

        logger.info(f"编码完成 | 输出向量数={len(vectors)}")
        return vectors

    def encode_query(self, text: str) -> list[float]:
        """查询时调用，自动加 query prefix。"""
        prefixed = self.QUERY_PREFIX + text.strip()
        return self.encode([prefixed])[0]


# ────────────────────────────────────────────────────────────
# 本地 BGE-M3 实现
# ────────────────────────────────────────────────────────────


class BGEM3Embedder(BaseEmbedder):
    """
    使用 FlagEmbedding 加载本地 BGE-M3 模型。
    BGE-M3 支持多语言，中文医疗场景表现优秀，dim=1024。

    所需依赖：pip install FlagEmbedding
    模型下载：BAAI/bge-m3（自动从 HuggingFace 下载，也可指定本地路径）
    """

    DEFAULT_MODEL = "BAAI/bge-m3"
    DEFAULT_BATCH_SIZE = 64
    QUERY_PREFIX = "Retrieve relevant passages that answer the question: "

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,  # None = 自动选 cuda/cpu
        use_fp16: bool = True,  # fp16 加速推理，CPU 下自动降级
        max_length: int = 8192,  # BGE-M3 最大支持 8192 tokens
    ):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError("请先安装依赖：pip install FlagEmbedding") from None

        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"正在加载 BGE-M3 模型 | path={model_name_or_path} | fp16={use_fp16}")
        self.model = BGEM3FlagModel(
            str(model_name_or_path),
            use_fp16=use_fp16,
            device=device,
        )
        logger.info("BGE-M3 模型加载完成")

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("encode() 收到空列表，直接返回")
            return []

        logger.info(f"开始编码 | 总条数={len(texts)}")
        try:
            output = self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,  # dense 向量用于语义检索
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = output["dense_vecs"].tolist()  # ty:ignore[unresolved-attribute]
            logger.info(f"编码完成 | 输出向量数={len(vectors)}")
            return vectors
        except Exception as e:
            logger.error(f"BGE-M3 编码失败 | error={e}")
            raise

    def encode_query(self, text: str) -> list[float]:
        """查询时调用，自动加 query prefix。"""
        prefixed = self.QUERY_PREFIX + text.strip()
        return self.encode([prefixed])[0]


# ────────────────────────────────────────────────────────────
# 工厂函数
# ────────────────────────────────────────────────────────────


def create_embedder(backend: str = "local", **kwargs: Any) -> BaseEmbedder:
    """
    根据 backend 名称创建 Embedder 实例。

    用法示例：
        # Qwen API（api_key 也可在 config.QWEN_API_KEY 中配置，无需显式传入）
        embedder = create_embedder("qwen")
        embedder = create_embedder("qwen", api_key="sk-...")

        # 本地 BGE-M3（自动下载或指定本地路径）
        embedder = create_embedder("local")
        embedder = create_embedder("local", model_name_or_path="/data/models/bge-m3")

    :param backend: "qwen" | "local"
    :param kwargs:  透传给对应 Embedder 的构造参数
    :return:        BaseEmbedder 实例
    """
    backend = backend.lower()
    logger.info(f"创建 Embedder | backend={backend}")

    if backend == "qwen":
        return QwenEmbedder(**kwargs)
    elif backend == "local":
        return BGEM3Embedder(**kwargs)
    else:
        raise ValueError(f"不支持的 backend: '{backend}'，可选值：qwen / local")
