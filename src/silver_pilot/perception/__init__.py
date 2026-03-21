"""
模块名称：perception
功能描述：Silver Pilot 多模态感知子包，提供文本向量化、语音识别和视觉理解能力。

核心组件：
    - Embedder: 文本向量化（Qwen API / 本地 BGE-M3）
    - VoiceProcessor: 语音转文字 + 情感识别（SenseVoice API）
    - VisionProcessor: 图像理解 + OCR 提取（Qwen-2.5-VL API）
"""

from .embedder import BaseEmbedder, create_embedder
from .vision import VisionProcessor
from .voice import VoiceProcessor, VoiceResult

__all__ = [
    # Embedder
    "create_embedder",
    "BaseEmbedder",
    # Voice
    "VoiceProcessor",
    "VoiceResult",
    # Vision
    "VisionProcessor",
]
