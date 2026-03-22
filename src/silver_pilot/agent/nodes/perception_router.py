"""
模块名称：perception
功能描述：感知层路由节点，负责识别用户输入的模态类型（文本/语音/图像），
         调用对应的感知服务（ASR/VLM），将多模态输入标准化为统一的文本表示，
         并提取情感标签和图像上下文注入 AgentState。

设计说明：
    - 阶段一：文本直接透传，语音/视觉预留接口
    - 阶段二：接入 ASR 和 vision API
"""

from pathlib import Path

from langchain_core.messages import HumanMessage

from silver_pilot.config import config
from silver_pilot.perception import VisionProcessor, VoiceProcessor, VoiceResult
from silver_pilot.utils import get_channel_logger

from ..state import AgentState

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "perception")

# ────────────────────────────────────────────────────────────
# 惰性初始化
# ────────────────────────────────────────────────────────────
_voice_processor: VoiceProcessor | None = None
_image_processor: VisionProcessor | None = None


def _get_voice_processor() -> VoiceProcessor:
    """获取或创建语音处理器（惰性初始化）。"""
    global _voice_processor
    if _voice_processor is None:
        _voice_processor = VoiceProcessor()
    return _voice_processor


def _get_image_processor() -> VisionProcessor:
    """获取或创建图像处理器（惰性初始化）。"""
    global _image_processor
    if _image_processor is None:
        _image_processor = VisionProcessor()
    return _image_processor


# ────────────────────────────────────────────────────────────
# 节点函数
# ────────────────────────────────────────────────────────────


def perception_router_node(state: AgentState) -> dict:
    """
    感知层路由节点：识别输入模态，标准化用户输入。

    处理逻辑：
        1. 从最新消息中提取用户输入
        2. 判断输入模态（文本/语音/图像）
        3. 调用对应感知服务处理
        4. 将标准化结果写入 State

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 input_modality、user_emotion、current_image_context、current_audio_context 的部分状态更新
    """
    messages = state["messages"]
    if not messages:
        logger.warning("消息列表为空，跳过感知处理")
        return {
            "input_modality": {"text": False, "voice": False, "image": False},
            "user_emotion": "NEUTRAL",
        }

    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        logger.warning("消息列表最后一条不是 HumanMessage，跳过感知处理")
        return {
            "input_modality": {"text": False, "voice": False, "image": False},
            "user_emotion": "NEUTRAL",
        }

    content = latest_message.content

    # ── 模态检测 ──
    logger.info("感知路由 | 开始检测模态")
    modality_info = _get_modality_info(content)
    logger.info(f"感知路由 | 模态数量={sum(1 for modality in modality_info.values() if modality)} ")
    input_modality = {
        "text": bool(modality_info["text"]),
        "audio": bool(modality_info["audio"]),
        "image": bool(modality_info["image"]),
    }
    # ── 根据模态分发处理，同时重置状态 ──
    emotion = "NEUTRAL"
    image_context = ""
    audio_context = ""
    standardized_messages = ""

    if modality_info.get("text"):
        logger.info("感知路由 | 检测到文本输入")
        standardized_messages += " ".join(modality_info["text"])
        logger.info(f"感知路由 | 文本输入：{modality_info['text'][:30]}...")

    if modality_info.get("audio"):
        logger.info("感知路由 | 检测到语音输入")
        voice_result = _process_audio(modality_info["audio"])
        standardized_messages += "\n\n" + voice_result.content
        emotion = voice_result.emotion
        audio_context = voice_result.content
        logger.info(
            f"语音转写完成 | text={voice_result.content[:30]}... | "
            f"emotion={emotion} | language={voice_result.language}"
        )

    if modality_info.get("image"):
        logger.info("感知路由 | 检测到图像输入")
        image_result = _process_image(modality_info["image"])
        image_context = image_result
        logger.info(f"图像识别完成 | context_len={len(image_context)}")

    return {
        "messages": [HumanMessage(content=standardized_messages)],
        "input_modality": input_modality,
        "user_emotion": emotion,
        "current_image_context": image_context,
        "current_audio_context": audio_context,
    }


def _get_modality_info(content: str | list[dict]) -> dict[str, list[str]]:
    """
    检测提取输入内容的模态内容。

    Args:
        content: 消息内容（字符串或多模态列表）

    Returns:
        dict: {"text": list[str], "image": list[str], "audio": list[str]}
    """
    modality_info: dict[str, list[str]] = {
        "text": [],
        "image": [],
        "audio": [],
    }
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    modality_info["text"].append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    modality_info["image"].append(item.get("image_url", ""))
                elif item.get("type") == "audio":
                    modality_info["audio"].append(item.get("audio", ""))
    else:
        modality_info["text"].append(content)
    return modality_info


# ────────────────────────────────────────────────────────────
# 感知服务接口
# ────────────────────────────────────────────────────────────


def _process_audio(audio_urls: list[str]) -> VoiceResult:
    """
    调用 ASR 处理语音输入。

    Args:
        audio_urls: 语音 URL 列表

    Returns:
        VoiceResult: 识别结果
    """
    # 强制使 audio 只保留一条
    audio_content = _get_voice_processor().process(audio_urls[-1])
    return audio_content


def _process_image(image_urls: list[str]) -> str:
    """
    调用 Qwen 处理图像输入。

    Args:
        image_urls: 图像 URL 列表

    Returns:
        str: 图像识别结果
    """
    processor = _get_image_processor()
    image_content = [processor.process(image_url) for image_url in image_urls]
    return "\n\n".join(image_content)
