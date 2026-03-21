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
from silver_pilot.utils import get_channel_logger

from ..state import AgentState

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "perception")


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
        dict: 包含 input_modality、user_emotion、current_image_context 的部分状态更新
    """
    messages = state["messages"]
    if not messages:
        logger.warning("消息列表为空，跳过感知处理")
        return {"input_modality": "text", "user_emotion": "NEUTRAL"}

    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return {}

    content = latest_message.content

    # ── 模态检测 ──
    modality = _detect_modality(content)

    logger.info(f"感知路由 | 模态={modality} | 内容前30字={str(content)[:30]}...")

    # ── 根据模态分发处理，同时重置状态 ──
    emotion = "NEUTRAL"
    image_context = ""

    if modality == "text":
        # 文本直接透传，无需额外处理
        pass

    elif modality == "voice":
        # 阶段二：调用 SenseVoice ASR
        # asr_result = _process_voice(content)
        # emotion = asr_result.get("emotion", "NEUTRAL")
        logger.info("语音模态检测到，当前阶段按文本处理（预留 ASR 接口）")

    elif modality == "image":
        # 阶段二：调用 Qwen-VL
        # vlm_result = _process_image(content)
        # image_context = vlm_result.get("description", "")
        logger.info("图像模态检测到，当前阶段按文本处理（预留 VLM 接口）")

    return {
        "input_modality": modality,
        "user_emotion": emotion,
        "current_image_context": image_context,
    }


def _detect_modality(content: str | list) -> str:
    """
    检测输入内容的模态类型。

    Args:
        content: 消息内容（字符串或多模态列表）

    Returns:
        str: "text" / "voice" / "image" / "multimodal"
    """
    # LangChain 多模态消息的 content 可能是列表
    if isinstance(content, list):
        has_image = any(
            isinstance(item, dict) and item.get("type") == "image_url" for item in content
        )
        has_audio = any(isinstance(item, dict) and item.get("type") == "audio" for item in content)
        if has_image and has_audio:
            return "multimodal"
        if has_image:
            return "image"
        if has_audio:
            return "voice"

    return "text"


# ────────────────────────────────────────────────────────────
# 预留感知服务接口（阶段二实现）
# ────────────────────────────────────────────────────────────


def _process_voice(audio_content: str) -> dict:
    """
    [预留] 调用 SenseVoice ASR 处理语音输入。

    Returns:
        dict: {"text": "识别文本", "emotion": "SAD", "language": "zh"}
    """
    # TODO: 阶段二接入 DashScope SenseVoice API
    return {"text": audio_content, "emotion": "NEUTRAL", "language": "zh"}


def _process_image(image_content: str) -> dict:
    """
    [预留] 调用 Qwen-VL 处理图像输入。

    Returns:
        dict: {"description": "药品说明书照片", "ocr_text": "阿司匹林 100mg..."}
    """
    # TODO: 阶段二接入 DashScope Qwen-VL API
    return {"description": "", "ocr_text": ""}
