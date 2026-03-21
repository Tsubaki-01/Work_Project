"""
模块名称：vision
功能描述：视觉感知模块，基于 DashScope Qwen-2.5-VL API 实现图像理解和 OCR 文字提取。
         Qwen-VL 采用动态分辨率机制，能有效保留药瓶小字、说明书表格等细节信息，
         OCR 结果注入 AgentState.current_image_context 供下游 RAG 检索使用。

核心能力：
    1. 药品/说明书 OCR：提取药品名称、规格、用法用量等关键字段
    2. 图像描述：生成面向老年人的简洁场景描述
    3. 医疗器械识别：识别血压计读数、血糖仪数值等

技术栈：Qwen-2.5-VL API（DashScope OpenAI 兼容接口）
"""

from pathlib import Path

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "perception", "vision")

# ================= 默认配置 =================
DEFAULT_VL_MODEL: str = getattr(config, "VISION_UNDERSTANDING_MODEL", "qwen-vl-max")

# ────────────────────────────────────────────────────────────
# 系统 Prompt
# ────────────────────────────────────────────────────────────

DEFAULT_PROMPT: str = """
你是一个专业的医疗图像识别助手，擅长从药品包装、说明书、处方单等图像中提取关键信息。

### 任务
用简洁的中文描述这张图片的内容。
如果图片中包含文字，请提取关键文字信息。
如果图片中包含问题，请简单回答问题。

### 输出要求
1. **药品相关图片**：提取药品名称、规格、用法用量、生产厂家、批准文号、有效期等字段
2. **医疗器械读数**：提取血压值、血糖值、体温等数值及单位
3. **处方/报告**：提取诊断结果、用药建议、检查指标等
4. **其他图片**：简洁描述图片内容

###
- 一句话总结图片的核心内容
- 提取的文字信息
- 如果图片中包含问题，请简单回答问题
"""


# ────────────────────────────────────────────────────────────
# 视觉处理器
# ────────────────────────────────────────────────────────────


class VisionProcessor:
    """
    基于 Qwen API 的视觉感知处理器。

    **核心特性**:
    - 多场景适配：药品 OCR、器械读数、报告提取、通用描述
    - 通过 DashScope 统一调用

    **使用示例**::

        processor = VisionProcessor()

        # 从文件路径识别
        result = processor.process("/path/to/drug_photo.jpg")

        print(result)
    """

    def __init__(
        self,
        model: str = DEFAULT_VL_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        logger.info(f"VisionProcessor 初始化完成 | model={self.model}")

    # ──────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────

    def process(
        self,
        file_path: str | Path,
        *,
        prompt: str = "",
    ) -> str:
        """
        识别本地图像文件。

        Args:
            file_path: 图像文件路径（支持 jpg/png/webp/bmp 等格式）
            prompt: 自定义提示词，为空则使用默认药品 OCR Prompt

        Returns:
            VisionResult: 视觉识别结果
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"图片文件不存在: {file_path}")
            return ""

        image_file_path = "file://" + str(file_path.absolute())
        messages = [
            {"role": "user", "content": [{"image": image_file_path}, {"text": DEFAULT_PROMPT}]}
        ]

        logger.info(f"开始识别图像文件: {file_path.name}")

        try:
            import dashscope

            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
            )
            if response is None:
                logger.error("视觉识别失败: response is None")
                return ""
            logger.info(f"图像文件识别完成: {file_path.name}")
            return response.output.choices[0].message.content[0]["text"]  # ty:ignore[unresolved-attribute]

        except Exception as e:
            logger.error(f"视觉识别失败: {e}")
            return ""
