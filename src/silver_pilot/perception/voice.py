"""
模块名称：audio
功能描述：语音感知模块，基于 DashScope API 实现语音转文字（ASR）和
         情感标签提取。原生支持情感识别（<|HAPPY|>、<|SAD|> 等），
         无需额外模型即可获取用户情绪状态，供 Supervisor 进行情绪感知型路由。

情感标签映射：
    输出的情感标签        → AgentState.user_emotion
    <|HAPPY|>                      → HAPPY
    <|SAD|>                        → SAD
    <|ANGRY|>                      → ANGRY
    <|NEUTRAL|>                    → NEUTRAL
    <|FEARFUL|>                    → FEARFUL
    <|DISGUSTED|>                  → DISGUSTED
    <|SURPRISED|>                  → SURPRISED
    无标签或解析失败                  → NEUTRAL（兜底）

技术栈：DashScope API ("qwen3-asr-flash")
"""

from pathlib import Path

from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "perception", "voice")

# ================= 默认配置 =================
DEFAULT_ASR_MODEL: str = getattr(config, "PERCEPTION_AUDIO_ASR_MODEL", "qwen3-asr-flash")


# ────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────


class AudioResult(BaseModel):
    """语音识别结果的结构化表示。"""

    content: str = Field("", description="识别文本。")
    """识别文本。"""

    emotion: str = Field(
        "NEUTRAL", description="提取的情感标签，如 'SAD', 'ANGRY'，默认 'NEUTRAL'。"
    )
    """提取的情感标签，如 'SAD', 'ANGRY'，默认 'NEUTRAL'。"""

    language: str = Field("zh", description="检测到的语种标签，如 'zh', 'en'。")
    """检测到的语种标签，如 'zh', 'en'。"""


# ────────────────────────────────────────────────────────────
# 语音处理器
# ────────────────────────────────────────────────────────────


class AudioProcessor:
    """
    基于 DashScopeAPI 的语音感知处理器。

    **核心能力**:
    1. 语音转文字（ASR）：支持中/英/日/韩/粤等 50+ 语种
    2. 情感识别：从 ASR 原生输出中解析情感标签
    3. 语种检测：自动识别输入语种

    **使用示例**::

        processor = AudioProcessor()

        result = processor.process("/path/to/audio.wav")

        print(result.content)  # "我胸口好痛"
        print(result.emotion)  # "FEARFUL"
    """

    def __init__(
        self,
        model: str = DEFAULT_ASR_MODEL,
        api_key: str = config.DASHSCOPE_API_KEY,
    ) -> None:
        self.model = model
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        logger.info(f"AudioProcessor 初始化完成 | model={self.model}")

    # ──────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────

    def process(self, file_path: str | Path) -> AudioResult:
        """
        识别本地音频文件。

        Args:
            file_path: 音频文件路径（支持 wav/mp3/pcm/opus 等格式，服务器端自动处理格式）

        Returns:
            AudioResult: 识别结果
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"音频文件不存在: {file_path}")
            return AudioResult(content="", emotion="NEUTRAL", language="zh")

        audio_file_path = "file://" + str(file_path.absolute())
        messages = [{"role": "user", "content": [{"audio": audio_file_path}]}]

        logger.info(f"开始识别音频文件: {file_path.name}")

        try:
            import dashscope

            response = dashscope.MultiModalConversation.call(
                # 新加坡/美国地域和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
                api_key=self.api_key,
                # 若使用美国地域的模型，需在模型后面加上“-us”后缀，例如qwen3-asr-flash-us
                model=self.model,
                messages=messages,
                result_format="message",
                asr_options={
                    "enable_itn": True,
                },
            )
            if response is None:
                logger.error("语音识别失败: response is None")
                return AudioResult(content="", emotion="NEUTRAL", language="zh")

            logger.info(f"语音识别完成: {file_path.name}")
            return AudioResult(
                content=response.output.choices[0].message.content[0]["text"],  # ty:ignore[unresolved-attribute]
                emotion=response.output.choices[0].message.annotations[0]["emotion"].upper(),  # ty:ignore[unresolved-attribute]
                language=response.output.choices[0].message.annotations[0]["language"],  # ty:ignore[unresolved-attribute]
            )

        except ImportError:
            logger.error("dashscope 未安装，请执行: pip install dashscope")
            return AudioResult(content="", emotion="NEUTRAL", language="zh")
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return AudioResult(content="", emotion="NEUTRAL", language="zh")
