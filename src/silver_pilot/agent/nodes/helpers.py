"""
模块名称：helpers
功能描述：Agent 节点的公共辅助函数，提供消息提取、用户画像格式化等共享能力，
         消除各节点中的重复实现。
"""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from ..state import AgentState


def extract_latest_query(state: AgentState) -> str:
    """
    从消息列表中提取最新的用户查询文本。

    兼容三种 content 格式：
        - str: 纯文本消息
        - list[dict]: 多模态消息（含 text / image_url / audio 块）
        - list[str]: 多段文本消息（LangChain 允许但罕见）
    """
    sub_query = state.get("current_sub_query", "")
    if sub_query:
        return sub_query

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return content_to_text(msg)
    return ""


def build_profile_summary(profile: dict) -> str:
    """
    将用户画像字典格式化为简洁的文本摘要。

    输出示例: "慢性病: 高血压, 糖尿病；过敏史: 青霉素；当前用药: 阿司匹林"
    画像为空时返回空字符串。
    """
    if not profile:
        return ""

    parts: list[str] = []

    for key, label in [
        ("chronic_diseases", "慢性病"),
        ("allergies", "过敏史"),
    ]:
        items = profile.get(key, [])
        if items:
            parts.append(f"{label}: {', '.join(items)}")

    meds = profile.get("current_medications", [])
    if meds:
        names = [m["name"] for m in meds if isinstance(m, dict) and m.get("name")]
        if names:
            parts.append(f"当前用药: {', '.join(names)}")

    return "；".join(parts)


def content_to_text(message: AnyMessage) -> str:
    """将 AnyMessage 的任意合法类型统一转换为纯文本。"""
    content = message.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_text = ""
        for item in content:
            if isinstance(item, str):
                result_text += item
            elif isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    result_text += item.get("text", "")
                elif item_type == "audio":
                    ctx = message.additional_kwargs.get("audio_context", "")
                    if ctx:
                        result_text += "\n[语音]" + ctx
                elif item_type == "image_url":
                    ctx = message.additional_kwargs.get("image_context", "")
                    if ctx:
                        result_text += "\n[图片]" + ctx
        return result_text

    return str(content)
