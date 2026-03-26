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
    有sub_query：子agent，直接返回sub_query
    无sub_query：supervisor，从消息列表中提取最新的用户查询文本

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
            return message_to_text(msg)
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


def message_to_text(message: AnyMessage) -> str:
    """将 AnyMessage 的任意合法类型统一转换为纯文本。"""
    content = message.content
    if isinstance(content, str):
        # 纯文本消息在经过percep重写后也可能在 additional_kwargs 中携带多模态上下文
        text = content
        extra_parts: list[str] = []
        audio_ctx = message.additional_kwargs.get("audio_context", "")
        if audio_ctx:
            extra_parts.append("[语音]" + audio_ctx)
        image_ctx = message.additional_kwargs.get("image_context", "")
        if image_ctx:
            extra_parts.append("[图片]" + image_ctx)
        if extra_parts:
            if text and not text.endswith("\n"):
                text += "\n"
            text += "\n".join(extra_parts)
        return text

    if isinstance(content, list):
        result_text = ""
        for item in content:
            if isinstance(item, str):
                result_text += item + "\n"
            elif isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    result_text += item.get("text", "") + "\n"
                elif item_type == "audio_url":
                    ctx = message.additional_kwargs.get("audio_context", "")
                    if ctx:
                        result_text += "[语音]" + ctx + "\n"
                elif item_type == "image_url":
                    ctx = message.additional_kwargs.get("image_context", "")
                    if ctx:
                        result_text += "[图片]" + ctx + "\n"
        return result_text

    return str(content)


def messages_to_text(messages: list[AnyMessage]) -> str:
    """将 LangChain Message 列表格式化为纯文本。"""
    role_map: dict[type[AnyMessage], str] = {
        HumanMessage: "用户",
        AIMessage: "助手",
        SystemMessage: "系统",
    }
    lines: list[str] = []
    for msg in messages:
        role = role_map.get(type(msg), "未知")
        lines.append(f"{role}: {message_to_text(msg)}")
    return "\n\n".join(lines)


def get_conversation_context(messages: list[AnyMessage], max_turns: int = 6) -> str:
    """
    提取本轮对话之前的历史消息作为上下文。

    从后向前找到最后一条 HumanMessage（即本轮用户输入），
    取它之前的 max_turns 条消息作为历史上下文。
    这样既不包含本轮用户输入，也不包含本轮其他 agent 的回答。

    时间线示意:
        [历史消息...] [本轮 HumanMessage] [本轮 agent 回答...]
        ^^^^^^^^^^^^
        只取这部分
    """
    # 找到最后一条 HumanMessage 的位置
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx <= 0:
        return ""

    # 取它之前的 max_turns 条
    history = messages[max(0, last_human_idx - max_turns) : last_human_idx]
    return messages_to_text(history)


def extract_ai_messages_after_last_human(
    messages: list[AnyMessage],
) -> tuple[str, list[AIMessage]]:
    """
    从消息列表中提取最后一条 HumanMessage 之后的所有 AIMessage 内容。

    Returns:
        tuple[str, list[AIMessage]]: (用户原始查询, AI 回复文本列表)
    """
    user_query = ""
    ai_contents: list[AIMessage] = []

    # 从后向前找到最后一条 HumanMessage
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            user_query = message_to_text(messages[i])
            break

    if last_human_idx < 0:
        return "", []

    # 收集该 HumanMessage 之后的所有 AIMessage
    for msg in messages[last_human_idx + 1 :]:
        if isinstance(msg, AIMessage):
            ai_contents.append(msg)

    return user_query, ai_contents
