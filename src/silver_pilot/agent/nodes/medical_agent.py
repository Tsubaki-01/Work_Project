"""
模块名称：medical
功能描述：Medical Agent 节点，负责处理医疗健康类查询。内部实现「检索-校验-生成」
         三阶段流水线：先调用 RAGPipeline 获取知识上下文，再进行忠实度自检，
         最后基于可信上下文生成面向老年人的温暖回答。

执行流程：
    1. 调用 RAGPipeline.retrieve() 获取 RetrievalContext
    2. 基于上下文生成初步回答
    3. Faithfulness Check 幻觉自检
    4. 通过 → 返回回答；未通过 → Fallback 安全提示
"""

from pathlib import Path

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from silver_pilot.config import config
from silver_pilot.prompts import prompt_manager
from silver_pilot.rag import RAGPipeline, RetrievalContext
from silver_pilot.utils import get_channel_logger

from ..llm import call_llm, call_llm_parse
from ..state import HALLUCINATION_THRESHOLD, AgentState
from .helpers import build_profile_summary, extract_latest_query

# ================= 日志 =================
LOG_FILE_DIR: Path = config.LOG_DIR / "agent"
logger = get_channel_logger(LOG_FILE_DIR, "medical_agent")

# ================= 默认配置 =================
GENERATION_MODEL: str = config.MEDICAL_AGENT_GENERATION_MODEL
FAITHFULNESS_MODEL: str = config.MEDICAL_AGENT_FAITHFULNESS_MODEL
GENERATION_PROMPT: str = "agent/medical_generate"
FAITHFULNESS_PROMPT: str = "agent/faithfulness_check"

FALLBACK_RESPONSE: str = (
    "抱歉，根据现有资料我无法确切回答这个问题。"
    "为了您的健康安全，建议您咨询专业医生或药师，"
    "他们能给您更准确的指导。"
)

# ────────────────────────────────────────────────────────────
# Pydantic 结构化输出 Schema（忠实度检测）
# ────────────────────────────────────────────────────────────


class FaithfulnessOutput(BaseModel):
    """忠实度自检的结构化输出。"""

    hallucination_score: float = Field(description="幻觉分数 0.0~1.0")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="未被参考资料支持的具体陈述"
    )
    verdict: str = Field(description="判定结果: pass / fail")


# ────────────────────────────────────────────────────────────
# Pipeline 注入
# ────────────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


def set_pipeline(pipeline: RAGPipeline) -> None:
    """
    注入已初始化的 RAGPipeline 实例。

    Args:
        pipeline: 已初始化的 RAGPipeline 实例
    """
    global _pipeline
    _pipeline = pipeline
    logger.info("RAGPipeline 实例已注入 Medical Agent")


# ────────────────────────────────────────────────────────────
# Medical Agent 节点
# ────────────────────────────────────────────────────────────


def medical_agent_node(state: AgentState) -> dict:
    """
    Medical Agent 节点：检索-校验-生成 流水线。

    Args:
        state: 当前 AgentState

    Returns:
        dict: 包含 messages、rag_context、linked_entities、 hallucination_score、sub_response 的状态更新
    """
    user_query = extract_latest_query(state)
    logger.info(f"Medical Agent 开始处理 | query={user_query[:50]}...")

    # ── 阶段 1: RAG 检索 ──
    retrieval_result = _retrieve(state, user_query)

    if not retrieval_result.context_text:
        logger.warning("RAG 检索无结果，返回 Fallback")
        return {
            "messages": [AIMessage(content=FALLBACK_RESPONSE)],
            "rag_context": "",
            "linked_entities": [],
            "hallucination_score": 0.0,
            "sub_response": state.get("sub_response", []) + [FALLBACK_RESPONSE],
        }

    rewritten_query = (
        retrieval_result.processed_query.rewritten_query
        if retrieval_result.processed_query
        else user_query
    )
    rag_context = retrieval_result.context_text
    linked_entities = [entity.to_dict() for entity in retrieval_result.linked_entities]

    # ── 阶段 2: 基于上下文生成回答 ──
    generated_answer = _generate_answer(rewritten_query, rag_context, state)

    # ── 阶段 3: 忠实度自检 ──
    hallucination_score = _check_faithfulness(rag_context, generated_answer, state)

    if hallucination_score >= HALLUCINATION_THRESHOLD:
        logger.warning(
            f"幻觉检测未通过 | score={hallucination_score:.2f} | "
            f"threshold={HALLUCINATION_THRESHOLD}"
        )
        return {
            "messages": [AIMessage(content=FALLBACK_RESPONSE)],
            "rag_context": rag_context,
            "linked_entities": linked_entities,
            "hallucination_score": hallucination_score,
            "sub_response": state.get("sub_response", []) + [FALLBACK_RESPONSE],
        }

    # ── 通过：返回生成的回答 ──
    logger.info(
        f"Medical Agent 完成 | 回答长度={len(generated_answer)} | "
        f"幻觉分数={hallucination_score:.2f}"
    )

    return {
        "messages": [AIMessage(content=generated_answer)],
        "rag_context": rag_context,
        "linked_entities": linked_entities,
        "hallucination_score": hallucination_score,
        "sub_response": state.get("sub_response", []) + [generated_answer],
    }


# ────────────────────────────────────────────────────────────
# 阶段 1: RAG 检索
# ────────────────────────────────────────────────────────────


def _retrieve(state: AgentState, user_query: str) -> RetrievalContext:
    """
    调用 RAGPipeline 执行检索。

    如果 RAGPipeline 不可用（未初始化或依赖缺失），降级为空上下文。

    Returns:
        RetrievalContext: 检索结果
    """
    if _pipeline is None:
        logger.warning("RAGPipeline 未注入，跳过检索")
        return RetrievalContext(context_text="")

    try:
        retrieval_result = _pipeline.retrieve(
            user_query=user_query,
            image_context=state.get("current_image_context", ""),
            conversation_context=state.get("conversation_summary", ""),
        )

        logger.info(f"RAG 检索完成 | stats={retrieval_result.retrieval_stats}")
        return retrieval_result

    except Exception as e:
        logger.error(f"RAG Pipeline 调用失败: {e}，返回空上下文")
        return RetrievalContext(context_text="")


# ────────────────────────────────────────────────────────────
# 阶段 2: 回答生成
# ────────────────────────────────────────────────────────────


def _generate_answer(user_query: str, rag_context: str, state: AgentState) -> str:
    """
    基于 RAG 上下文生成面向老年人的医疗回答。

    Args:
        user_query: 用户查询
        rag_context: RAG 检索到的上下文文本
        state: AgentState（用于提取用户画像）

    Returns:
        str: 生成的回答文本
    """
    # 构建用户画像摘要
    profile_summary = build_profile_summary(state.get("user_profile", {}))

    messages = prompt_manager.build_prompt(
        GENERATION_PROMPT,
        user_query=user_query,
        current_image_context=state.get("current_image_context", ""),
        rag_context=rag_context,
        user_profile_summary=profile_summary,
    )

    try:
        answer = call_llm(GENERATION_MODEL, messages) or FALLBACK_RESPONSE
        logger.debug(f"回答生成完成 | 长度={len(answer)}")
        return answer

    except Exception as e:
        logger.error(f"回答生成 LLM 调用失败: {e}")
        return FALLBACK_RESPONSE


# ────────────────────────────────────────────────────────────
# 阶段 3: 忠实度自检
# ────────────────────────────────────────────────────────────


def _check_faithfulness(rag_context: str, generated_answer: str, state: AgentState) -> float:
    """
    调用 LLM 评估生成回答的忠实度。

    Args:
        rag_context: RAG 上下文
        generated_answer: 生成的回答

    Returns:
        float: 幻觉分数（0.0~1.0），越低越好
    """
    messages = prompt_manager.build_prompt(
        FAITHFULNESS_PROMPT,
        rag_context=rag_context,
        current_image_context=state.get("current_image_context", ""),
        generated_answer=generated_answer,
    )

    try:
        parsed = call_llm_parse(FAITHFULNESS_MODEL, messages, FaithfulnessOutput, temperature=0.0)
        if parsed is None:
            logger.warning("忠实度检测解析失败，默认通过")
            return 0.0

        logger.info(
            f"忠实度检测 | score={parsed.hallucination_score:.2f} | "
            f"verdict={parsed.verdict} | unsupported={len(parsed.unsupported_claims)}"
        )
        return parsed.hallucination_score

    except Exception as e:
        logger.error(f"忠实度检测 LLM 调用失败: {e}，默认通过")
        return 0.0
