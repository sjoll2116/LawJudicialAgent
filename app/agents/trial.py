"""
法庭辩论与裁判 Agent 节点
实现子图2的核心逻辑：
  - 正方 Agent (原告代理)
  - 反方 Agent (被告代理)
  - 裁判 Agent (法官)
  - 文书 Agent
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage

from app.llm import chat_completion
from app.config import settings
from app.graph.state import CaseState, EvidenceStatus
from app.prompts.templates import (
    PLAINTIFF_SYSTEM_PROMPT,
    DEFENDANT_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    DOCUMENT_SYSTEM_PROMPT,
)
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def _build_fact_basis(state: CaseState) -> str:
    """构建已锁定的事实基座，供法庭辩论使用"""
    lines = ["【已锁定的案件事实与证据链】"]
    for c in state.claims:
        status = "✅已证明" if c.status == EvidenceStatus.PROVEN else "❓未证明"
        evidence = ", ".join(c.supporting_evidence) if c.supporting_evidence else "无"
        lines.append(f"  [{status}] {c.description} | 证据: {evidence}")
    lines.append(f"\n【被告可能的抗辩要点】")
    for d in state.defense_points:
        lines.append(f"  - {d}")
    return "\n".join(lines)


def plaintiff_node(state: CaseState) -> dict:
    """
    正方 Agent（原告代理）：基于锁定事实提出归责论证。
    对抗逻辑：检索历史上原告如何“进攻”。
    """
    retriever = HybridRetriever()
    results = retriever.search_all(
        query_text=state.case_summary,
        user_role=state.user_role,
        phase="evidence_exchange", # 核心：博弈对抗模式
        n_results=5
    )
    rag_context = retriever.format_context_for_prompt(results)
    fact_basis = _build_fact_basis(state)

    messages = [
        {"role": "system", "content": PLAINTIFF_SYSTEM_PROMPT},
        {"role": "user", "content": f"案件概述：{state.case_summary}\n\n{fact_basis}\n\n【RAG 参考弹药（含历史主张）】\n{rag_context}\n\n请按 IRAC 结构提出原告的法律论证。注意：如果参考弹药中包含历史上类似的进攻路径，请予以借鉴。"}
    ]
    response = chat_completion(messages, temperature=0.2, max_tokens=4096)
    return {"plaintiff_arguments": response, "rag_context": rag_context, "debate_round": state.debate_round + 1}


def defendant_node(state: CaseState) -> dict:
    """
    反方 Agent（被告代理）：基于锁定事实提出抗辩。
    对抗逻辑：检索历史上被告如何“拆招”。
    """
    retriever = HybridRetriever()
    results = retriever.search_all(
        query_text=state.case_summary + " " + state.plaintiff_arguments[:500],
        user_role=state.user_role,
        phase="evidence_exchange",
        n_results=5
    )
    rag_context = retriever.format_context_for_prompt(results)
    fact_basis = _build_fact_basis(state)

    messages = [
        {"role": "system", "content": DEFENDANT_SYSTEM_PROMPT},
        {"role": "user", "content": f"案件概述：{state.case_summary}\n\n【原官观点】\n{state.plaintiff_arguments}\n\n{fact_basis}\n\n【RAG 参考弹药（含历史抗辩）】\n{rag_context}\n\n请向原告发起反击，寻找免责事由。"}
    ]
    response = chat_completion(messages, temperature=0.2, max_tokens=4096)
    return {"defendant_arguments": response}


def debate_router(state: CaseState) -> str:
    """辩论轮次路由：控制最大辩论轮数"""
    if state.debate_round >= settings.system.max_debate_rounds:
        return "judgment"
    return "continue_debate"


def judge_node(state: CaseState) -> dict:
    """
    裁判 Agent（法官）：基于 IRAC 模式权衡正反双方论证。
    关键：二次检索，锁定裁判说理块与事实关联。
    """
    retriever = HybridRetriever()
    # 法官专属：结论性检索逻辑
    judge_results = retriever.search_all(
        query_text=state.evidence_snapshot if state.evidence_snapshot else state.case_summary,
        user_role=state.user_role,
        phase="conclusion", # 核心：说理锁定模式
        n_results=3
    )
    judge_rag = retriever.format_context_for_prompt(judge_results)
    fact_basis = _build_fact_basis(state)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"案件基础事实：{fact_basis}\n\n"
            f"【原告论证】\n{state.plaintiff_arguments}\n\n"
            f"【被告抗辩】\n{state.defendant_arguments}\n\n"
            f"【最高法/地方法官裁判说理参考】\n{judge_rag}\n\n"
            "请主持裁判。你的理由必须与检索到的裁判逻辑保持法律原则的一致性。"
        )}
    ]
    response = chat_completion(messages, temperature=0.1, max_tokens=6000)
    return {"judgment": response, "phase": "document"}


def document_node(state: CaseState) -> dict:
    """
    文书 Agent：将裁判结果转化为标准法律文书。
    """
    messages = [
        {"role": "system", "content": DOCUMENT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"案件信息：{state.case_summary}\n"
            f"当事人：{json.dumps(state.parties, ensure_ascii=False)}\n\n"
            f"裁判推理结果：\n{state.judgment}\n\n"
            "请将以上裁判推理结果转化为标准格式的民事裁判文书初稿。"
        )},
    ]

    response = chat_completion(messages, temperature=0.2, max_tokens=8000)

    return {
        "final_document": response,
        "messages": [AIMessage(content=f"📄 裁判文书已生成：\n\n{response}")],
        "phase": "completed",
    }
