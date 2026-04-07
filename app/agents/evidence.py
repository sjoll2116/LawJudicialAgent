"""
庭前举证质证循环 Agent 节点
实现子图1的核心逻辑：
  - 事实查明质证 (Fact-finding Cross-examination)
  - Dual Check (语义重合度 + LLM 判定)
  - Final Call (最后通牒)
  - Summary & Lock (事实快照确认)
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import HumanMessage, AIMessage

from app.llm import chat_completion
from app.config import settings
from app.graph.state import CaseState, Claim, EvidenceStatus
from app.prompts.templates import (
    DEVILS_ADVOCATE_SYSTEM_PROMPT,
    DUAL_CHECK_SYSTEM_PROMPT,
    FINAL_CALL_TEMPLATE,
    EVIDENCE_SNAPSHOT_TEMPLATE,
)
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def devils_advocate_node(state: CaseState) -> dict:
    """
    事实查明质证节点：
    1. 加载案由插槽模板。
    2. 识别缺失的关键事实并索要证据。
    3. 结合 RAG 检索相似抗辩点进行压力测试。
    """
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graph", "case_templates.json")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
    except Exception as e:
        logger.error(f"加载案由模板失败: {e}")
        templates = {}

    current_template = templates.get(state.case_type, templates.get("general_civil_case", {}))
    required_slots = current_template.get("required_slots", {})
    
    # 识别缺失插槽
    missing_slots_info = []
    for slot_id, slot_meta in required_slots.items():
        if slot_id not in state.slots or not state.slots[slot_id]:
            missing_slots_info.append(f"- {slot_meta['desc']} (类型: {slot_meta['type']})")
    
    # RAG 检索
    retriever = HybridRetriever()
    case_summary = state.case_summary or "民商事纠纷"
    filter_dict = {"domain": {"$eq": "外商投资"}} if "外商" in case_summary else None
    
    # 模拟“对手的子弹”：仅检索对抗性板块 (attack/counter)，实现质证阶段的专家级聚焦
    evidence_filter = filter_dict.copy() if filter_dict else {}
    evidence_filter["logic_type"] = ["attack", "counter"]

    case_results = retriever.search_court_cases(
        q=case_summary + " 抗辩 免责 证据瑕疵 程序问题",
        where=evidence_filter,
        n=5,
    )
    rag_context = retriever.format_context_for_prompt({"law_articles": [], "court_cases": case_results})

    # 构造 Prompt 内容
    slots_status = json.dumps(state.slots, ensure_ascii=False)
    missing_text = "\n".join(missing_slots_info) if missing_slots_info else "基础要素已覆盖，进入深度质证阶段。"

    messages = [
        {"role": "system", "content": DEVILS_ADVOCATE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"【当前案由】: {current_template.get('name', '未知')}\n"
            f"【已锁定事实】: {slots_status}\n\n"
            f"【必须补全的缺失要素】:\n{missing_text}\n\n"
            f"【历史判例抗辩参考】:\n{rag_context}\n\n"
            "请作为事实查明质证官，针对缺失要素向用户索要详情及证据，"
            "并结合参考案例，对用户已提供的事实发起至少两项犀利的质问。"
        )},
    ]

    response_text = chat_completion(messages, temperature=0.3)

    # 尝试解析输出
    try:
        parsed = json.loads(response_text)
        defense_points = parsed.get("defense_points", [])
        # 如果解析成功，更新 missing_slots 为解析后的缺失项 (可选)
    except Exception:
        defense_points = []

    return {
        "messages": [AIMessage(content=response_text)],
        "defense_points": state.defense_points + defense_points,
        "evidence_round": state.evidence_round + 1,
        "missing_slots": [s.split(" (")[0][2:] for s in missing_slots_info],
        "rag_context": rag_context,
    }


def dual_check_node(state: CaseState) -> dict:
    """
    Dual Check 节点：
    判断用户最新回答是否引入了新的法律事实。
    """
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if len(user_messages) < 2:
        return {"phase": "evidence_exchange"}

    prev_msg = user_messages[-2].content
    curr_msg = user_messages[-1].content

    messages = [
        {"role": "system", "content": DUAL_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"【用户上一次回答】\n{prev_msg}\n\n"
            f"【用户本次回答】\n{curr_msg}"
        )},
    ]

    response_text = chat_completion(messages, temperature=0.1)

    try:
        parsed = json.loads(response_text)
        has_new_facts = parsed.get("has_new_facts", True)
        recommendation = parsed.get("recommendation", "continue")
    except json.JSONDecodeError:
        has_new_facts = True
        recommendation = "continue"

    if not has_new_facts or recommendation == "trigger_final_call":
        return {"phase": "final_call"}

    return {"phase": "evidence_exchange"}


def evidence_round_router(state: CaseState) -> str:
    """
    举证轮次路由器：决定是继续举证、触发 Final Call，还是锁定。
    """
    # 用户已确认快照 -> 进入法庭辩论
    if state.user_confirmed_snapshot:
        return "trial"

    # 证据已锁定 -> 等待用户确认
    if state.evidence_locked:
        return "await_confirmation"

    # 达到最大轮数 -> Final Call
    if state.evidence_round >= settings.system.max_evidence_rounds:
        return "final_call"

    # 来自 Dual Check 的判定
    if state.phase == "final_call":
        return "final_call"

    return "continue_evidence"


def final_call_node(state: CaseState) -> dict:
    """
    Final Call 节点：发送最后通牒，询问用户是否有新证据。
    """
    # 汇总已有证据
    evidence_items = []
    for c in state.claims:
        status_icon = "✅" if c.status == EvidenceStatus.PROVEN else "❓"
        evidence = ", ".join(c.supporting_evidence) if c.supporting_evidence else "暂无"
        evidence_items.append(f"  {status_icon} {c.description}（证据: {evidence}）")

    evidence_summary = "\n".join(evidence_items) if evidence_items else "  暂未收集到具体证据项"

    reply = FINAL_CALL_TEMPLATE.format(evidence_summary=evidence_summary)

    return {
        "messages": [AIMessage(content=reply)],
        "phase": "awaiting_final_call_response",
    }


def summary_lock_node(state: CaseState) -> dict:
    """
    Summary & Lock 节点：
    生成事实快照（庭审笔录确认书），等待用户确认。
    """
    # 已证明事实
    proven = [
        f"  ✅ {c.description}\n     证据: {', '.join(c.supporting_evidence)}"
        for c in state.claims if c.status == EvidenceStatus.PROVEN
    ]
    proven_text = "\n".join(proven) if proven else "  （暂无已完全证明的事实）"

    # 未证明事实
    unproven = [
        f"  ❓ {c.description}\n     备注: {c.notes or '缺乏充分证据'}"
        for c in state.claims if c.status == EvidenceStatus.UNPROVEN
    ]
    unproven_text = "\n".join(unproven) if unproven else "  （无）"

    # 抗辩要点
    defense_text = "\n".join([f"  ⚠️ {d}" for d in state.defense_points]) if state.defense_points else "  （暂无）"

    snapshot = EVIDENCE_SNAPSHOT_TEMPLATE.format(
        case_summary=state.case_summary or "外商投资纠纷案件",
        proven_facts=proven_text,
        unproven_facts=unproven_text,
        defense_points=defense_text,
    )

    return {
        "messages": [AIMessage(content=snapshot)],
        "evidence_snapshot": snapshot,
        "evidence_locked": True,
        "phase": "awaiting_snapshot_confirmation",
    }
