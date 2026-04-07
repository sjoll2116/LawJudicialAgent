"""
Evidence stage nodes: cross-exam, dual-check, final call, snapshot lock.
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.core.structured_output import call_with_json_retry
from app.graph.state import CaseState, Claim, EvidenceStatus
from app.prompts.templates import (
    DEVILS_ADVOCATE_SYSTEM_PROMPT,
    DUAL_CHECK_SYSTEM_PROMPT,
    EVIDENCE_SNAPSHOT_TEMPLATE,
    FINAL_CALL_TEMPLATE,
)
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class InferredClaim(BaseModel):
    claim_id: str
    description: str
    supporting_evidence: list[str] = Field(default_factory=list)
    status: str = "unproven"
    notes: str = ""


class DevilsAdvocateOutput(BaseModel):
    reply_to_user: str
    defense_points: list[str] = Field(default_factory=list)
    inferred_claims: list[InferredClaim] = Field(default_factory=list)


class DualCheckOutput(BaseModel):
    has_new_facts: bool = True
    recommendation: str = "continue"
    reason: str = ""


_UNAVAILABLE_PATTERNS = [
    "不知道",
    "不清楚",
    "记不清",
    "想不起来",
    "无法提供",
    "没有",
    "无",
    "仅有",
    "只有",
    "找不到",
    "丢失",
]


def _looks_unavailable_response(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(p in t for p in _UNAVAILABLE_PATTERNS)

def _merge_claims_from_payload(state: CaseState, payload: list[dict] | None) -> list[Claim]:
    claim_map = {c.claim_id: c for c in state.claims}

    for item in payload or []:
        claim_id = str(item.get("claim_id") or item.get("description") or "").strip()
        if not claim_id:
            continue

        description = str(item.get("description") or claim_id)
        evidence = item.get("supporting_evidence") or item.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]

        status_raw = str(item.get("status") or "unproven").lower()
        status = EvidenceStatus.PROVEN if status_raw == "proven" else EvidenceStatus.UNPROVEN

        if claim_id in claim_map:
            claim = claim_map[claim_id]
            claim.description = description
            claim.status = status
            for ev in evidence:
                ev_text = str(ev).strip()
                if ev_text and ev_text not in claim.supporting_evidence:
                    claim.supporting_evidence.append(ev_text)
        else:
            claim_map[claim_id] = Claim(
                claim_id=claim_id,
                description=description,
                supporting_evidence=[str(ev).strip() for ev in evidence if str(ev).strip()],
                status=status,
                notes=str(item.get("notes") or ""),
            )

    return list(claim_map.values())


def devils_advocate_node(state: CaseState) -> dict:
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graph", "case_templates.json")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
    except Exception as e:
        logger.error("加载案由模板失败: %s", e)
        templates = {}

    current_template = templates.get(state.case_type, templates.get("general_civil_case", {}))
    required_slots = current_template.get("required_slots", {})

    # Update slot_blocked based on the user's latest response to the last requested slots.
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    latest_user_msg = user_messages[-1].content if user_messages else ""
    blocked_map = dict(state.slot_blocked)
    if state.last_requested_slots and _looks_unavailable_response(latest_user_msg):
        reason = latest_user_msg[:180]
        for slot_id in state.last_requested_slots:
            blocked_map[slot_id] = reason

    ask_count_map = dict(state.slot_query_count)

    missing_slots_info = []
    for slot_id, slot_meta in required_slots.items():
        if slot_id not in state.slots or not state.slots[slot_id]:
            missing_slots_info.append(
                {
                    "slot_id": slot_id,
                    "desc": slot_meta.get("desc", slot_id),
                    "type": slot_meta.get("type", "unknown"),
                }
            )

    # Build ask candidates:
    # 1) slot still missing
    # 2) user has not explicitly blocked it
    # 3) ask count below budget
    ask_candidates = [
        item
        for item in missing_slots_info
        if item["slot_id"] not in blocked_map
        and ask_count_map.get(item["slot_id"], 0) < settings.system.max_slot_ask_rounds
    ]
    # Ask at most 2 key slots per round.
    ask_slots = ask_candidates[:2]
    ask_slot_ids = [item["slot_id"] for item in ask_slots]

    # If there are no askable slots, stop chasing and move to final call.
    if not ask_slots:
        final_updates = final_call_node(state)
        final_updates["slot_blocked"] = blocked_map
        final_updates["last_requested_slots"] = []
        return final_updates

    retriever = HybridRetriever()
    case_summary = state.case_summary or "民商事纠纷"

    filters = {
        "logic_type": ["attack", "counter"],
    }
    if state.recommended_filters:
        if state.recommended_filters.get("elements"):
            filters["elements"] = state.recommended_filters["elements"]
        if state.recommended_filters.get("keywords"):
            filters["keywords"] = state.recommended_filters["keywords"]

    case_results = retriever.search_court_cases(
        query_text=f"{case_summary} 抗辩 免责 证据瑕疵 程序问题",
        where=filters,
        n_results=5,
    )
    evidence_source_ids = [r.doc_id for r in case_results]
    rag_context = retriever.format_context_for_prompt({"law_articles": [], "court_cases": case_results})

    slots_status = json.dumps(state.slots, ensure_ascii=False)
    missing_text = json.dumps(ask_slots, ensure_ascii=False)
    blocked_text = json.dumps(
        [{"slot_id": k, "reason": v} for k, v in blocked_map.items()],
        ensure_ascii=False,
    )

    user_prompt = (
        f"【当前案由】{current_template.get('name', '未知')}\n"
        f"【已锁定事实】{slots_status}\n"
        f"【缺失要素】{missing_text}\n"
        f"【用户明确无法提供（禁止重复追问）】{blocked_text}\n"
        f"【历史判例参考】\n{rag_context}\n\n"
        "请仅围绕【缺失要素】中的 slot_id 进行追问，不要询问其他已获取或被禁止重复追问的信息。\n"
        "请严格输出JSON对象，格式如下：\n"
        "{\n"
        '  "reply_to_user": "给用户的追问与质证文本",\n'
        '  "defense_points": ["抗辩点1", "抗辩点2"],\n'
        '  "inferred_claims": [\n'
        "    {\"claim_id\": \"xxx\", \"description\": \"xxx\", \"supporting_evidence\": [\"yyy\"], \"status\": \"proven|unproven\"}\n"
        "  ]\n"
        "}\n"
        "不要输出任何JSON以外文本。"
    )

    messages = [
        {"role": "system", "content": DEVILS_ADVOCATE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    output, _ = call_with_json_retry(
        messages=messages,
        schema_model=DevilsAdvocateOutput,
        temperature=0.3,
        retries=2,
    )
    if output is None:
        return {
            "messages": [AIMessage(content="系统未能产出结构化质证结果，请继续补充关键事实。")],
            "evidence_round": state.evidence_round + 1,
            "missing_slots": [item["desc"] for item in missing_slots_info],
            "rag_context": rag_context,
        }

    defense_points = output.defense_points
    merged_claims = _merge_claims_from_payload(
        state, [c.model_dump() for c in output.inferred_claims]
    )

    for slot_id in ask_slot_ids:
        ask_count_map[slot_id] = ask_count_map.get(slot_id, 0) + 1

    return {
        "messages": [AIMessage(content=output.reply_to_user or "请继续补充证据。")],
        "defense_points": state.defense_points + [p for p in defense_points if p],
        "claims": merged_claims,
        "evidence_round": state.evidence_round + 1,
        "evidence_source_ids": list(dict.fromkeys(state.evidence_source_ids + evidence_source_ids)),
        "slot_query_count": ask_count_map,
        "slot_blocked": blocked_map,
        "last_requested_slots": ask_slot_ids,
        "missing_slots": [item["desc"] for item in missing_slots_info],
        "rag_context": rag_context,
    }


def dual_check_node(state: CaseState) -> dict:
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if len(user_messages) < 2:
        return {"phase": "evidence_exchange"}

    prev_msg = user_messages[-2].content
    curr_msg = user_messages[-1].content

    messages = [
        {"role": "system", "content": DUAL_CHECK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"【用户上一次回答】\n{prev_msg}\n\n【用户本次回答】\n{curr_msg}",
        },
    ]

    output, _ = call_with_json_retry(
        messages=messages,
        schema_model=DualCheckOutput,
        temperature=0.1,
        retries=1,
    )
    if output is None:
        return {"phase": "evidence_exchange"}

    has_new_facts = output.has_new_facts
    recommendation = output.recommendation

    if not has_new_facts or recommendation == "trigger_final_call":
        return {"phase": "final_call"}

    return {"phase": "evidence_exchange"}


def evidence_round_router(state: CaseState) -> str:
    if state.user_confirmed_snapshot:
        return "trial"

    if state.evidence_locked:
        return "await_confirmation"

    if state.evidence_round >= settings.system.max_evidence_rounds:
        return "final_call"

    if state.phase == "final_call":
        return "final_call"

    return "continue_evidence"


def final_call_node(state: CaseState) -> dict:
    evidence_items = []
    for claim in state.claims:
        status_icon = "[已证明]" if claim.status == EvidenceStatus.PROVEN else "[未证明]"
        evidence = ", ".join(claim.supporting_evidence) if claim.supporting_evidence else "暂无"
        evidence_items.append(f"  {status_icon} {claim.description}（证据: {evidence}）")

    evidence_summary = "\n".join(evidence_items) if evidence_items else "  暂未收集到具体证据项"
    reply = FINAL_CALL_TEMPLATE.format(evidence_summary=evidence_summary)

    return {
        "messages": [AIMessage(content=reply)],
        "phase": "awaiting_final_call_response",
        "last_requested_slots": [],
    }


def summary_lock_node(state: CaseState) -> dict:
    proven = [
        f"  [已证明] {c.description}\n     证据: {', '.join(c.supporting_evidence)}"
        for c in state.claims
        if c.status == EvidenceStatus.PROVEN
    ]
    proven_text = "\n".join(proven) if proven else "  （暂无已完全证明的事实）"

    unproven = [
        f"  [未证明] {c.description}\n     备注: {c.notes or '缺乏充分证据'}"
        for c in state.claims
        if c.status == EvidenceStatus.UNPROVEN
    ]
    unproven_text = "\n".join(unproven) if unproven else "  （无）"

    defense_text = "\n".join([f"  - {d}" for d in state.defense_points]) if state.defense_points else "  （暂无）"

    snapshot = EVIDENCE_SNAPSHOT_TEMPLATE.format(
        case_summary=state.case_summary or "民商事纠纷案件",
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


def awaiting_final_call_response_node(state: CaseState) -> dict:
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    latest = user_messages[-1].content if user_messages else ""
    markers = ["确认", "没有", "无补充", "继续"]

    if any(m in latest for m in markers):
        updates = summary_lock_node(state)
        updates["last_requested_slots"] = []
        return updates

    return {
        "phase": "evidence_exchange",
        "last_requested_slots": [],
        "messages": [AIMessage(content="已收到补充信息，继续进行事实质证。")],
    }


def awaiting_snapshot_confirmation_node(state: CaseState) -> dict:
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    latest = user_messages[-1].content if user_messages else ""
    markers = ["确认", "同意", "继续开庭"]

    if any(m in latest for m in markers):
        return {
            "user_confirmed_snapshot": True,
            "phase": "trial",
            "last_requested_slots": [],
            "messages": [AIMessage(content="事实快照已确认，进入法庭辩论阶段。")],
        }

    return {
        "evidence_locked": False,
        "phase": "evidence_exchange",
        "last_requested_slots": [],
        "messages": [AIMessage(content="已取消锁定，请继续补充事实和证据。")],
    }
