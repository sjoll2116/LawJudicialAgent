"""
负责识别用户意图、收集关键要素并启动状态机。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.core.structured_output import call_with_json_retry
from app.graph.state import CaseState, Claim, EvidenceStatus, IntentType, PartyInfo
from app.llm import chat_completion
from app.prompts.templates import RECEPTION_SYSTEM_PROMPT, SIMPLE_QA_SYSTEM_PROMPT
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class ReceptionOutput(BaseModel):
    intent: str = "unclear"
    cause_of_action: str = "general_civil_case"
    core_judgment: str = ""
    reply_to_user: str = ""
    user_role: str = "unclear"
    parties: dict = Field(default_factory=dict)
    collected_slots: dict = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    risk_alerts: list[str] = Field(default_factory=list)
    recommended_filters: dict[str, list[str]] = Field(
        default_factory=lambda: {"elements": [], "keywords": []}
    )
    slot_filled: bool = False

def _infer_claims_from_slots(existing_claims: list[Claim], slots: dict) -> list[Claim]:
    claim_map = {c.claim_id: c for c in existing_claims}

    for slot_key, slot_value in (slots or {}).items():
        value_text = json.dumps(slot_value, ensure_ascii=False) if isinstance(slot_value, (dict, list)) else str(slot_value)
        status = EvidenceStatus.PROVEN if value_text.strip() else EvidenceStatus.UNPROVEN

        if slot_key in claim_map:
            claim = claim_map[slot_key]
            claim.description = f"要素[{slot_key}]：{value_text[:300]}"
            claim.status = status
            if value_text.strip() and value_text not in claim.supporting_evidence:
                claim.supporting_evidence.append(value_text[:300])
        else:
            claim_map[slot_key] = Claim(
                claim_id=slot_key,
                description=f"要素[{slot_key}]：{value_text[:300]}",
                supporting_evidence=[value_text[:300]] if value_text.strip() else [],
                status=status,
            )

    return list(claim_map.values())


def reception_node(state: CaseState) -> dict:
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"messages": [AIMessage(content="您好，请描述您当前的法律问题。")]}

    latest_user_msg = user_messages[-1].content
    retriever = HybridRetriever()

    rag_results = retriever.search_all(query_text=latest_user_msg, domain=None, n_results=5)
    rag_context = retriever.format_context_for_prompt(rag_results)

    meta_registry = retriever.get_available_metadata()
    sys_prompt = (
        RECEPTION_SYSTEM_PROMPT.replace("{{available_elements}}", json.dumps(meta_registry.get("elements", []), ensure_ascii=False))
        .replace("{{available_keywords}}", json.dumps(meta_registry.get("keywords", []), ensure_ascii=False))
        .replace("{{rag_context}}", rag_context)
    )

    parties_dict = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v.dict())
        for k, v in state.parties.items()
    }
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"用户输入：{latest_user_msg}\n\n当前已收集信息：{json.dumps(parties_dict, ensure_ascii=False)}",
        },
    ]

    output, _ = call_with_json_retry(
        messages=messages,
        schema_model=ReceptionOutput,
        temperature=0.2,
        retries=2,
    )
    if output is None:
        return {"messages": [AIMessage(content="系统暂时无法生成可用结构化结果，请换种说法后重试。")]}

    intent_str = str(output.intent)
    is_simple_qa = (intent_str == IntentType.SIMPLE_QA.value)

    updates: dict = {
        "messages": [AIMessage(content=output.reply_to_user or "已收到。请继续补充事实与证据。")] if not is_simple_qa else [],
        "rag_context": rag_context,
    }

    intent_str = str(output.intent)
    if intent_str in {e.value for e in IntentType}:
        updates["intent"] = IntentType(intent_str)

    collected_parties = output.parties or {}
    if collected_parties:
        for party_id, info_dict in collected_parties.items():
            try:
                party_info = PartyInfo(**info_dict)
            except Exception:
                continue

            if party_info.birth_date or party_info.age:
                try:
                    age_val = None
                    if party_info.birth_date:
                        birth_dt = datetime.strptime(party_info.birth_date, "%Y-%m-%d")
                        age_val = (datetime.now() - birth_dt).days // 365
                        party_info.age = str(age_val)
                    elif party_info.age and str(party_info.age).isdigit():
                        age_val = int(str(party_info.age))

                    if age_val is not None:
                        if age_val >= 18:
                            party_info.capacity_status = "完全民事行为能力"
                        elif age_val >= 8:
                            party_info.capacity_status = "限制民事行为能力"
                        else:
                            party_info.capacity_status = "无民事行为能力"
                except Exception as e:
                    logger.warning("无法计算行为能力：%s", e)

            state.parties[party_id] = party_info

        updates["parties"] = state.parties

    collected_slots = output.collected_slots or {}
    merged_slots = dict(state.slots)
    merged_slots.update(collected_slots)
    updates["slots"] = merged_slots

    if not state.case_summary:
        updates["case_summary"] = latest_user_msg[:800]
    elif collected_slots:
        updates["case_summary"] = state.case_summary + "\n[已收集要素] " + json.dumps(collected_slots, ensure_ascii=False)

    updates["user_role"] = output.user_role
    updates["risk_alerts"] = output.risk_alerts
    updates["missing_slots"] = output.missing_slots
    updates["case_type"] = output.cause_of_action or state.case_type or "general_civil_case"
    updates["recommended_filters"] = output.recommended_filters

    updated_claims = _infer_claims_from_slots(state.claims, merged_slots)
    if updated_claims:
        updates["claims"] = updated_claims

    if output.slot_filled and updates["user_role"] != "unclear":
        updates["slot_filled"] = True
        updates["phase"] = "evidence_exchange"
        updates["defense_points"] = updates.get("risk_alerts", [])

    return updates


def intent_router(state: CaseState) -> str:
    if state.intent == IntentType.SIMPLE_QA:
        return "simple_qa"
    if state.intent == IntentType.COMPLEX_CASE and state.slot_filled:
        return "evidence_exchange"
    return "continue_reception"


def simple_qa_node(state: CaseState) -> dict:
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {}

    query = user_messages[-1].content
    retriever = HybridRetriever()

    results = retriever.search_law_articles(query_text=query, n_results=3)
    context = retriever.format_context_for_prompt({"law_articles": results, "court_cases": []})

    messages = [
        {"role": "system", "content": SIMPLE_QA_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    answer = chat_completion(messages, temperature=0.2)

    return {
        "messages": [AIMessage(content=answer)],
        "simple_answer": answer,
        "rag_context": context,
    }
