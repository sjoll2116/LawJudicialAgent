"""
智能体工作流定义
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.agents.evidence import (
    awaiting_final_call_response_node,
    awaiting_snapshot_confirmation_node,
    devils_advocate_node,
    dual_check_node,
    evidence_round_router,
    final_call_node,
    summary_lock_node,
)
from app.agents.reception import intent_router, reception_node, simple_qa_node
from app.agents.trial import debate_router, defendant_node, document_node, judge_node, plaintiff_node
from app.graph.state import CaseState

logger = logging.getLogger(__name__)


def phase_router(_: CaseState) -> dict:
    return {}


def entry_phase_route(state: CaseState) -> str:
    if state.phase == "awaiting_snapshot_confirmation":
        return "awaiting_snapshot_confirmation"
    if state.phase == "awaiting_final_call_response":
        return "awaiting_final_call_response"
    if state.phase == "evidence_exchange" and state.evidence_round > 0:
        return "dual_check"
    if state.phase == "trial":
        return "plaintiff"
    return "reception"


def final_call_response_router(state: CaseState) -> str:
    if state.phase == "evidence_exchange":
        return "devils_advocate"
    return "end"


def snapshot_response_router(state: CaseState) -> str:
    if state.phase == "trial":
        return "plaintiff"
    if state.phase == "evidence_exchange":
        return "devils_advocate"
    return "end"


def build_graph() -> StateGraph:
    workflow = StateGraph(CaseState)

    workflow.add_node("phase_router", phase_router)

    workflow.add_node("reception", reception_node)
    workflow.add_node("simple_qa", simple_qa_node)

    workflow.add_node("devils_advocate", devils_advocate_node)
    workflow.add_node("dual_check", dual_check_node)
    workflow.add_node("final_call", final_call_node)
    workflow.add_node("summary_lock", summary_lock_node)
    workflow.add_node("awaiting_final_call_response", awaiting_final_call_response_node)
    workflow.add_node("awaiting_snapshot_confirmation", awaiting_snapshot_confirmation_node)

    workflow.add_node("plaintiff", plaintiff_node)
    workflow.add_node("defendant", defendant_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("document", document_node)

    workflow.set_entry_point("phase_router")

    workflow.add_conditional_edges(
        "phase_router",
        entry_phase_route,
        {
            "reception": "reception",
            "dual_check": "dual_check",
            "awaiting_final_call_response": "awaiting_final_call_response",
            "awaiting_snapshot_confirmation": "awaiting_snapshot_confirmation",
            "plaintiff": "plaintiff",
        },
    )

    workflow.add_conditional_edges(
        "reception",
        intent_router,
        {
            "simple_qa": "simple_qa",
            "evidence_exchange": "devils_advocate",
            "continue_reception": END,
        },
    )

    workflow.add_edge("simple_qa", END)
    workflow.add_edge("devils_advocate", END)

    workflow.add_conditional_edges(
        "dual_check",
        evidence_round_router,
        {
            "continue_evidence": "devils_advocate",
            "final_call": "final_call",
            "await_confirmation": END,
            "trial": "plaintiff",
        },
    )

    workflow.add_edge("final_call", END)
    workflow.add_edge("summary_lock", END)

    workflow.add_conditional_edges(
        "awaiting_final_call_response",
        final_call_response_router,
        {
            "devils_advocate": "devils_advocate",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "awaiting_snapshot_confirmation",
        snapshot_response_router,
        {
            "plaintiff": "plaintiff",
            "devils_advocate": "devils_advocate",
            "end": END,
        },
    )

    workflow.add_edge("plaintiff", "defendant")

    workflow.add_conditional_edges(
        "defendant",
        debate_router,
        {
            "continue_debate": "plaintiff",
            "judgment": "judge",
        },
    )

    workflow.add_edge("judge", "document")
    workflow.add_edge("document", END)

    return workflow.compile()


app_graph = build_graph()
