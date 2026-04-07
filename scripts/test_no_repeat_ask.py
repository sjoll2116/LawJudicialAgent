from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.agents import evidence as evidence_mod
from app.graph.state import CaseState


class _FakeRetriever:
    def search_court_cases(self, *args, **kwargs):
        return []

    def format_context_for_prompt(self, _res):
        return "<mock-rag-context>"


def test_no_repeat_asking_for_blocked_slot(monkeypatch):
    monkeypatch.setattr(evidence_mod, "HybridRetriever", lambda: _FakeRetriever())

    def _fake_call_with_json_retry(**kwargs):
        del kwargs
        output = evidence_mod.DevilsAdvocateOutput(
            reply_to_user="请补充另一个要素。",
            defense_points=[],
            inferred_claims=[],
        )
        return output, "{}"

    monkeypatch.setattr(evidence_mod, "call_with_json_retry", _fake_call_with_json_retry)

    state = CaseState(
        case_type="commercial_contract",
        phase="evidence_exchange",
        last_requested_slots=["contract_evidence"],
        messages=[HumanMessage(content="只有一张借条，没有其他材料")],
    )

    updates = evidence_mod.devils_advocate_node(state)

    assert "contract_evidence" in updates.get("slot_blocked", {})
    assert "contract_evidence" not in updates.get("last_requested_slots", [])
