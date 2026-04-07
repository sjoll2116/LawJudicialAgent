"""
Offline regression test for phase routing and workflow continuity.
This test avoids external dependencies (Chroma/LLM APIs) via monkeypatching.
"""

from __future__ import annotations

import importlib
import json


class FakeRetriever:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def __init__(self):
        pass

    def get_available_metadata(self):
        return {"elements": ["合同", "付款"], "keywords": ["违约", "证据"]}

    def search_all(self, *args, **kwargs):
        return {"law_articles": [], "court_cases": []}

    def search_law_articles(self, *args, **kwargs):
        return []

    def search_court_cases(self, *args, **kwargs):
        return []

    def format_context_for_prompt(self, res):
        del res
        return "<mock-rag-context>"

    def build_source_catalog(self, res):
        del res
        return [
            {
                "source_id": "mock_law_1",
                "source_type": "law_article",
                "label": "测试法条",
                "law_name": "测试法条",
                "case_id": "",
            }
        ]


def _fake_chat_factory():
    call_idx = {"json": 0}

    def fake_chat_completion(messages, temperature=0.3, max_tokens=4096, json_mode=False, **kwargs):
        del temperature, max_tokens, kwargs
        if json_mode:
            call_idx["json"] += 1
            idx = call_idx["json"]
            if idx == 1:
                return json.dumps(
                    {
                        "intent": "complex_case",
                        "cause_of_action": "commercial_contract",
                        "core_judgment": "可以起诉，需补强履约和违约证据",
                        "reply_to_user": "初步判断可以起诉，下面进入事实质证。",
                        "user_role": "plaintiff_side",
                        "parties": {"甲方": {"name": "A公司", "age": "30"}},
                        "collected_slots": {
                            "contract_evidence": "已签合同",
                            "performance_evidence": "已付款流水",
                        },
                        "missing_slots": ["breach_facts", "penalty_clause"],
                        "risk_alerts": ["违约责任条款存在解释空间"],
                        "recommended_filters": {"elements": ["履约"], "keywords": ["违约金"]},
                        "slot_filled": True,
                    },
                    ensure_ascii=False,
                )
            if idx == 2:
                return json.dumps(
                    {
                        "reply_to_user": "请补充违约发生时间和催告记录。",
                        "defense_points": ["对方可能主张未到期"],
                        "inferred_claims": [
                            {
                                "claim_id": "breach_facts",
                                "description": "被告逾期付款",
                                "supporting_evidence": ["催告函"],
                                "status": "proven",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            if idx == 3:
                return json.dumps(
                    {
                        "has_new_facts": False,
                        "recommendation": "trigger_final_call",
                        "reason": "用户未补充新事实",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "has_new_facts": True,
                    "recommendation": "continue",
                    "reason": "fallback",
                },
                ensure_ascii=False,
            )

        return "[mock-narrative-output]"

    return fake_chat_completion


def test_workflow_regression_offline(monkeypatch):
    retriever_mod = importlib.import_module("app.rag.retriever")
    reception_mod = importlib.import_module("app.agents.reception")
    evidence_mod = importlib.import_module("app.agents.evidence")
    trial_mod = importlib.import_module("app.agents.trial")
    services_mod = importlib.import_module("app.api.services")
    structured_mod = importlib.import_module("app.core.structured_output")

    monkeypatch.setattr(retriever_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(reception_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(evidence_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(trial_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(services_mod, "HybridRetriever", FakeRetriever)

    fake_chat = _fake_chat_factory()
    monkeypatch.setattr(reception_mod, "chat_completion", fake_chat)
    monkeypatch.setattr(structured_mod, "chat_completion", fake_chat)
    monkeypatch.setattr(trial_mod, "chat_completion", fake_chat)

    service = services_mod.LegalApiService()
    state = None

    state = service.chat("我想起诉对方违约，请帮我分析", state)
    assert state.get("phase") == "evidence_exchange"
    assert state.get("evidence_round", 0) >= 1

    state = service.chat("没有新事实", state)
    assert state.get("phase") == "awaiting_final_call_response"

    state = service.chat("确认", state)
    assert state.get("phase") == "awaiting_snapshot_confirmation"

    state = service.chat("确认", state)
    assert state.get("phase") == "completed"
    assert state.get("final_document")
