"""Run an end-to-end workflow test and record all LLM outputs.

Usage examples:
  python scripts/run_full_workflow_test.py --mock
  python scripts/run_full_workflow_test.py --out logs/llm_trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


class LLMRecorder:
    def __init__(self, out_file: Path):
        self.out_file = out_file
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self._records = 0

    def wrap(self, source: str, fn: Callable[..., str]) -> Callable[..., str]:
        def _wrapped(
            messages: list[dict[str, Any]],
            model: str | None = None,
            temperature: float = 0.3,
            max_tokens: int = 4096,
            json_mode: bool = False,
            **kwargs,
        ) -> str:
            start = time.time()
            entry: dict[str, Any] = {
                "ts": datetime.now(UTC).isoformat(),
                "source": source,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "messages": messages,
            }
            try:
                response = fn(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    **kwargs,
                )
                entry["response"] = response
                entry["ok"] = True
                return response
            except Exception as e:
                entry["ok"] = False
                entry["error"] = str(e)
                raise
            finally:
                entry["latency_ms"] = int((time.time() - start) * 1000)
                with self.out_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._records += 1

        return _wrapped

    @property
    def records(self) -> int:
        return self._records


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

    def format_context_for_prompt(self, _res):
        return "<mock-rag-context>"

    def build_source_catalog(self, _res):
        return [
            {
                "source_id": "mock_law_1",
                "source_type": "law_article",
                "label": "测试法条",
                "law_name": "测试法条",
                "case_id": "",
            }
        ]


def make_fake_chat_completion() -> Callable[..., str]:
    calls = {"json": 0}

    def _fake(
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        del messages, model, temperature, max_tokens, kwargs
        if json_mode:
            calls["json"] += 1
            idx = calls["json"]
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
            if idx == 4:
                return json.dumps(
                    {
                        "judgment_text": "本院认为被告构成违约，应承担给付责任。",
                        "citations": [
                            {
                                "source_id": "mock_law_1",
                                "source_type": "law_article",
                                "rationale": "支持违约责任认定",
                            }
                        ],
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

    return _fake


def default_next_input(state: dict[str, Any]) -> str:
    phase = state.get("phase", "reception")
    evidence_round = int(state.get("evidence_round", 0) or 0)

    if phase == "awaiting_final_call_response":
        return "确认"
    if phase == "awaiting_snapshot_confirmation":
        return "确认"
    if phase == "evidence_exchange":
        if evidence_round <= 1:
            return "补充：合同已签署，对方应于2025-03-01付款但未付款，我方有合同和银行流水。"
        return "没有新事实补充"
    if phase == "reception":
        return "我方是原告，诉求返还欠款并支付违约金。"
    return "继续"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full workflow test and record LLM outputs.")
    parser.add_argument("--mock", action="store_true", help="Run with mocked retriever and LLM.")
    parser.add_argument("--max-turns", type=int, default=8, help="Maximum conversation turns.")
    parser.add_argument("--out", type=str, default="", help="Output JSONL for LLM trace.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.out) if args.out else Path("logs") / f"llm_trace_{ts}.jsonl"
    convo_file = out_file.with_name(out_file.stem + "_conversation.json")

    recorder = LLMRecorder(out_file)

    import app.agents.reception as reception_mod
    import app.agents.trial as trial_mod
    import app.api.services as services_mod
    import app.core.structured_output as structured_mod
    import app.llm as llm_mod
    import app.rag.retriever as retriever_mod

    base_chat = llm_mod.chat_completion
    if args.mock:
        fake_chat = make_fake_chat_completion()
        base_chat = fake_chat

        retriever_mod.HybridRetriever = FakeRetriever
        reception_mod.HybridRetriever = FakeRetriever
        import app.agents.evidence as evidence_mod

        evidence_mod.HybridRetriever = FakeRetriever
        trial_mod.HybridRetriever = FakeRetriever
        services_mod.HybridRetriever = FakeRetriever

    reception_mod.chat_completion = recorder.wrap("reception", base_chat)
    trial_mod.chat_completion = recorder.wrap("trial", base_chat)
    structured_mod.chat_completion = recorder.wrap("structured_output", base_chat)

    service = services_mod.LegalApiService()

    conversation: list[dict[str, Any]] = []
    user_input = "我想起诉对方违约，请帮我分析"
    state: dict[str, Any] | None = None

    for turn in range(1, args.max_turns + 1):
        state = service.chat(user_input, state)

        messages = state.get("messages", []) if state else []
        last_msg = messages[-1] if messages else {"role": "ai", "content": ""}
        if hasattr(last_msg, "content"):
            ai_text = getattr(last_msg, "content", "")
        elif isinstance(last_msg, dict):
            ai_text = last_msg.get("content", "")
        else:
            ai_text = str(last_msg)

        record = {
            "turn": turn,
            "user_input": user_input,
            "phase": state.get("phase") if state else None,
            "evidence_round": state.get("evidence_round") if state else None,
            "ai_output": ai_text,
        }
        conversation.append(record)

        if state and state.get("phase") == "completed":
            break

        user_input = default_next_input(state or {})

    convo_file.write_text(json.dumps(conversation, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"TRACE_FILE={out_file}")
    print(f"CONVERSATION_FILE={convo_file}")
    print(f"LLM_RECORDS={recorder.records}")
    print(f"FINAL_PHASE={(state or {}).get('phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
