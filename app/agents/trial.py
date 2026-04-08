"""
庭审阶段节点：原告、被告、法官及文书撰写。
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.core.structured_output import call_with_json_retry
from app.graph.state import CaseState, Citation, EvidenceStatus
from app.llm import chat_completion
from app.prompts.templates import (
    DEFENDANT_SYSTEM_PROMPT,
    DOCUMENT_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PLAINTIFF_SYSTEM_PROMPT,
)
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class JudgeCitationOutput(BaseModel):
    source_id: str
    source_type: str
    rationale: str = ""


class JudgeOutput(BaseModel):
    judgment_text: str
    citations: list[JudgeCitationOutput] = Field(default_factory=list)


def _build_fact_basis(state: CaseState) -> str:
    lines = ["【已锁定的案件事实与证据链】"]
    for claim in state.claims:
        status = "已证明" if claim.status == EvidenceStatus.PROVEN else "未证明"
        evidence = ", ".join(claim.supporting_evidence) if claim.supporting_evidence else "无"
        lines.append(f"  [{status}] {claim.description} | 证据: {evidence}")

    lines.append("\n【被告可能的抗辩要点】")
    for dp in state.defense_points:
        lines.append(f"  - {dp}")

    return "\n".join(lines)


def plaintiff_node(state: CaseState) -> dict:
    retriever = HybridRetriever()
    results = retriever.search_all(
        query_text=state.case_summary or "民商事纠纷",
        n_results=5,
        user_role=state.user_role,
        phase="trial",
        filters=state.recommended_filters or None,
    )
    rag_context = retriever.format_context_for_prompt(results)
    fact_basis = _build_fact_basis(state)

    messages = [
        {"role": "system", "content": PLAINTIFF_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"案件概述：{state.case_summary}\n\n{fact_basis}\n\n"
                f"【RAG 参考】\n{rag_context}\n\n"
                "请按 IRAC 结构提出原告的法律论证。"
            ),
        },
    ]
    response = chat_completion(messages, temperature=0.2, max_tokens=4096)
    return {
        "plaintiff_arguments": response,
        "rag_context": rag_context,
        "debate_round": state.debate_round + 1,
    }


def defendant_node(state: CaseState) -> dict:
    retriever = HybridRetriever()
    query_text = f"{state.case_summary} {state.plaintiff_arguments[:500]}"
    results = retriever.search_all(
        query_text=query_text,
        n_results=5,
        user_role=state.user_role,
        phase="trial",
        filters=state.recommended_filters or None,
    )
    rag_context = retriever.format_context_for_prompt(results)
    fact_basis = _build_fact_basis(state)

    messages = [
        {"role": "system", "content": DEFENDANT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"案件概述：{state.case_summary}\n\n"
                f"【原告观点】\n{state.plaintiff_arguments}\n\n"
                f"{fact_basis}\n\n"
                f"【RAG 参考】\n{rag_context}\n\n"
                "请对原告观点提出反驳并给出抗辩路径。"
            ),
        },
    ]
    response = chat_completion(messages, temperature=0.2, max_tokens=4096)
    return {"defendant_arguments": response}


def debate_router(state: CaseState) -> str:
    if state.debate_round >= settings.system.max_debate_rounds:
        return "judgment"
    return "continue_debate"


def judge_node(state: CaseState) -> dict:
    retriever = HybridRetriever()
    judge_results = retriever.search_all(
        query_text=state.evidence_snapshot if state.evidence_snapshot else (state.case_summary or "民商事纠纷"),
        n_results=3,
        user_role=state.user_role,
        phase="conclusion",
        filters=state.recommended_filters or None,
    )
    judge_rag = retriever.format_context_for_prompt(judge_results)
    source_catalog = retriever.build_source_catalog(judge_results)
    source_ids = [item["source_id"] for item in source_catalog]
    evidence_source_set = set(state.evidence_source_ids or [])
    fact_basis = _build_fact_basis(state)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"案件基础事实：{fact_basis}\n\n"
                f"【原告论证】\n{state.plaintiff_arguments}\n\n"
                f"【被告抗辩】\n{state.defendant_arguments}\n\n"
                f"【裁判参考】\n{judge_rag}\n\n"
                f"【质证阶段已命中来源ID】\n{json.dumps(state.evidence_source_ids, ensure_ascii=False)}\n\n"
                f"【可引用来源ID清单】\n{json.dumps(source_catalog, ensure_ascii=False)}\n\n"
                "请严格输出 JSON：\n"
                "{\n"
                '  "judgment_text": "完整裁判推理与结论",\n'
                '  "citations": [\n'
                '    {"source_id":"...", "source_type":"law_article|court_case", "rationale":"该来源支持的要点"}\n'
                "  ]\n"
                "}\n"
                "约束：citations 中的 source_id 必须来自可引用来源ID清单；不得编造新ID。"
                "优先引用质证阶段已命中来源ID，若不足可补充新增来源。"
            ),
        },
    ]
    output, raw = call_with_json_retry(
        messages=messages,
        schema_model=JudgeOutput,
        temperature=0.1,
        max_tokens=6000,
        retries=2,
    )
    if output is None:
        return {"judgment": raw or "裁判推理生成失败。", "judgment_citations": [], "phase": "document"}

    allowed = set(source_ids)
    citations: list[Citation] = []
    catalog_map = {item["source_id"]: item for item in source_catalog}
    for citation in output.citations:
        if citation.source_id not in allowed:
            continue
        src = catalog_map.get(citation.source_id, {})
        citations.append(
            Citation(
                source_id=citation.source_id,
                source_type=citation.source_type,
                label=src.get("label", ""),
                case_id=src.get("case_id", ""),
                law_name=src.get("law_name", ""),
                rationale=citation.rationale,
                from_evidence_stage=citation.source_id in evidence_source_set,
            )
        )

    return {
        "judgment": output.judgment_text,
        "judgment_citations": citations,
        "phase": "document",
    }


def document_node(state: CaseState) -> dict:
    parties_dict = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v)
        for k, v in state.parties.items()
    }

    citation_lines = []
    for c in state.judgment_citations:
        citation_lines.append(
            f"- source_id={c.source_id} type={c.source_type} from_evidence_stage={c.from_evidence_stage} "
            f"label={c.label} case_id={c.case_id} law={c.law_name} rationale={c.rationale}"
        )
    citation_block = "\n".join(citation_lines) if citation_lines else "（无）"

    messages = [
        {"role": "system", "content": DOCUMENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"案件信息：{state.case_summary}\n"
                f"当事人：{json.dumps(parties_dict, ensure_ascii=False)}\n\n"
                f"裁判推理结果：\n{state.judgment}\n\n"
                f"可追溯引用：\n{citation_block}\n\n"
                "请将以上推理结果转化为规范格式的民事裁判文书初稿，并在文末增加“引用依据”小节列出 source_id。"
            ),
        },
    ]

    response = chat_completion(messages, temperature=0.2, max_tokens=8000)

    return {
        "final_document": response,
        "messages": [AIMessage(content=f"裁判文书已生成：\n\n{response}")],
        "phase": "completed",
    }
