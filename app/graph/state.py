""" 智能体工作流状态模型"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """用户意图分类"""

    SIMPLE_QA = "simple_qa"
    COMPLEX_CASE = "complex_case"
    UNCLEAR = "unclear"


class EvidenceStatus(str, Enum):
    """证据状态"""

    PROVEN = "proven"
    UNPROVEN = "unproven"
    DISPUTED = "disputed"


class Claim(BaseModel):
    """证据阶段的法律主张"""

    claim_id: str
    description: str
    supporting_evidence: list[str] = Field(default_factory=list)
    status: EvidenceStatus = EvidenceStatus.UNPROVEN
    notes: str = ""


class PartyInfo(BaseModel):
    """当事人信息"""

    name: str = "未知"
    type: str = "自然人"
    birth_date: str = ""  # YYYY-MM-DD
    age: str | None = None
    capacity_status: str = "未知"


class Citation(BaseModel):
    """可追溯的来源引用，由法官/文档节点使用"""

    source_id: str
    source_type: str  # law_article | court_case
    label: str = ""
    case_id: str = ""
    law_name: str = ""
    rationale: str = ""
    from_evidence_stage: bool = False


class CaseState(BaseModel):
    """跨越所有图阶段的端到端案件状态"""

    messages: Annotated[list, add_messages] = Field(default_factory=list)

    intent: IntentType = IntentType.UNCLEAR
    user_role: str = "unclear"  # plaintiff_side | defendant_side | unclear
    phase: str = "reception"

    case_summary: str = ""
    parties: dict[str, PartyInfo] = Field(default_factory=dict)
    slot_filled: bool = False

    claims: list[Claim] = Field(default_factory=list)
    defense_points: list[str] = Field(default_factory=list)
    evidence_source_ids: list[str] = Field(default_factory=list)
    evidence_round: int = 0
    evidence_locked: bool = False
    evidence_snapshot: str = ""
    user_confirmed_snapshot: bool = False

    plaintiff_arguments: str = ""
    defendant_arguments: str = ""
    debate_round: int = 0

    judgment: str = ""
    judgment_citations: list[Citation] = Field(default_factory=list)
    final_document: str = ""

    case_type: str = "general_civil_case"
    slots: dict[str, Any] = Field(default_factory=dict)
    is_fact_locked: bool = False
    slot_query_count: dict[str, int] = Field(default_factory=dict)
    slot_blocked: dict[str, str] = Field(default_factory=dict)  # slot_id -> user reason/snippet
    last_requested_slots: list[str] = Field(default_factory=list)

    rag_context: str = ""

    risk_alerts: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    recommended_filters: dict[str, list[str]] = Field(default_factory=dict)

    simple_answer: str = ""
