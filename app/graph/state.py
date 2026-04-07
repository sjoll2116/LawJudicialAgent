"""
LangGraph 全局状态定义 (CaseState)
定义贯穿整个审判流程的状态数据结构。

状态在两个子图之间共享：
  子图1: 庭前举证质证循环 (Pre-Trial Evidence Exchange)
  子图2: 法庭辩论与裁判 (Trial & Judgment)
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class IntentType(str, Enum):
    """用户意图类型"""
    SIMPLE_QA = "simple_qa"             # 简单知识问答
    COMPLEX_CASE = "complex_case"       # 复杂案件裁判
    UNCLEAR = "unclear"                 # 意图不明确，需追问


class EvidenceStatus(str, Enum):
    """证据状态"""
    PROVEN = "proven"           # 已证明（有证据支撑）
    UNPROVEN = "unproven"       # 未证明（缺乏证据）
    DISPUTED = "disputed"       # 有争议（双方各执一词）


class Claim(BaseModel):
    """单条法律主张"""
    claim_id: str
    description: str                              # 主张描述
    supporting_evidence: list[str] = Field(default_factory=list)  # 支撑证据列表
    status: EvidenceStatus = EvidenceStatus.UNPROVEN
    notes: str = ""                               # 备注 (如质证意见)


class PartyInfo(BaseModel):
    """当事人身份信息"""
    name: str = "未知"
    type: str = "自然人"  # "自然人" 或 "法人/非法人组织"
    birth_date: str = ""  # YYYY-MM-DD
    age: str | None = None
    capacity_status: str = "未知"  # "完全民事行为能力", "限制民事行为能力"


class CaseState(BaseModel):
    """
    全局案件状态 - 贯穿整个 LangGraph 流程。

    Attributes:
        messages: LangGraph 消息历史 (Human/AI/System)
        intent: 用户意图分类
        phase: 当前流程阶段

        # --- 案件基础信息 (由 Reception Agent 采集) ---
        case_summary: 案件概述
        parties: 当事人信息
        slot_filled: 关键槽位是否已填充

        # --- 庭前举证质证循环 ---
        claims: 正方主张列表
        defense_points: 反方抗辩要点
        evidence_round: 当前举证轮次
        evidence_locked: 举证是否已锁定 (Summary & Lock)
        evidence_snapshot: 锁定时的事实快照 (庭审笔录确认)

        # --- 法庭辩论 ---
        plaintiff_arguments: 正方辩论文书
        defendant_arguments: 反方辩论文书
        debate_round: 当前辩论轮次

        # --- 裁判结果 ---
        judgment: 裁判 Agent 的 IRAC 推理结果
        final_document: 文书 Agent 生成的最终裁判文书

        # --- RAG 上下文 ---
        rag_context: 当前轮次的 RAG 检索上下文
    """

    # LangGraph 消息链
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # 流程控制
    intent: IntentType = IntentType.UNCLEAR
    user_role: str = "unclear"  # plaintiff_side, defendant_side, unclear
    phase: str = "reception"  # reception -> evidence_exchange -> trial -> judgment -> document

    # 案件基础信息
    case_summary: str = ""
    parties: dict[str, PartyInfo] = Field(default_factory=dict)
    slot_filled: bool = False

    # 庭前举证质证
    claims: list[Claim] = Field(default_factory=list)
    defense_points: list[str] = Field(default_factory=list)
    evidence_round: int = 0
    evidence_locked: bool = False
    evidence_snapshot: str = ""
    user_confirmed_snapshot: bool = False

    # 法庭辩论
    plaintiff_arguments: str = ""
    defendant_arguments: str = ""
    debate_round: int = 0

    # 裁判结果
    judgment: str = ""
    final_document: str = ""

    # 插槽管理 (Static Slots)
    case_type: str = "general_civil_case"  # 匹配 case_templates.json 的 key
    slots: dict[str, Any] = Field(default_factory=dict)
    is_fact_locked: bool = False

    # RAG 上下文
    rag_context: str = ""

    # 实时审计插槽 (用于展示 AI 的思考过程)
    risk_alerts: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    
    # AI 推荐的检索过滤器 (要素与关键词)
    recommended_filters: dict[str, list[str]] = Field(default_factory=dict)

    # 简单QA直接回复
    simple_answer: str = ""
