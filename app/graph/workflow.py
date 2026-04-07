"""
LangGraph 主工作流定义
编排双子图结构：
  子图1: 庭前举证质证循环 (Pre-Trial Evidence Exchange)
  子图2: 法庭辩论与裁判 (Trial & Judgment)

整体拓扑:
  Reception -> [简单QA] -> END
  Reception -> [复杂案件] -> 事实查明质证 <-> 用户补充举证 -> Final Call -> Summary&Lock -> 法庭辩论 -> 裁判 -> 文书 -> END
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from app.graph.state import CaseState
from app.agents.reception import reception_node, intent_router, simple_qa_node
from app.agents.evidence import (
    devils_advocate_node,
    dual_check_node,
    evidence_round_router,
    final_call_node,
    summary_lock_node,
)
from app.agents.trial import (
    plaintiff_node,
    defendant_node,
    debate_router,
    judge_node,
    document_node,
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    构建并返回完整的 LangGraph 工作流。

    Returns:
        编译后的 StateGraph
    """
    workflow = StateGraph(CaseState)

    # ========================
    # 添加所有节点
    # ========================

    # --- 接入层 ---
    workflow.add_node("reception", reception_node)
    workflow.add_node("simple_qa", simple_qa_node)

    # --- 子图1: 庭前举证质证 ---
    workflow.add_node("devils_advocate", devils_advocate_node)
    workflow.add_node("dual_check", dual_check_node)
    workflow.add_node("final_call", final_call_node)
    workflow.add_node("summary_lock", summary_lock_node)

    # --- 子图2: 法庭辩论与裁判 ---
    workflow.add_node("plaintiff", plaintiff_node)
    workflow.add_node("defendant", defendant_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("document", document_node)

    # ========================
    # 定义边 (Edges)
    # ========================

    # 入口节点
    workflow.set_entry_point("reception")

    # Reception -> 意图路由
    workflow.add_conditional_edges(
        "reception",
        intent_router,
        {
            "simple_qa": "simple_qa",
            "evidence_exchange": "devils_advocate",
            "continue_reception": END,  # 等待用户继续输入
        },
    )

    # 简单QA -> 结束
    workflow.add_edge("simple_qa", END)

    # 事实查明质证 -> 等待用户回复 (返回 END 等用户输入)
    workflow.add_edge("devils_advocate", END)

    # Dual Check -> 路由
    workflow.add_conditional_edges(
        "dual_check",
        evidence_round_router,
        {
            "continue_evidence": "devils_advocate",
            "final_call": "final_call",
            "await_confirmation": "summary_lock",
            "trial": "plaintiff",
        },
    )

    # Final Call -> 等待用户确认
    workflow.add_edge("final_call", END)

    # Summary & Lock -> 等待用户确认
    workflow.add_edge("summary_lock", END)

    # --- 子图2: 法庭辩论 ---
    # 正方 -> 反方
    workflow.add_edge("plaintiff", "defendant")

    # 反方 -> 辩论轮次路由
    workflow.add_conditional_edges(
        "defendant",
        debate_router,
        {
            "continue_debate": "plaintiff",  # 继续辩论
            "judgment": "judge",              # 进入裁判
        },
    )

    # 裁判 -> 文书
    workflow.add_edge("judge", "document")

    # 文书 -> 结束
    workflow.add_edge("document", END)

    return workflow.compile()


# 全局编译后的图实例
app_graph = build_graph()
