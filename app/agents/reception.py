"""
接入对接 Agent (Reception Agent)
负责：意图识别、槽位填充、反向排雷、分流路由。
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from datetime import datetime

from app.llm import chat_completion
from app.graph.state import CaseState, IntentType, Claim, EvidenceStatus, PartyInfo
from app.prompts.templates import RECEPTION_SYSTEM_PROMPT
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def reception_node(state: CaseState) -> dict:
    """
    接入对接节点：分析用户输入，收集槽位，判断意图。
    """
    # 获取最近的用户消息
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"messages": [AIMessage(content="您好！请问有什么法律问题需要咨询？")]}

    latest_user_msg = user_messages[-1].content

    # 1. 获取全局元数据并注入 Prompt
    retriever = HybridRetriever()
    
    # 执行 RAG 检索 (全量检索案例与法条，获取导诉背景)
    rag_results = retriever.search_all(query_text=latest_user_msg)
    rag_context = retriever.format_context_for_prompt(rag_results)

    meta_registry = retriever.get_available_metadata()
    sys_prompt = RECEPTION_SYSTEM_PROMPT.replace(
        "{{available_elements}}", json.dumps(meta_registry.get("elements", []), ensure_ascii=False)
    ).replace(
        "{{available_keywords}}", json.dumps(meta_registry.get("keywords", []), ensure_ascii=False)
    ).replace(
        "{{rag_context}}", rag_context
    )

    # 2. 构造消息链
    parties_dict = {k: (v.model_dump() if hasattr(v, 'model_dump') else v.dict()) for k, v in state.parties.items()}
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"用户输入：{latest_user_msg}\n\n当前已收集信息：{json.dumps(parties_dict, ensure_ascii=False)}"},
    ]

    response_text = chat_completion(messages, temperature=0.2)

    # 智能解析响应中的 JSON
    import re
    def extract_json(text: str) -> dict | None:
        try:
            # 1. 尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            # 2. 尝试提取 ```json ... ``` 块
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            # 3. 尝试提取最外层的 { ... }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
        return None

    parsed = extract_json(response_text)
    
    if parsed is None:
        # 如果解析彻底失败，尝试清理掉所有的 JSON 部分给出一个纯文本回复
        clean_text = re.sub(r"```json.*?```", "", response_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"\{.*?\}", "", clean_text, flags=re.DOTALL).strip()
        return {
            "messages": [AIMessage(content=clean_text if clean_text else "抱歉，系统处理出现逻辑错误，请稍后再试。")],
        }

    # 更新状态
    reply_content = parsed.get("reply_to_user", response_text)
    updates: dict = {
        "messages": [AIMessage(content=reply_content)],
    }

    # 更新意图
    intent_str = parsed.get("intent", "unclear")
    if intent_str in [e.value for e in IntentType]:
        updates["intent"] = IntentType(intent_str)

    # 更新由大模型解析的当事人信息，计算年龄并打上行为能力标签
    collected_parties = parsed.get("parties", {})
    if collected_parties:
        for party_id, info_dict in collected_parties.items():
            party_info = PartyInfo(**info_dict)
            # 只有在明确给出出生日期或年龄是数字字符串时，才自动计算行为能力
            if party_info.birth_date or party_info.age:
                try:
                    age_val = None
                    if party_info.birth_date:
                        birth_dt = datetime.strptime(party_info.birth_date, "%Y-%m-%d")
                        age_val = (datetime.now() - birth_dt).days // 365
                        party_info.age = str(age_val)
                    elif party_info.age and party_info.age.isdigit():
                        age_val = int(party_info.age)
                    
                    if age_val is not None:
                        if age_val >= 18:
                            party_info.capacity_status = "完全民事行为能力"
                        elif age_val >= 8:
                            party_info.capacity_status = "限制民事行为能力"
                        else:
                            party_info.capacity_status = "无民事行为能力"
                except Exception as e:
                    logger.warning("无法解析年龄或日期进行行为能力判定: %s", e)
            state.parties[party_id] = party_info
        updates["parties"] = state.parties

    # 更新槽位信息 (放入 state.parties 中作为附属结构存储，不过为了简单这里直接合入)
    collected_slots = parsed.get("collected_slots", {})
    collected_slots = parsed.get("collected_slots", {})
    if collected_slots:
        # Here we just use a top-level key or save to original parties dict.
        # Since state.parties is now dict[str, PartyInfo], we cannot just merge random slots directly into it.
        # We will add an extra slot memory in CaseState or just push strings to case_summary.
        # Let's keep it in case_summary for now since CaseState slot_filled is a boolean.
        updates["case_summary"] = state.case_summary + "\n[已收集要素]: " + json.dumps(collected_slots, ensure_ascii=False)

    # 更新概述
    if "case_summary" not in updates and collected_slots:
        updates["case_summary"] = latest_user_msg[:500]
        
    # 获取用户角色身份
    user_role = parsed.get("user_role", "unclear")
    updates["user_role"] = user_role

    # 实时审计插槽同步 (无论槽位是否填满都展示)
    updates["risk_alerts"] = parsed.get("risk_alerts", [])
    updates["missing_slots"] = parsed.get("missing_slots", [])

    # 更新过滤器建议与案由
    case_type = parsed.get("cause_of_action", "general_civil_case")
    updates["case_type"] = case_type
    updates["recommended_filters"] = parsed.get("recommended_filters", {"elements": [], "keywords": []})

    # 更新插槽信息 (Static Slots)
    collected_slots = parsed.get("collected_slots", {})
    if collected_slots:
        # 合并存入 state.slots (深度合并或直接覆盖)
        new_slots = state.slots.copy()
        new_slots.update(collected_slots)
        updates["slots"] = new_slots
        
        # 同时反映到 case_summary 以便 RAG 使用
        updates["case_summary"] = state.case_summary + "\n[已收集要素]: " + json.dumps(collected_slots, ensure_ascii=False)

    # 标记槽位是否已填满 (且身份明确)
    if parsed.get("slot_filled", False) and user_role != "unclear":
        updates["slot_filled"] = True
        updates["phase"] = "evidence_exchange"
        # 将最终风险点存入防御出口
        updates["defense_points"] = updates.get("risk_alerts", [])

    return updates


def intent_router(state: CaseState) -> str:
    """
    意图路由：根据当前状态决定下一步流向。
    """
    if state.intent == IntentType.SIMPLE_QA:
        return "simple_qa"
    elif state.intent == IntentType.COMPLEX_CASE and state.slot_filled:
        return "evidence_exchange"
    else:
        return "continue_reception"


def simple_qa_node(state: CaseState) -> dict:
    """
    简单问答节点：走普通 RAG 回答路径。
    """
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {}

    query = user_messages[-1].content
    retriever = HybridRetriever()

    # 检索法条（简单 QA 一般只需法条）
    results = retriever.search_law_articles(
        query_text=query,
        n_results=3,
    )
    context = retriever.format_context_for_prompt({"law_articles": results, "court_cases": []})

    messages = [
        {"role": "system", "content": f"你是一位专业的法律顾问。请基于以下资料回答用户的法律问题。\n\n{context}"},
        {"role": "user", "content": query},
    ]

    answer = chat_completion(messages, temperature=0.2)

    return {
        "messages": [AIMessage(content=answer)],
        "simple_answer": answer,
        "rag_context": context,
    }
