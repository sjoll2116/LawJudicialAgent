
import sys
import os
import json
from langchain_core.messages import HumanMessage

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.state import CaseState, IntentType
from app.agents.reception import reception_node
from app.agents.evidence import devils_advocate_node

def test_slot_filling_flow():
    print("=== 开始测试: 股权转让纠纷事实查明流程 ===")
    
    # 1. 模拟初始接入
    state = CaseState(
        messages=[HumanMessage(content="我想咨询关于股权转让的事，我们签了协议，但我还没拿到钱。对方是张三。")]
    )
    
    print("\n[Step 1: Reception]")
    reception_updates = reception_node(state)
    state = state.model_copy(update=reception_updates)
    
    print(f"识别到的案由: {state.case_type}")
    print(f"已收集插槽: {json.dumps(state.slots, ensure_ascii=False)}")
    print(f"缺失插槽: {state.missing_slots}")
    print(f"AI 回复: {state.messages[-1].content}")

    # 2. 模拟进入魔鬼代言人逻辑 (假设 slot_filled 为 True 或手动触发)
    print("\n[Step 2: Devil's Advocate]")
    state.phase = "evidence_exchange"
    da_updates = devils_advocate_node(state)
    state = state.model_copy(update=da_updates)
    
    print(f"魔鬼代言人追问内容:\n{state.messages[-1].content}")
    print(f"当前抗辩要点: {state.defense_points}")

if __name__ == "__main__":
    test_slot_filling_flow()
