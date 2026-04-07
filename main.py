"""
民商事司法智能体 - 命令行入口
用于本地调试和测试，后续将对接 Web 前端。
"""

import logging
import sys

from langchain_core.messages import HumanMessage

from app.graph.state import CaseState
from app.graph.workflow import app_graph

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def main():
    print("=" * 60)
    print("  📜 民商事司法智能体 (外商投资纠纷 MVP)")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()

    # 初始化全局状态
    state = CaseState()

    while True:
        try:
            user_input = input("👤 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 插入用户消息
        state.messages.append(HumanMessage(content=user_input))

        # 处理用户确认快照的特殊逻辑
        if state.phase == "awaiting_snapshot_confirmation":
            if "确认" in user_input:
                state.user_confirmed_snapshot = True
                state.phase = "trial"
                # 直接进入法庭辩论 (跳过 reception)
                # 手动触发子图2的执行
                print("\n🔒 事实已锁定，进入法庭辩论阶段...\n")

                # 执行子图2: 正方 -> 反方 -> 裁判 -> 文书
                from app.agents.trial import plaintiff_node, defendant_node, judge_node, document_node

                result = plaintiff_node(state)
                state.plaintiff_arguments = result.get("plaintiff_arguments", "")
                state.rag_context = result.get("rag_context", state.rag_context)
                state.debate_round = result.get("debate_round", state.debate_round)
                print("⚖️ 正方论证完成...")

                result = defendant_node(state)
                state.defendant_arguments = result.get("defendant_arguments", "")
                print("⚖️ 反方抗辩完成...")

                result = judge_node(state)
                state.judgment = result.get("judgment", "")
                print("👨‍⚖️ 裁判推理完成...")

                result = document_node(state)
                state.final_document = result.get("final_document", "")
                for m in result.get("messages", []):
                    print(f"\n🤖 系统: {m.content}")
                continue
            else:
                print("\n🤖 系统: 收到，请补充您需要修正的内容。\n")
                state.evidence_locked = False
                state.phase = "evidence_exchange"

        # 处理 Final Call 回复
        if state.phase == "awaiting_final_call_response":
            if "确认" in user_input or "没有" in user_input:
                # 用户确认无新证据 -> 进入 Summary & Lock
                from app.agents.evidence import summary_lock_node
                result = summary_lock_node(state)
                for key, val in result.items():
                    if key == "messages":
                        for m in val:
                            print(f"\n🤖 系统: {m.content}")
                    else:
                        setattr(state, key, val)
                continue
            else:
                # 用户补充了新信息 -> 继续举证循环
                state.phase = "evidence_exchange"

        # 执行 LangGraph 工作流 (带有回复路由逻辑)
        try:
            # 根据当前 phase 决定入口
            if state.phase == "evidence_exchange" and state.evidence_round > 0:
                # 用户回复了追问 -> 通过 Dual Check 判断
                from app.agents.evidence import dual_check_node, devils_advocate_node

                check_result = dual_check_node(state)
                next_phase = check_result.get("phase", "evidence_exchange")

                if next_phase == "final_call":
                    from app.agents.evidence import final_call_node
                    result = final_call_node(state)
                    for key, val in result.items():
                        if key == "messages":
                            for m in val:
                                print(f"\n🤖 系统: {m.content}")
                        else:
                            setattr(state, key, val)
                else:
                    result = devils_advocate_node(state)
                    for key, val in result.items():
                        if key == "messages":
                            for m in val:
                                print(f"\n🤖 系统: {m.content}")
                        elif key == "defense_points":
                            state.defense_points = val
                        else:
                            setattr(state, key, val)
            else:
                # 通过完整的 LangGraph 运行
                result = app_graph.invoke(state.model_dump())

                # 提取 AI 回复
                if "messages" in result:
                    msgs = result["messages"]
                    # 获取最新的 AI 消息
                    for m in reversed(msgs):
                        if hasattr(m, "content") and m.content:
                            print(f"\n🤖 系统: {m.content}")
                            break

                # 将 graph 结果同步回 state
                for key in result:
                    if hasattr(state, key) and key != "messages":
                        setattr(state, key, result[key])
                if "messages" in result:
                    state.messages = result["messages"]

        except Exception as e:
            logger.error("执行出错: %s", e, exc_info=True)
            print(f"\n⚠️ 系统出错: {e}\n")

        print()  # 空行分隔


if __name__ == "__main__":
    main()
