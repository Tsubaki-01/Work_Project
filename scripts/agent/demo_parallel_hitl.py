"""
模块名称：demo_parallel_hitl
功能描述：验证“并行多意图 + HITL 中断恢复”的后端链路。

使用方式：
    python scripts/agent/demo_parallel_hitl.py

覆盖场景：
    1. 一条复合指令同时触发 medical_agent 与 device_agent（并行）
    2. device_agent 命中中风险工具调用，触发 interrupt
    3. 用户确认后图继续执行，最终进入 response_synthesizer / output_guard
"""

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from silver_pilot.agent import create_initial_state, initialize_agent
from silver_pilot.agent.memory import UserProfileManager


def _print_interrupts(graph_state: object) -> None:
    tasks = getattr(graph_state, "tasks", [])
    for task in tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            for intr in task.interrupts:
                val = intr.value if isinstance(intr.value, dict) else {"message": str(intr.value)}
                print("[HITL] message:", val.get("message", ""))
                print("[HITL] tool:", val.get("tool_name", ""))
                print("[HITL] risk:", val.get("risk_level", ""))


def main() -> None:
    print("=" * 68)
    print("并行 + HITL 回归 Demo")
    print("=" * 68)

    graph = initialize_agent(skip_rag=True)
    cfg = {"configurable": {"thread_id": "parallel_hitl_demo_001"}}

    # 注入一个有设备与医疗相关背景的画像（用于提高命中稳定性）
    manager = UserProfileManager()
    manager.update_profile(
        "elderly_parallel_demo",
        {
            "chronic_diseases": ["高血压"],
            "current_medications": [{"name": "阿司匹林", "dosage": "100mg/日"}],
            "emergency_contacts": [{"name": "王小明（儿子）", "phone": "13800138000"}],
        },
    )
    profile = manager.get_profile("elderly_parallel_demo")

    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="把客厅空调关了，再帮我看看阿司匹林和降压药能不能一起吃")
    ]
    state["user_profile"] = profile

    print("\n[用户] 把客厅空调关了，再帮我看看阿司匹林和降压药能不能一起吃")
    print("[系统] 首次 invoke（预期：并行执行，且 device 分支可能 HITL 中断）")

    result = graph.invoke(state, config=cfg)
    graph_state = graph.get_state(cfg)
    next_nodes = list(getattr(graph_state, "next", []) or [])
    print("[系统] next:", next_nodes)

    if next_nodes:
        print("[系统] 检测到中断，输出中断详情")
        _print_interrupts(graph_state)

        print("\n[用户] 确认")
        result = graph.invoke(Command(resume="确认"), config=cfg)

    final_response = result.get("final_response", "")
    print("\n[助手]", final_response if final_response else "(无 final_response)")

    messages = result.get("messages", [])
    ai_count = sum(1 for m in messages if getattr(m, "type", "") == "ai")
    print(f"[调试] 本轮 AI 消息数: {ai_count}")
    print(f"[调试] tool_results: {result.get('tool_results', [])}")
    print(f"[调试] safety_flags: {result.get('safety_flags', [])}")

    print("\n[断言提示] 若 final_response 非空且 tool_results 已更新，则并行+HITL 主流程通过。")


if __name__ == "__main__":
    main()
