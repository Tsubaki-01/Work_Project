"""
模块名称：demo_hitl
功能描述：演示 Device Agent 的 HITL（Human-In-The-Loop）确认流程。
         纯后端实现，无需任何前端。展示 interrupt 暂停和 Command 恢复的完整链路。

使用方式：
    python scripts/agent/demo_hitl.py

执行序列：
    1. 用户说"把空调调到18度" → supervisor 路由到 device_agent
    2. device_agent 解析为 control_device 工具 → 风险 MEDIUM → interrupt 暂停
    3. 图暂停，返回确认提示给调用方
    4. 调用方模拟用户回复"确认" → Command(resume="确认") 恢复图
    5. device_agent 从 interrupt 处继续，带 user_confirmed=True 执行
    6. 执行完成 → supervisor → output_guard → memory_writer → END
"""

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from silver_pilot.agent import create_initial_state, initialize_agent


def main() -> None:
    print("=" * 60)
    print("HITL 确认流程 Demo")
    print("=" * 60)

    # 1. 构建图
    graph = initialize_agent(skip_rag=True)
    thread_config = {"configurable": {"thread_id": "hitl_demo_001"}}

    # 2. 第一次 invoke — 用户发出指令
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="把客厅空调调到18度")]

    print("\n[用户] 把客厅空调调到18度")
    print("[系统] 第一次 invoke，图开始执行...\n")

    # 图会执行到 device_agent 的 interrupt() 处暂停
    # invoke 返回的不是最终结果，而是中断时的 state 快照
    result = graph.invoke(state, config=thread_config)

    # 3. 检查是否有 interrupt（通过 get_state 查看）
    graph_state = graph.get_state(thread_config)

    # graph_state.next 告诉你图暂停在哪个节点
    # graph_state.tasks 里有 interrupt 抛出的数据
    if graph_state.next:
        print(f"[系统] 图暂停在节点: {graph_state.next}")

        # 从 tasks 中提取 interrupt 数据
        for task in graph_state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    interrupt_data = intr.value
                    print(f"[系统] 确认请求: {interrupt_data['message']}")
                    print(f"[系统] 风险等级: {interrupt_data['risk_level']}")
                    print(f"[系统] 工具: {interrupt_data['tool_name']}")

        # 4. 模拟用户确认
        user_reply = "确认"
        print(f"\n[用户] {user_reply}")
        print("[系统] 第二次 invoke，用 Command 恢复图...\n")

        # 用 Command(resume=回复) 恢复中断的图
        result = graph.invoke(Command(resume=user_reply), config=thread_config)

    # 5. 输出最终结果
    print(f"[助手] {result.get('final_response', '(无回复)')}")
    print(f"\n[调试] tool_results: {result.get('tool_results', [])}")
    print(f"[调试] safety_flags: {result.get('safety_flags', [])}")

    # ────────────────────────────────────
    # 演示拒绝场景
    # ────────────────────────────────────
    print("\n" + "=" * 60)
    print("场景 2: 用户拒绝")
    print("=" * 60)

    thread_config_2 = {"configurable": {"thread_id": "hitl_demo_002"}}
    state2 = create_initial_state()
    state2["messages"] = [HumanMessage(content="把卧室空调关掉")]

    print("\n[用户] 把卧室空调关掉")

    result2 = graph.invoke(state2, config=thread_config_2)
    graph_state_2 = graph.get_state(thread_config_2)

    if graph_state_2.next:
        for task in graph_state_2.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    print(f"[系统] {intr.value['message']}")

        user_reply = "不要"
        print(f"[用户] {user_reply}")

        result2 = graph.invoke(Command(resume=user_reply), config=thread_config_2)

    print(f"[助手] {result2.get('final_response', '(无回复)')}")


if __name__ == "__main__":
    main()
