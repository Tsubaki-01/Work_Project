"""
模块名称：demo_agent
功能描述：Agent 系统端到端使用示例。

使用方式：
    python scripts/agent/demo_agent.py
"""

from langchain_core.messages import HumanMessage

from silver_pilot.agent import create_initial_state, initialize_agent
from silver_pilot.agent.memory import UserProfileManager


def main() -> None:
    print("=" * 60)
    print("Silver Pilot Agent Demo")
    print("=" * 60)

    # ── 一行启动：初始化所有组件 + 构建图 ──
    graph = initialize_agent(skip_rag=True)  # 开发阶段跳过 RAG

    # ── 准备用户画像 ──
    manager = UserProfileManager()
    manager.update_profile(
        "elderly_demo",
        {
            "chronic_diseases": ["高血压"],
            "allergies": ["青霉素"],
            "current_medications": [{"name": "氨氯地平", "dosage": "5mg/日"}],
            "emergency_contacts": [{"name": "张三（儿子）", "phone": "13800138000"}],
        },
    )
    profile = manager.get_profile("elderly_demo")

    # ── 测试用例 ──
    test_cases = [
        ("设备控制", "明天早上7点提醒我吃降压药"),
        ("医疗咨询", "阿司匹林和华法林能一起吃吗？"),
        ("日常闲聊", "今天天气真好啊"),
    ]

    for idx, (name, query) in enumerate(test_cases):
        print(f"\n{'─' * 60}")
        print(f"测试 {idx + 1}: {name}")
        print(f"用户: {query}")

        state = create_initial_state()
        state["messages"] = [HumanMessage(content=query)]
        state["user_profile"] = profile

        try:
            result = graph.invoke(
                state,
                config={"configurable": {"thread_id": f"demo_{idx}"}},
            )
            print(f"助手: {result.get('final_response', '(无回复)')}")
        except Exception as e:
            print(f"❌ 失败: {e}")

    print(f"\n{'=' * 60}")
    print("Demo 完成")


if __name__ == "__main__":
    main()
