import logging

from langchain_core.messages import HumanMessage

from silver_pilot.agent.nodes.perception_router import perception_router_node
from silver_pilot.config import config

# 配置日志输出到控制台方便查看
logging.basicConfig(level=logging.INFO)


def main() -> None:
    current_dir = config.SCRIPTS_DIR / "perception"

    audio_path = current_dir / "test_audio_1.mp3"
    img_path1 = current_dir / "test_img_1.jpg"
    img_path2 = current_dir / "test_img_2.jpg"
    img_path3 = current_dir / "test_img_3.jpg"

    state = [
        {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "你好"},
                        {"type": "audio", "audio": str(audio_path)},
                    ]
                )
            ]
        },
        {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "这怎么用"},
                        {"type": "image_url", "image_url": str(img_path1)},
                    ]
                )
            ]
        },
        {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "这是什么"},
                        {"type": "image_url", "image_url": str(img_path2)},
                    ]
                )
            ]
        },
        {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "这药怎么吃"},
                        {"type": "image_url", "image_url": str(img_path3)},
                    ]
                )
            ]
        },
    ]

    print("====================================")
    print("Testing perception_router_node...")
    print("====================================")

    try:
        for i, s in enumerate(state):
            print(f"\n====== State {i + 1} ======")
            result = perception_router_node(s)
            print("\n\n====== Perception Router Result ======")
            for key, value in result.items():
                print(f"\n[{key}]:")
                if key == "messages":
                    for msg in value:
                        print(f"  Type: {type(msg).__name__}")
                        print(
                            f"  Content:\n{max(msg.content[:500], msg.content) if isinstance(msg.content, str) else msg.content}..."
                        )
                elif isinstance(value, dict):
                    import json

                    print(f"  {json.dumps(value, indent=2, ensure_ascii=False)}")
                else:
                    print(f"  {value}")
            print("====================================\n")

    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
