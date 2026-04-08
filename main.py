"""
项目入口文件：负责启动 FastAPI 服务。
"""

import logging
import sys

from app.api.services import LegalApiService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def main():
    print("=" * 60)
    print("  民商事司法智能体 (MVP)")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()

    service = LegalApiService()
    state = None

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见")
            break

        try:
            result = service.chat(user_input, state)
            state = result
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    print(f"\n系统: {last.content}\n")
                elif isinstance(last, dict):
                    print(f"\n系统: {last.get('content', '')}\n")
                else:
                    print(f"\n系统: {last}\n")
        except Exception as e:
            logger.error("执行出错: %s", e, exc_info=True)
            print(f"\n系统错误: {e}\n")


if __name__ == "__main__":
    main()
