"""
模块名称：mcp_client
功能描述：MCP Client 封装，负责：
         1. 将 weather_server.py 以子进程方式启动（stdio transport）
         2. 通过 MCP 协议调用工具（call_tool）
         3. 提供同步接口，与现有同步执行器无缝集成

设计要点：
    - 每次 call_tool 在独立线程 + 独立 Event Loop 中运行，
      避免与 uvicorn/FastAPI 的事件循环产生冲突
    - 通过模块级 _mcp_client 单例和 set_mcp_client() 实现依赖注入，
      与 ToolExecutor、RAGPipeline 的注入风格保持一致
"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

# ================= 日志 =================
logger = get_channel_logger(config.LOG_DIR / "agent", "mcp_client")

# MCP Server 脚本的绝对路径
_SERVER_SCRIPT: Path = Path(__file__).resolve().parent / "weather_server.py"

# MCP 工具集合（由本 Client 负责路由的工具名）
MCP_TOOL_NAMES: frozenset[str] = frozenset({"query_weather", "weather_forecast"})


# ────────────────────────────────────────────────────────────
# MCPClient
# ────────────────────────────────────────────────────────────


class MCPClient:
    """
    MCP stdio Client 封装。

    通信流程（每次 call_tool 独立完成握手）：
        1. ThreadPoolExecutor 在新线程中运行 asyncio.new_event_loop()
        2. stdio_client 启动 weather_server.py 子进程
        3. ClientSession.initialize() 完成 MCP 握手（获取 Server Capabilities）
        4. ClientSession.call_tool() 发送 JSON-RPC tools/call 请求
        5. 解析并返回结果，子进程随上下文管理器退出

    为什么每次调用重新建立连接？
        - weather_server 是无状态服务，重连开销可接受（实测约 200ms）
        - 避免长连接在 uvicorn reload 后出现僵尸进程

    使用示例::

        client = MCPClient()
        result = client.call_tool("query_weather", {"location": "上海"})
        print(result)  # {"status": "success", "weather": "晴", ...}
    """

    def __init__(self) -> None:
        if not _SERVER_SCRIPT.exists():
            raise FileNotFoundError(f"MCP Server 脚本不存在: {_SERVER_SCRIPT}")
        logger.info(f"MCPClient 初始化完成 | server={_SERVER_SCRIPT}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        同步调用 MCP 工具。

        Args:
            tool_name: 工具名称，需在 MCP_TOOL_NAMES 中
            arguments: 工具参数字典

        Returns:
            工具返回的结构化字典

        Raises:
            ValueError: 工具名不在白名单中
            RuntimeError: MCP 调用失败
        """
        if tool_name not in MCP_TOOL_NAMES:
            raise ValueError(f"MCPClient 不支持工具: {tool_name}，支持: {MCP_TOOL_NAMES}")

        logger.info(f"MCP 工具调用 | tool={tool_name} | args={arguments}")

        # 在独立线程中运行独立 Event Loop，避免与外部事件循环冲突
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_in_thread, tool_name, arguments)
            result = future.result(timeout=30)

        logger.info(f"MCP 工具返回 | tool={tool_name} | result={result}")
        return result

    def _run_in_thread(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """在新线程的新 Event Loop 中运行异步调用。"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._call_tool_async(tool_name, arguments))
        finally:
            loop.close()

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        异步核心：建立 stdio Client Session 并调用工具。

        MCP 握手流程：
            Client                    Server
              |  --- initialize() --->  |   （协议版本协商、Capabilities 交换）
              |  <-- initialized  ---   |
              |  --- tools/call() -->   |   （工具调用 JSON-RPC 请求）
              |  <-- result       ---   |   （工具执行结果）
        """
        server_params = StdioServerParameters(
            command=sys.executable,  # 使用当前 Python 解释器，确保虚拟环境一致
            args=[str(_SERVER_SCRIPT)],
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # MCP 握手：Client 声明协议版本，Server 返回支持的工具列表
                await session.initialize()

                # 工具说明
                # TODO：目前没有用上，后续修改代码逻辑与prompt为自动读取动态加载
                tools = await session.list_tools()
                logger.info(f"MCP 工具列表 | tools={tools}")

                # 调用工具：发送 JSON-RPC tools/call 请求
                result = await session.call_tool(tool_name, arguments)

                # result.content 是 list[TextContent | ImageContent | EmbeddedResource]
                # 天气工具返回 TextContent，其 text 是 JSON 字符串
                if result.content:
                    import json

                    # TODO: 增加对 ImageContent 和 EmbeddedResource 的处理
                    if isinstance(result.content[0], TextContent):
                        raw = result.content[0].text
                        try:
                            return json.loads(raw)
                        except (json.JSONDecodeError, AttributeError):
                            return {"status": "success", "raw": raw}

                return {"status": "empty_response"}
