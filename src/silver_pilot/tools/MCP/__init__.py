"""
模块名称：tools.MCP
功能描述：MCP 工具子包，包含 FastMCP Server（weather_server）和同步 MCPClient。
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mcp_client import MCPClient

__all__ = ["MCPClient", "MCP_TOOL_NAMES"]


def __getattr__(name: str) -> Any:
    if name == "MCPClient":
        from .mcp_client import MCPClient

        return MCPClient
    if name == "MCP_TOOL_NAMES":
        from .mcp_client import MCP_TOOL_NAMES

        return MCP_TOOL_NAMES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
