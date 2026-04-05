"""
模块名称：weather_server
功能描述：基于 FastMCP 实现的天气查询 MCP Server。
         对外暴露两个工具：query_weather（当日天气）和 weather_forecast（多日预报）。
         使用 Open-Meteo 免费 API（无需 Key），通过 stdio 传输协议与 MCP Client 通信。

运行方式（由 MCPClient 自动启动，也可单独调试）：
    python weather_server.py

MCP 协议要点：
    - Server 通过 stdin 接收 JSON-RPC 2.0 格式的请求
    - 将执行结果通过 stdout 返回给 Client
    - stderr 用于日志输出，不影响协议通信
"""

from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ────────────────────────────────────────────────────────────
# FastMCP 实例：相当于 Flask 的 app = Flask(__name__)
# name 是 Server 的标识符，Client 在 initialize 握手时可获取它
# ────────────────────────────────────────────────────────────
mcp = FastMCP("silver-pilot-weather")

# ────────────────────────────────────────────────────────────
# Open-Meteo API 端点（完全免费，无需注册、无需 API Key）
# ────────────────────────────────────────────────────────────
_GEO_API = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_API = "https://api.open-meteo.com/v1/forecast"

# WMO 天气状态码 → 中文描述（WMO Weather Interpretation Codes）
_WMO_CODE: dict[int, str] = {
    0: "晴",
    1: "基本晴",
    2: "部分多云",
    3: "阴",
    45: "有雾",
    48: "结冰雾",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    95: "雷暴",
    99: "强雷暴伴冰雹",
}


# ────────────────────────────────────────────────────────────
# 私有辅助函数
# ────────────────────────────────────────────────────────────


async def _geocode(city: str) -> tuple[float, float]:
    """
    城市名 → (纬度, 经度)。
    调用 Open-Meteo Geocoding API，支持中文城市名。
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            _GEO_API,
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results")
    if not results:
        raise ValueError(f"找不到城市: {city}")

    return results[0]["latitude"], results[0]["longitude"]


async def _fetch_current_weather(lat: float, lon: float) -> dict[str, Any]:
    """获取当前天气（current_weather 块）。"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            _WEATHER_API,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _fetch_forecast(lat: float, lon: float, days: int) -> dict[str, Any]:
    """获取多日天气预报（daily 块）。"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            _WEATHER_API,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,weathercode",
                "timezone": "auto",
                "forecast_days": days,
            },
        )
        resp.raise_for_status()
        return resp.json()


# ────────────────────────────────────────────────────────────
# MCP 工具定义
# @mcp.tool() 将普通 async 函数注册为 MCP 工具：
#   - 函数签名自动生成 JSON Schema（供 Client 发现）
#   - docstring 成为工具描述（LLM 用来理解工具用途）
# ────────────────────────────────────────────────────────────


@mcp.tool()
async def query_weather(location: str, date: str = "today") -> dict[str, Any]:
    """
    查询指定城市的当日实时天气。

    Args:
        location: 城市名称，支持中文，如 '上海'、'北京'
        date: 查询日期，当前版本固定返回实时数据（today）

    Returns:
        包含城市、温度、天气状态、风速的字典
    """
    lat, lon = await _geocode(location)
    data = await _fetch_current_weather(lat, lon)

    cw = data.get("current_weather", {})
    wmo = int(cw.get("weathercode", 0))
    weather_desc = _WMO_CODE.get(wmo, f"未知({wmo})")

    return {
        "status": "success",
        "location": location,
        "weather": weather_desc,
        "temperature": f"{cw.get('temperature', 'N/A')}°C",
        "wind_speed": f"{cw.get('windspeed', 'N/A')} km/h",
        "source": "Open-Meteo",
    }


@mcp.tool()
async def weather_forecast(location: str, days: int = 3) -> dict[str, Any]:
    """
    查询指定城市的多日天气预报。

    Args:
        location: 城市名称，支持中文，如 '广州'
        days: 预报天数，范围 1-7，默认 3 天

    Returns:
        包含逐日最高/最低温度和天气状态的字典
    """
    days = max(1, min(days, 7))  # 限制在 1-7 天
    lat, lon = await _geocode(location)
    data = await _fetch_forecast(lat, lon, days)

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    codes = daily.get("weathercode", [])

    forecast_list = [
        {
            "date": dates[i],
            "weather": _WMO_CODE.get(int(codes[i]), f"未知({int(codes[i])})"),
            "temp_max": f"{max_temps[i]}°C",
            "temp_min": f"{min_temps[i]}°C",
        }
        for i in range(len(dates))
    ]

    return {
        "status": "success",
        "location": location,
        "forecast": forecast_list,
        "source": "Open-Meteo",
    }


# ────────────────────────────────────────────────────────────
# 入口：以 stdio 传输模式启动 MCP Server
# stdio transport 含义：
#   - Client 通过 subprocess.Popen 启动本脚本
#   - 双方通过 stdin/stdout 交换 JSON-RPC 2.0 消息
#   - 每行一个 JSON 对象（换行分隔）
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
