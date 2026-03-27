from __future__ import annotations

from typing import Any

import pytest
from langgraph.types import Command

from silver_pilot.agent.nodes.supervisor import route_by_intent
from silver_pilot.server import app as server_app


def test_parallel_dispatch_groups_same_agent_type() -> None:
    """并行分发时，同类型意图应合并为单个 Send 分支。"""
    state: dict[str, Any] = {
        "current_agent": "parallel",
        "pending_intents": [
            {"type": "MEDICAL_QUERY", "sub_query": "阿司匹林怎么吃"},
            {"type": "DEVICE_CONTROL", "sub_query": "设置吃药提醒"},
            {"type": "MEDICAL_QUERY", "sub_query": "阿司匹林副作用"},
        ],
    }

    routed = route_by_intent(state)

    assert isinstance(routed, list)
    assert len(routed) == 2

    # LangGraph Send 对象包含 node 和 arg 字段
    nodes = {send.node for send in routed}
    assert nodes == {"medical_agent", "device_agent"}

    medical_send = next(send for send in routed if send.node == "medical_agent")
    assert "阿司匹林怎么吃" in medical_send.arg["current_sub_query"]
    assert "阿司匹林副作用" in medical_send.arg["current_sub_query"]


class _FakeGraph:
    def __init__(self) -> None:
        self.resume_value: str | None = None

    def stream(self, inp: Any, config: dict, stream_mode: str):
        assert stream_mode == "updates"
        assert isinstance(inp, Command)
        self.resume_value = inp.resume

        yield {
            "device_agent": {
                "tool_results": [{"tool_name": "control_device", "success": True}],
            }
        }
        yield {"response_synthesizer": {"sub_response": ["ok"]}}


def test_hitl_resume_stream_returns_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """_resume 应使用 Command(resume=...) 恢复，并返回后续节点事件。"""
    fake_graph = _FakeGraph()
    monkeypatch.setattr(server_app, "_graph", fake_graph)

    events = server_app._resume("确认", {"configurable": {"thread_id": "t_hitl"}})

    assert fake_graph.resume_value == "确认"
    assert len(events) == 2
    assert events[0][0] == "device_agent"
    assert events[1][0] == "response_synthesizer"


def test_timing_summary_parallel_wall_clock() -> None:
    """timing 摘要需满足并行段 max、串行段 sum 的口径。"""
    dbg: dict[str, Any] = {
        "pipeline": [
            {"name": "Perception", "time": "100ms", "parallel": False},
            {"name": "Medical Agent", "time": "2.0s", "parallel": True},
            {"name": "Device Agent", "time": "500ms", "parallel": True},
            {"name": "Output Guard", "time": "50ms", "parallel": False},
        ]
    }

    server_app._fill_timing_summary(dbg)

    timing = dbg["timing"]
    assert timing["total_ms_sum"] == pytest.approx(2650.0)
    assert timing["serial_ms_sum"] == pytest.approx(150.0)
    assert timing["parallel_wall_ms"] == pytest.approx(2000.0)
