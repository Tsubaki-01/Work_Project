"""
模块名称：app
功能描述：Silver Pilot FastAPI 服务端。

变更说明：
    - 数据层切换至 Redis（RedisStore 同时管理会话和用户画像）
    - WebSocket 事件流修复：正确的节点计时、完整的 debug 数据构建
    - 前端 Pipeline 可视化数据与后端 Agent 真实执行过程联动
    - 降级策略：Redis 不可用时回退到内存 SessionStore
"""

import asyncio
import json
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from .models import (
    HealthOverview,
    MessageRecord,
    ReminderItem,
    SessionCreate,
    SessionMeta,
    WSIncoming,
    WSOutgoing,
)

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

_graph: CompiledStateGraph | None = None
_store: Any = None  # RedisStore or SessionStore

# ── 节点名 → 前端展示名映射 ──
NODE_DISPLAY_NAMES: dict[str, str] = {
    "perception_router": "Perception",
    "supervisor": "Supervisor",
    "medical_agent": "Medical Agent",
    "device_agent": "Device Agent",
    "chat_agent": "Chat Agent",
    "emergency_agent": "Emergency Agent",
    "response_synthesizer": "Synthesizer",
    "output_guard": "Output Guard",
    "memory_writer": "Memory Writer",
}

NODE_COLORS: dict[str, str] = {
    "Perception": "var(--n-per)",
    "Supervisor": "var(--n-sup)",
    "Medical Agent": "var(--n-med)",
    "Device Agent": "var(--n-dev)",
    "Chat Agent": "var(--yellow)",
    "Emergency Agent": "var(--red)",
    "Synthesizer": "var(--accent)",
    "Output Guard": "var(--n-grd)",
    "Memory Writer": "var(--text-hint)",
}

INTENT_COLORS: dict[str, str] = {
    "MEDICAL_QUERY": "var(--n-med)",
    "DEVICE_CONTROL": "var(--n-dev)",
    "CHITCHAT": "var(--yellow)",
    "EMERGENCY": "var(--red)",
}

INTENT_MAP: dict[str, str] = {
    "medical": "MEDICAL_QUERY",
    "device": "DEVICE_CONTROL",
    "chat": "CHITCHAT",
    "emergency": "EMERGENCY",
}

# ── Demo 数据 ──
_DEMO_REMINDERS = [
    {
        "id": "r1",
        "time": "07:00",
        "message": "吃阿司匹林",
        "repeat": "每天",
        "active": True,
        "done": True,
    },
    {
        "id": "r2",
        "time": "08:00",
        "message": "测血糖",
        "repeat": "每天",
        "active": True,
        "done": True,
    },
    {
        "id": "r3",
        "time": "09:30",
        "message": "晨练散步30分钟",
        "repeat": "每天",
        "active": True,
        "done": True,
    },
    {
        "id": "r4",
        "time": "12:00",
        "message": "吃二甲双胍",
        "repeat": "每天",
        "active": True,
        "done": False,
    },
    {
        "id": "r5",
        "time": "15:00",
        "message": "去社区医院复查",
        "repeat": "不重复",
        "active": True,
        "done": False,
    },
    {
        "id": "r6",
        "time": "18:00",
        "message": "吃二甲双胍",
        "repeat": "每天",
        "active": True,
        "done": False,
    },
]
_DEMO_PROFILE = {
    "user_id": "default_user",
    "name": "王爷爷",
    "age": 72,
    "chronic_diseases": ["高血压", "糖尿病"],
    "allergies": ["青霉素"],
    "current_medications": [
        {"name": "阿司匹林", "dosage": "100mg/日"},
        {"name": "二甲双胍", "dosage": "500mg/次, 每日2次"},
        {"name": "氨氯地平", "dosage": "5mg/日"},
    ],
    "emergency_contacts": [
        {"name": "王小明（儿子）", "phone": "138****1234"},
        {"name": "李阿姨（邻居）", "phone": "139****5678"},
    ],
}


def _init_store() -> Any:
    """初始化数据存储层：优先 Redis，失败回退到内存。"""
    try:
        from .redis_store import RedisStore

        store = RedisStore()
        print("✓ 使用 Redis 数据存储")
        return store
    except Exception as e:
        print(f"✗ Redis 不可用 ({e})，回退到内存存储")
        from .session_store import SessionStore

        return SessionStore()


@asynccontextmanager
async def lifespan(application: FastAPI):  # type: ignore[no-untyped-def]
    global _graph, _store

    # 初始化存储层
    _store = _init_store()

    if DEMO_MODE:
        print("=" * 50 + "\n  DEMO_MODE — 跳过 Agent 初始化\n" + "=" * 50)
    else:
        print("正在初始化 Agent 系统...")
        try:
            from silver_pilot.agent import initialize_agent

            # 如果是 RedisStore，直接传给 bootstrap 作为 profile_manager
            profile_mgr = None
            from .redis_store import RedisStore

            if isinstance(_store, RedisStore):
                profile_mgr = _store

            _graph = initialize_agent(skip_rag=False, profile_manager=profile_mgr)
            print("Agent 系统初始化完成")
        except Exception as e:
            print(f"Agent 初始化失败: {e}")
            traceback.print_exc()
            print("自动回退到 mock 响应")

    # 确保有默认会话
    if not _store.list_sessions():
        _store.create_session("欢迎对话", user_id="default_user")

    yield
    print("Server 关闭")


app = FastAPI(title="Silver Pilot API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def serve_index() -> FileResponse | JSONResponse:
    p = STATIC_DIR / "index.html"
    return (
        FileResponse(p)
        if p.exists()
        else JSONResponse({"error": "请将 index.html 放入 static/"}, 404)
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ═══════════════════════════════════════
#  REST API
# ═══════════════════════════════════════


@app.get("/api/sessions", response_model=list[SessionMeta])
async def list_sessions(user_id: str = "default_user") -> list[SessionMeta]:
    return _store.list_sessions(user_id)


@app.post("/api/sessions", response_model=SessionMeta)
async def create_session(req: SessionCreate) -> SessionMeta:
    return _store.create_session(name=req.name, user_id=req.user_id)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, bool]:
    return {"deleted": _store.delete_session(session_id)}


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageRecord])
async def get_messages(session_id: str) -> list[MessageRecord]:
    return _store.get_messages(session_id)


@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str) -> dict[str, Any]:
    if DEMO_MODE or _graph is None:
        return _DEMO_PROFILE
    try:
        return _store.get_profile(user_id)
    except Exception as e:
        return {**_DEMO_PROFILE, "_error": str(e)}


@app.get("/api/health/{user_id}", response_model=HealthOverview)
async def get_health(user_id: str) -> HealthOverview:
    return HealthOverview()


@app.get("/api/reminders/{user_id}", response_model=list[ReminderItem])
async def get_reminders(user_id: str) -> list[ReminderItem]:
    return [ReminderItem(**r) for r in _DEMO_REMINDERS]


# ═══════════════════════════════════════
#  WebSocket 对话
# ═══════════════════════════════════════


@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    # 确保 session 存在
    if not _store.get_session(session_id):
        _store.create_session("新对话", user_id="default_user")
    try:
        while True:
            raw = await websocket.receive_text()
            incoming = WSIncoming.model_validate_json(raw)
            if incoming.type == "message":
                await _handle_chat(websocket, session_id, incoming)
    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_text(WSOutgoing(type="error", message=str(e)).model_dump_json())
        except Exception:
            pass


async def _handle_chat(ws: WebSocket, sid: str, inc: WSIncoming) -> None:
    _store.add_message(sid, MessageRecord(role="user", content=inc.content))
    if DEMO_MODE or _graph is None:
        await _demo_response(ws, inc)
        return
    await _agent_response(ws, sid, inc)


# ═══════════════════════════════════════
#  核心 Agent 调用（修复版）
# ═══════════════════════════════════════


async def _agent_response(ws: WebSocket, sid: str, inc: WSIncoming) -> None:
    """处理真实 Agent 调用，逐节点发送 WS 事件。"""
    # 构建多模态消息
    mc: str | list[dict] = inc.content
    if inc.modality.get("image") and inc.image_path:
        mc = [
            {"type": "text", "text": inc.content},
            {"type": "image_url", "image_url": inc.image_path},
        ]
    elif inc.modality.get("audio") and inc.audio_path:
        mc = [{"type": "audio_url", "audio_url": inc.audio_path}]
        if inc.content:
            mc.insert(0, {"type": "text", "text": inc.content})

    cfg = {"configurable": {"thread_id": sid}}
    inp = {"messages": [HumanMessage(content=mc)]}

    # debug 数据结构（前端 DD 对象）
    dbg: dict[str, Any] = {
        "pipeline": [],
        "intents": [],
        "entities": [],
        "rag": None,
        "tools": [],
        "perception": None,
    }

    try:
        print(f"[Agent] 处理: {inc.content[:50]}...")

        # 逐节点实时流式处理：astream 每完成一个节点就 yield 一个 chunk，
        # 立即发送 node_start / node_end，前端可实时看到节点推进。
        # stream_mode="updates" 在节点执行完毕后才 yield，因此 node_start 用于
        # 前端动画触发，duration_ms 通过相邻 chunk 到达时间差近似估算节点耗时。
        assert _graph is not None
        node_count = 0
        t_prev_chunk = time.perf_counter()
        async for chunk in _graph.astream(inp, config=cfg, stream_mode="updates"):
            t_chunk_arrived = time.perf_counter()
            for node_name, node_output in chunk.items():
                display_name = NODE_DISPLAY_NAMES.get(node_name, node_name)
                color = NODE_COLORS.get(display_name, "var(--text-sub)")
                node_output = node_output if isinstance(node_output, dict) else {}

                # 发送 node_start（驱动前端动画；节点实际已执行完毕）
                await ws.send_text(
                    WSOutgoing(type="node_start", node=display_name).model_dump_json()
                )

                # 构建 debug 数据
                _fill_debug(node_name, node_output, dbg)

                # 用相邻 chunk 到达时间差近似节点耗时
                duration_ms = (t_chunk_arrived - t_prev_chunk) * 1000
                time_str = (
                    f"{duration_ms:.0f}ms" if duration_ms < 1000 else f"{duration_ms / 1000:.1f}s"
                )
                dbg["pipeline"].append(
                    {
                        "name": display_name,
                        "color": color,
                        "time": time_str,
                        "status": "done",
                    }
                )

                # 发送 node_end（包含耗时）
                await ws.send_text(
                    WSOutgoing(
                        type="node_end",
                        node=display_name,
                        data=_safe(node_output),
                        duration_ms=round(duration_ms, 1),
                    ).model_dump_json()
                )

                node_count += 1
                print(f"  [stream] {node_name}")

            t_prev_chunk = time.perf_counter()

        print(f"[Agent] 共 {node_count} 个节点事件")

        # 提取最终响应
        resp = await asyncio.to_thread(_final_resp, cfg)
        print(f"[Agent] 响应: {resp[:80]}...")

        _store.add_message(sid, MessageRecord(role="assistant", content=resp))
        await ws.send_text(WSOutgoing(type="response", content=resp, debug=dbg).model_dump_json())

    except Exception as e:
        nm = type(e).__name__
        if "GraphInterrupt" in nm:
            print("[Agent] HITL interrupt")
            await _hitl(ws, sid, cfg, dbg)
        else:
            traceback.print_exc()
            await ws.send_text(WSOutgoing(type="error", message=f"{nm}: {e}").model_dump_json())


def _final_resp(cfg: dict) -> str:
    """提取最终响应文本。"""
    assert _graph is not None
    st = _graph.get_state(cfg)
    v = st.values
    r = v.get("final_response", "")
    if r:
        return r
    for m in reversed(v.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return "抱歉，我暂时无法回答。"


# ═══════════════════════════════════════
#  Debug 数据构建
# ═══════════════════════════════════════


def _fill_debug(name: str, out: dict, dbg: dict) -> None:
    """从节点输出中提取前端 debug drawer 需要的数据。"""
    try:
        if name == "perception_router":
            img_ctx = out.get("current_image_context", "")
            audio_ctx = out.get("current_audio_context", "")
            if img_ctx or audio_ctx:
                dbg["perception"] = {
                    "image_context": img_ctx,
                    "audio_context": audio_ctx,
                    "emotion": out.get("user_emotion", "NEUTRAL"),
                    "modality": out.get("input_modality", {}),
                }

        elif name == "supervisor":
            ca = out.get("current_agent", "")
            sq = out.get("current_sub_query", "")
            rl = out.get("risk_level", "low")
            if ca and ca != "done":
                intent_type = INTENT_MAP.get(ca, "CHITCHAT")
                dbg["intents"].insert(
                    0,
                    {
                        "type": intent_type,
                        "query": sq,
                        "priority": 0,
                        "color": INTENT_COLORS.get(intent_type, "var(--text-sub)"),
                    },
                )
            for i in out.get("pending_intents", []):
                itype = i.get("type", "CHITCHAT")
                dbg["intents"].append(
                    {
                        "type": itype,
                        "query": i.get("sub_query", ""),
                        "priority": i.get("priority", 9),
                        "color": INTENT_COLORS.get(itype, "var(--text-sub)"),
                    }
                )

        elif name == "medical_agent":
            if out.get("rag_context"):
                dbg["rag"] = {
                    "query_rewrite": "",  # 来自 pipeline 内部，暂用空
                    "context_text": out.get("rag_context", ""),
                    "graph_results": [],
                    "vector_results": [],
                    "hallucination_score": out.get("hallucination_score", 0),
                    "verdict": "pass" if out.get("hallucination_score", 0) < 0.5 else "fail",
                }
            if out.get("linked_entities"):
                ents = out["linked_entities"]
                dbg["entities"] = [
                    {
                        "name": e.get("original_name", e.get("name", "")),
                        "label": e.get("label", "Unknown"),
                        "linked": e.get("is_linked", False),
                        "neo4j_name": e.get("neo4j_name", ""),
                        "score": e.get("similarity_score", 0),
                    }
                    for e in (ents if isinstance(ents, list) else [])
                ]

        elif name == "device_agent":
            if out.get("tool_calls"):
                for tc in out["tool_calls"]:
                    tool_results = out.get("tool_results", [])
                    matching_result = next(  # type: ignore[var-annotated]
                        (r for r in tool_results if r.get("tool_name") == tc.get("tool_name")),
                        {},
                    )
                    dbg["tools"].append(
                        {
                            "name": tc.get("tool_name", "unknown"),
                            "args": tc.get("arguments", {}),
                            "risk": matching_result.get("risk_level", "low"),
                            "needs_confirmation": matching_result.get("needs_confirmation", False),
                            "result": matching_result.get("result"),
                            "confirmation_message": (
                                matching_result.get("result", {}).get("confirmation_message", "")
                                if matching_result.get("needs_confirmation")
                                else ""
                            ),
                        }
                    )

    except Exception as e:
        print(f"[debug] fill error for {name}: {e}")


# ═══════════════════════════════════════
#  HITL 中断处理
# ═══════════════════════════════════════


async def _hitl(ws: WebSocket, sid: str, cfg: dict, dbg: dict) -> None:
    idata = await asyncio.to_thread(_get_interrupt, cfg)
    print(f"[HITL] data: {idata}")
    await ws.send_text(WSOutgoing(type="hitl_request", data=idata).model_dump_json())

    raw = await ws.receive_text()
    resp = WSIncoming.model_validate_json(raw)
    rv = "确认" if resp.confirmed else "取消"
    print(f"[HITL] 用户: {rv}")

    try:
        evts = await asyncio.to_thread(_resume, rv, cfg)
        for n, o, _ in evts:
            display = NODE_DISPLAY_NAMES.get(n, n)
            await ws.send_text(WSOutgoing(type="node_end", node=display).model_dump_json())
            _fill_debug(n, o, dbg)
            color = NODE_COLORS.get(display, "var(--text-sub)")
            dbg["pipeline"].append(
                {"name": display, "color": color, "time": "...", "status": "done"}
            )
        fr = await asyncio.to_thread(_final_resp, cfg)
        _store.add_message(sid, MessageRecord(role="assistant", content=fr))
        await ws.send_text(WSOutgoing(type="response", content=fr, debug=dbg).model_dump_json())
    except Exception as e:
        traceback.print_exc()
        await ws.send_text(WSOutgoing(type="error", message=f"HITL 失败: {e}").model_dump_json())


def _get_interrupt(cfg: dict) -> dict:
    assert _graph is not None
    st = _graph.get_state(cfg)
    for t in st.tasks:
        if hasattr(t, "interrupts") and t.interrupts:
            v = t.interrupts[0].value
            return v if isinstance(v, dict) else {"message": str(v)}
    return {}


def _resume(rv: str, cfg: dict) -> list[tuple[str, dict, float]]:
    evts: list[tuple[str, dict, float]] = []
    assert _graph is not None
    for chunk in _graph.stream(Command(resume=rv), config=cfg, stream_mode="updates"):
        for n, o in chunk.items():
            evts.append((n, o if isinstance(o, dict) else {}, 0))
    return evts


# ═══════════════════════════════════════
#  Demo 响应（保留完整的 demo 模式）
# ═══════════════════════════════════════

_DEMO_R = {
    "阿司匹林": {
        "pipeline": [
            {"name": "Perception", "time": "12ms", "color": "var(--n-per)", "status": "done"},
            {"name": "Supervisor", "time": "320ms", "color": "var(--n-sup)", "status": "done"},
            {"name": "Medical Agent", "time": "1.8s", "color": "var(--n-med)", "status": "done"},
            {"name": "Synthesizer", "time": "200ms", "color": "var(--accent)", "status": "done"},
            {"name": "Output Guard", "time": "45ms", "color": "var(--n-grd)", "status": "done"},
        ],
        "intents": [
            {
                "type": "MEDICAL_QUERY",
                "query": "阿司匹林的用法用量",
                "priority": 1,
                "color": "var(--n-med)",
            }
        ],
        "entities": [
            {
                "name": "阿司匹林",
                "label": "Drug",
                "linked": True,
                "neo4j_name": "阿司匹林",
                "score": 1.0,
            }
        ],
        "rag": {
            "query_rewrite": "阿司匹林的用法用量及不良反应",
            "graph_results": [
                {
                    "content": "阿司匹林用法：口服，每日1次，100mg",
                    "source": "知识图谱",
                    "layer": "local_fact",
                    "score": 1.0,
                }
            ],
            "vector_results": [
                {"content": "阿司匹林 口服 成人一次0.3~0.6g", "source": "医学文献", "score": 0.91}
            ],
            "hallucination_score": 0.08,
            "verdict": "pass",
        },
        "tools": [],
        "response": "根据医学资料：\n\n**用法用量**\n口服，每日1次，每次100mg，饭后服用 [知识图谱]\n\n**常见副作用**\n· 胃肠道反应\n· 出血倾向\n\n（温馨提示：具体用药请遵医嘱。）",
    },
}

_DEMO_D = {
    "pipeline": [
        {"name": "Perception", "time": "10ms", "color": "var(--n-per)", "status": "done"},
        {"name": "Supervisor", "time": "300ms", "color": "var(--n-sup)", "status": "done"},
        {"name": "Chat Agent", "time": "500ms", "color": "var(--yellow)", "status": "done"},
        {"name": "Synthesizer", "time": "50ms", "color": "var(--accent)", "status": "done"},
        {"name": "Output Guard", "time": "20ms", "color": "var(--n-grd)", "status": "done"},
    ],
    "intents": [{"type": "CHITCHAT", "query": "日常对话", "priority": 3, "color": "var(--yellow)"}],
    "entities": [],
    "rag": None,
    "tools": [],
    "response": "我在呢，有什么可以帮您的吗？",
}


async def _demo_response(ws: WebSocket, inc: WSIncoming) -> None:
    d = _DEMO_D
    for kw, r in _DEMO_R.items():
        if kw in inc.content:
            d = r
            break
    for n in d["pipeline"]:  # type: ignore[attr-defined]
        await ws.send_text(WSOutgoing(type="node_start", node=n["name"]).model_dump_json())
        await asyncio.sleep(0.3)
        t = n["time"]
        ms = float(t.replace("ms", "").replace("s", "000")) if isinstance(t, str) else 0
        await ws.send_text(
            WSOutgoing(type="node_end", node=n["name"], duration_ms=ms).model_dump_json()
        )
    await ws.send_text(
        WSOutgoing(
            type="response",
            content=d["response"],
            debug={k: d.get(k) for k in ("pipeline", "intents", "entities", "rag", "tools")},
        ).model_dump_json()
    )


# ── Utils ──


def _safe(obj: Any) -> dict:
    try:
        json.dumps(obj)
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError):
        r = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    json.dumps(v)
                    r[k] = v
                except Exception:
                    r[k] = str(v)
        return r


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("silver_pilot.server.app:app", host="0.0.0.0", port=8080, reload=True)
