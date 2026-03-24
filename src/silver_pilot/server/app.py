"""
模块名称：app
功能描述：Silver Pilot FastAPI 服务端。

修复：graph.stream() 用 asyncio.to_thread() 包装，避免阻塞 async 事件循环。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from .commit_review import build_commit_review_report
from .models import (
    CommitReviewRequest,
    CommitReviewResponse,
    HealthOverview,
    MessageRecord,
    ReminderItem,
    SessionCreate,
    SessionMeta,
    WSIncoming,
    WSOutgoing,
)
from .session_store import SessionStore

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if not STATIC_DIR.exists():
    STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

_graph: CompiledStateGraph | None = None
_store = SessionStore()

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


@asynccontextmanager
async def lifespan(application: FastAPI):  # type: ignore[no-untyped-def]
    global _graph
    if DEMO_MODE:
        print("=" * 50 + "\n  DEMO_MODE — 跳过 Agent 初始化\n" + "=" * 50)
    else:
        print("正在初始化 Agent 系统...")
        try:
            from silver_pilot.agent import initialize_agent

            _graph = initialize_agent(skip_rag=False)
            print("Agent 系统初始化完成")
        except Exception as e:
            print(f"Agent 初始化失败: {e}")
            traceback.print_exc()
            print("自动回退到 mock 响应")
    if not _store.list_sessions():
        _store.create("欢迎对话", user_id="default_user")
    yield
    print("Server 关闭")


app = FastAPI(title="Silver Pilot API", version="0.1.0", lifespan=lifespan)
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


# ── REST ──
@app.get("/api/sessions", response_model=list[SessionMeta])
async def list_sessions(user_id: str = "default_user") -> list[SessionMeta]:
    return _store.list_sessions(user_id)


@app.post("/api/sessions", response_model=SessionMeta)
async def create_session(req: SessionCreate) -> SessionMeta:
    return _store.create(name=req.name, user_id=req.user_id)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, bool]:
    return {"deleted": _store.delete(session_id)}


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageRecord])
async def get_messages(session_id: str) -> list[MessageRecord]:
    return _store.get_messages(session_id)


@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str) -> dict[str, Any]:
    if DEMO_MODE or _graph is None:
        return _DEMO_PROFILE
    try:
        from silver_pilot.agent.memory.user_profile import UserProfileManager

        return UserProfileManager().get_profile(user_id)
    except Exception as e:
        return {**_DEMO_PROFILE, "_error": str(e)}


@app.get("/api/health/{user_id}", response_model=HealthOverview)
async def get_health(user_id: str) -> HealthOverview:
    return HealthOverview()


@app.get("/api/reminders/{user_id}", response_model=list[ReminderItem])
async def get_reminders(user_id: str) -> list[ReminderItem]:
    return [ReminderItem(**r) for r in _DEMO_REMINDERS]


@app.post("/api/commit-review", response_model=CommitReviewResponse)
async def commit_review(req: CommitReviewRequest) -> CommitReviewResponse:
    try:
        report = build_commit_review_report(REPO_ROOT, req.commit_hashes)
        return CommitReviewResponse(report=report)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ════════════════════════════════════════════════
# WebSocket 对话
# ════════════════════════════════════════════════


@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    if not _store.get(session_id):
        _store._sessions[session_id] = SessionMeta(session_id=session_id, name="新对话")
        _store._messages[session_id] = []
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


# ════ 核心：Agent 调用 (to_thread 修复) ════


async def _agent_response(ws: WebSocket, sid: str, inc: WSIncoming) -> None:
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
    dbg: dict[str, Any] = {"pipeline": [], "intents": [], "entities": [], "rag": None, "tools": []}

    try:
        print(f"[Agent] 处理: {inc.content[:50]}...")

        # ══ 关键：同步 graph.stream() 放入线程池 ══
        events = await asyncio.to_thread(_stream_collect, inp, cfg)
        print(f"[Agent] 收集 {len(events)} 个事件")

        for name, out, ms in events:
            await ws.send_text(WSOutgoing(type="node_start", node=name).model_dump_json())
            _fill_debug(name, out, dbg)
            await ws.send_text(
                WSOutgoing(
                    type="node_end", node=name, data=_safe(out), duration_ms=ms
                ).model_dump_json()
            )

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


def _stream_collect(inp: dict, cfg: dict) -> list[tuple[str, dict, float]]:
    """同步 — 线程池中执行。"""
    evts: list[tuple[str, dict, float]] = []
    assert _graph is not None
    for chunk in _graph.stream(inp, config=cfg, stream_mode="updates"):
        for n, o in chunk.items():
            t0 = time.time()
            ms = (time.time() - t0) * 1000
            evts.append((n, o if isinstance(o, dict) else {}, round(ms, 1)))
            print(f"  [stream] {n}")
    return evts


def _final_resp(cfg: dict) -> str:
    """同步 — 提取最终响应。"""
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


# ── HITL ──


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
            await ws.send_text(WSOutgoing(type="node_end", node=n).model_dump_json())
            _fill_debug(n, o, dbg)
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


# ── Demo ──

_DEMO_R = {
    "阿司匹林": {
        "pipeline": [
            {"name": "Perception", "time": "12ms"},
            {"name": "Supervisor", "time": "320ms"},
            {"name": "Medical Agent", "time": "1.8s"},
            {"name": "Output Guard", "time": "45ms"},
        ],
        "intents": [{"type": "MEDICAL_QUERY", "query": "阿司匹林的用法用量", "priority": 1}],
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
        "sources": ["知识图谱", "医学文献"],
    },
}
_DEMO_D = {
    "pipeline": [
        {"name": "Perception", "time": "10ms"},
        {"name": "Supervisor", "time": "300ms"},
        {"name": "Chat Agent", "time": "500ms"},
        {"name": "Output Guard", "time": "20ms"},
    ],
    "intents": [{"type": "CHITCHAT", "query": "日常对话", "priority": 3}],
    "entities": [],
    "rag": None,
    "tools": [],
    "response": "我在呢，有什么可以帮您的吗？",
    "sources": [],
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
            debug={
                "pipeline": d["pipeline"],
                "intents": d["intents"],
                "entities": d["entities"],
                "rag": d["rag"],
                "tools": d["tools"],
            },
        ).model_dump_json()
    )


# ── Utils ──


def _fill_debug(name: str, out: dict, dbg: dict) -> None:
    try:
        if name == "supervisor" and "pending_intents" in out:
            ca = out.get("current_agent", "")
            sq = out.get("current_sub_query", "")
            if ca:
                dbg["intents"].insert(
                    0,
                    {
                        "type": {
                            "medical": "MEDICAL_QUERY",
                            "device": "DEVICE_CONTROL",
                            "chat": "CHITCHAT",
                            "emergency": "EMERGENCY",
                        }.get(ca, "CHITCHAT"),
                        "query": sq,
                        "priority": 0,
                    },
                )
            for i in out.get("pending_intents", []):
                dbg["intents"].append(i)
        elif name == "medical_agent":
            if out.get("rag_context"):
                dbg["rag"] = {
                    "context_text": out.get("rag_context", ""),
                    "hallucination_score": out.get("hallucination_score", 0),
                    "verdict": "pass" if out.get("hallucination_score", 0) < 0.5 else "fail",
                }
            if out.get("linked_entities"):
                dbg["entities"] = out["linked_entities"]
        elif name == "device_agent" and out.get("tool_calls"):
            dbg["tools"] = out["tool_calls"]
        dbg["pipeline"].append({"name": name, "time": "..."})
    except Exception:
        pass


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
                except:
                    r[k] = str(v)
        return r


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("silver_pilot.server.app:app", host="0.0.0.0", port=8080, reload=True)
