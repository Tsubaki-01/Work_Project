/**
 * Silver Pilot — API Connector
 *
 * 将前端的 mock 数据替换为真实的后端 API 调用。
 *
 * 使用方式：在 silver-pilot-demo.html 的 </body> 前添加：
 *   <script src="/static/api-connector.js"></script>
 *
 * 功能：
 *   1. REST: 会话 CRUD、用户画像、健康数据、提醒
 *   2. WebSocket: 实时对话 + Agent 事件流
 *   3. 自动降级: 后端不可用时回退到本地 mock
 */

(function () {
  "use strict";

  const API_BASE = window.location.origin;
  const WS_BASE = API_BASE.replace("http", "ws");
  const USER_ID = (() => {
    try {
      const q = new URLSearchParams(window.location.search);
      const fromQuery = q.get("user_id");
      const fromWindow = window.SILVER_PILOT_USER_ID;
      const fromStorage = window.localStorage?.getItem("silver_pilot_user_id");
      return String(fromQuery || fromWindow || fromStorage || "default_user");
    } catch {
      return "default_user";
    }
  })();

  let _ws = null;
  let _wsSessionId = null;
  let _activeSessionId = null;
  let _connected = false;
  let _pendingDebug = null; // 累积的 debug 数据
  let _parallelFlow = false;
  let _pipelineInsertCounter = 0;
  let _hitlPending = false;
  const _orderingUtil = (window.PipelineOrdering && typeof window.PipelineOrdering.normalizePipelineOrder === "function")
    ? window.PipelineOrdering
    : {
      groupRank: (groupId) => {
        if (!groupId || typeof groupId !== "string") return 99;
        if (groupId.startsWith("0-")) return 0;
        if (groupId.startsWith("1-")) return 1;
        if (groupId.startsWith("2-")) return 2;
        return 99;
      },
      normalizePipelineOrder: (pipeline) => {
        if (!Array.isArray(pipeline)) return [];
        pipeline.sort((a, b) => {
          const ga = (!a || typeof a.group_id !== "string") ? 99 : (a.group_id.startsWith("0-") ? 0 : (a.group_id.startsWith("1-") ? 1 : (a.group_id.startsWith("2-") ? 2 : 99)));
          const gb = (!b || typeof b.group_id !== "string") ? 99 : (b.group_id.startsWith("0-") ? 0 : (b.group_id.startsWith("1-") ? 1 : (b.group_id.startsWith("2-") ? 2 : 99)));
          if (ga !== gb) return ga - gb;

          const sa = Number.isFinite(a?.event_seq) ? a.event_seq : Number.MAX_SAFE_INTEGER;
          const sb = Number.isFinite(b?.event_seq) ? b.event_seq : Number.MAX_SAFE_INTEGER;
          if (sa !== sb) return sa - sb;

          const ia = Number.isFinite(a?._insert_idx) ? a._insert_idx : Number.MAX_SAFE_INTEGER;
          const ib = Number.isFinite(b?._insert_idx) ? b._insert_idx : Number.MAX_SAFE_INTEGER;
          return ia - ib;
        });
        return pipeline;
      },
    };

  function _findLatestPipelineNode(nodeName, preferActive = false) {
    if (!_pendingDebug || !_pendingDebug.pipeline) return null;
    for (let i = _pendingDebug.pipeline.length - 1; i >= 0; i -= 1) {
      const n = _pendingDebug.pipeline[i];
      if (n.name !== nodeName) continue;
      if (!preferActive || n.status === "active") return n;
    }
    return null;
  }

  function _normalizePipelineOrder() {
    if (!_pendingDebug || !Array.isArray(_pendingDebug.pipeline)) return;
    _orderingUtil.normalizePipelineOrder(_pendingDebug.pipeline);
  }

  function _setInputLock(locked) {
    _hitlPending = !!locked;

    const field = document.getElementById("iField");
    const sendBtn = document.getElementById("sBtn");

    if (field) {
      if (_hitlPending) {
        if (!field.dataset.prevPlaceholder) {
          field.dataset.prevPlaceholder = field.placeholder || "";
        }
        field.placeholder = "等待确认中，请先完成弹窗操作...";
      } else if (field.dataset.prevPlaceholder) {
        field.placeholder = field.dataset.prevPlaceholder;
      }
      field.disabled = _hitlPending;
    }

    if (sendBtn) {
      sendBtn.disabled = _hitlPending;
      sendBtn.style.opacity = _hitlPending ? "0.55" : "";
      sendBtn.style.cursor = _hitlPending ? "not-allowed" : "";
      sendBtn.title = _hitlPending ? "等待 HITL 确认中" : "";
    }

    if (!_hitlPending && typeof window.updBtn === "function") {
      window.updBtn();
    }
  }

  // ── 检测后端 ──
  async function checkBackend() {
    try {
      const resp = await fetch(`${API_BASE}/api/sessions?user_id=${USER_ID}`, {
        signal: AbortSignal.timeout(3000),
      });
      _connected = resp.ok;
    } catch {
      _connected = false;
    }
    console.log(`[API] Backend ${_connected ? "✓ connected" : "✗ offline (using mock)"}`);
    return _connected;
  }

  // ═══════════════════════════════════════
  //  REST: Sessions
  // ═══════════════════════════════════════

  async function fetchSessions() {
    if (!_connected) return null;
    try {
      const resp = await fetch(`${API_BASE}/api/sessions?user_id=${USER_ID}`);
      if (!resp.ok) return null;
      return await resp.json();
    }
    catch { return null; }
  }

  async function createSession(name) {
    if (!_connected) return null;
    try {
      const resp = await fetch(`${API_BASE}/api/sessions`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name || "新对话", user_id: USER_ID }),
      });
      if (!resp.ok) return null;
      return await resp.json();
    } catch { return null; }
  }

  async function deleteSession(id) {
    if (!_connected) return false;
    try {
      const sid = String(id);
      const resp = await fetch(`${API_BASE}/api/sessions/${encodeURIComponent(sid)}`, {
        method: "DELETE",
      });
      if (!resp.ok) return false;
      const data = await resp.json().catch(() => null);
      return !!(data && data.deleted === true);
    }
    catch { return false; }
  }

  async function fetchMessages(id) {
    if (!_connected) return null;
    try {
      const sid = String(id);
      const resp = await fetch(`${API_BASE}/api/sessions/${encodeURIComponent(sid)}/messages`);
      if (!resp.ok) return null;
      return await resp.json();
    }
    catch { return null; }
  }

  // ═══════════════════════════════════════
  //  REST: Profile / Health / Reminders
  // ═══════════════════════════════════════

  async function fetchProfile() {
    if (!_connected) return null;
    try { return await (await fetch(`${API_BASE}/api/profile/${USER_ID}`)).json(); }
    catch { return null; }
  }

  async function fetchReminders() {
    if (!_connected) return null;
    try { return await (await fetch(`${API_BASE}/api/reminders/${USER_ID}`)).json(); }
    catch { return null; }
  }

  // ═══════════════════════════════════════
  //  WebSocket: 对话
  // ═══════════════════════════════════════

  function connectChat(sessionId) {
    const sid = String(sessionId);
    if (
      _ws &&
      _wsSessionId === sid &&
      (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)
    ) {
      return _ws;
    }
    if (_ws) { _ws.close(); _ws = null; }

    _wsSessionId = sid;
    const socket = new WebSocket(`${WS_BASE}/ws/chat/${encodeURIComponent(sid)}`);
    _ws = socket;

    socket.onopen = () => console.log(`[WS] Connected: ${sid}`);
    socket.onmessage = (e) => {
      try {
        _handleWS(JSON.parse(e.data));
      } catch (err) {
        console.error("[WS] Parse:", err);
      }
    };
    socket.onclose = () => {
      console.log("[WS] Disconnected");
      _setInputLock(false);
      if (_ws === socket) {
        _ws = null;
        _wsSessionId = null;
      }
    };
    socket.onerror = (e) => console.error("[WS] Error:", e);
    return socket;
  }

  async function _ensureChatConnected(sessionId, timeoutMs = 1500) {
    const sid = String(sessionId);
    const ws = connectChat(sid);
    if (!ws) return false;
    if (ws.readyState === WebSocket.OPEN) return true;

    return await new Promise((resolve) => {
      let done = false;
      const finish = (ok) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        ws.removeEventListener("open", onOpen);
        ws.removeEventListener("close", onClose);
        ws.removeEventListener("error", onError);
        resolve(ok);
      };
      const onOpen = () => finish(true);
      const onClose = () => finish(false);
      const onError = () => finish(false);
      const timer = setTimeout(() => finish(false), timeoutMs);

      ws.addEventListener("open", onOpen);
      ws.addEventListener("close", onClose);
      ws.addEventListener("error", onError);
    });
  }

  function sendMessage(content, modality, imagePath, audioPath, imageUrl) {
    if (!_ws || _ws.readyState !== WebSocket.OPEN) return false;
    _ws.send(JSON.stringify({
      type: "message", content, modality: modality || { text: true, audio: false, image: false },
      image_path: imagePath || "", audio_path: audioPath || "", image_url: imageUrl || "",
    }));
    return true;
  }

  function sendHITL(confirmed) {
    if (!_ws || _ws.readyState !== WebSocket.OPEN) return false;
    _ws.send(JSON.stringify({ type: "hitl_response", confirmed }));
    return true;
  }

  // ── WS 事件处理（核心修复）──
  function _handleWS(msg) {
    switch (msg.type) {
      case "node_start":
        // 实时驱动前端 Pipeline 动画：将节点标记为 active
        _onNodeStart(msg.node, msg.event_seq || 0, msg.group_id || "");
        break;

      case "node_end":
        // 节点完成：标记 done + 写入耗时
        _onNodeEnd(
          msg.node,
          msg.data || {},
          Number.isFinite(msg.duration_ms) ? msg.duration_ms : null,
          msg.event_seq || 0,
          msg.group_id || ""
        );
        break;

      case "hitl_request":
        _onHITLRequest(msg.data || {});
        break;

      case "response":
        _setInputLock(false);
        _onFinalResponse(msg.content, msg.debug || {});
        break;

      case "error":
        _setInputLock(false);
        console.error("[Agent Error]", msg.message);
        if (typeof hTyp === "function") hTyp();
        if (typeof addMsg === "function") addMsg("assistant", `⚠️ ${msg.message}`);
        break;
    }
  }

  // ── Pipeline 动画驱动 ──

  function _onNodeStart(nodeName, eventSeq, groupId) {
    if (!_pendingDebug) return;
    if (_isRuntimeMetaNode(nodeName)) return;

    // 尽量匹配最近一个同名 active 节点，避免并行/恢复场景下覆盖历史节点
    const existingActive = _findLatestPipelineNode(nodeName, true);
    if (!existingActive) {
      _pendingDebug.pipeline.push({
        name: nodeName,
        color: _nodeColor(nodeName),
        time: "...",
        status: "active",
        parallel: _parallelFlow && /Medical Agent|Device Agent|Chat Agent/.test(nodeName),
        event_seq: eventSeq || 0,
        group_id: groupId || "",
        _insert_idx: ++_pipelineInsertCounter,
      });
    } else {
      existingActive.status = "active";
      if (eventSeq) existingActive.event_seq = eventSeq;
      if (groupId) existingActive.group_id = groupId;
    }

    _normalizePipelineOrder();

    // 刷新 drawer
    window.DD = _pendingDebug;
    if (typeof updDr === "function") updDr();
  }

  function _onNodeEnd(nodeName, data, durationMs, eventSeq, groupId) {
    if (!_pendingDebug) return;
    if (_isRuntimeMetaNode(nodeName)) return;

    const hasDuration = Number.isFinite(durationMs) && durationMs >= 0;
    const normalizedDuration = hasDuration ? Math.max(durationMs, 1) : null;
    const timeStr = normalizedDuration === null
      ? "..."
      : (normalizedDuration < 1000 ? `${Math.round(normalizedDuration)}ms` : `${(normalizedDuration / 1000).toFixed(1)}s`);
    const isParallel = !!(data && data.parallel);

    if (nodeName === "Supervisor") {
      _parallelFlow = !!(data && data.current_agent === "parallel");
    }
    if (nodeName === "Synthesizer" || nodeName === "Output Guard") {
      _parallelFlow = false;
    }

    const existingActive = _findLatestPipelineNode(nodeName, true);
    if (existingActive) {
      existingActive.status = "done";
      existingActive.time = timeStr;
      existingActive.duration_ms = normalizedDuration;
      existingActive.parallel = isParallel || existingActive.parallel;
      if (eventSeq) existingActive.event_seq = eventSeq;
      if (groupId) existingActive.group_id = groupId;
    } else {
      _pendingDebug.pipeline.push({
        name: nodeName,
        color: _nodeColor(nodeName),
        time: timeStr,
        duration_ms: normalizedDuration,
        status: "done",
        parallel: isParallel || (_parallelFlow && /Medical Agent|Device Agent|Chat Agent/.test(nodeName)),
        event_seq: eventSeq || 0,
        group_id: groupId || "",
        _insert_idx: ++_pipelineInsertCounter,
      });
    }

    _normalizePipelineOrder();

    // 合并后端传来的 debug 分片数据
    if (data) {
      // intents
      if (data.pending_intents || data.current_agent) {
        // supervisor 输出
      }
    }

    window.DD = _pendingDebug;
    if (typeof updDr === "function") updDr();
  }

  function _onHITLRequest(data) {
    _setInputLock(true);
    if (typeof hTyp === "function") hTyp();

    // 兼容后端 interrupt 原始字段：risk_level / message / tool_name
    const normalized = {
      name: data?.name || data?.tool_name || "unknown_tool",
      args: (data && typeof data.args === "object" && data.args) ? data.args : {},
      risk: data?.risk || data?.risk_level || "medium",
      confirmation_message:
        data?.confirmation_message || data?.message || "检测到风险操作，是否继续执行？",
    };

    // 构造 HITL 卡片
    if (typeof showHTL === "function") {
      showHTL({
        debug: { tools: [normalized] },
        response_confirmed: "✅ 操作已执行",
        response_cancelled: "好的，已取消。",
      });
    }
  }

  function _onFinalResponse(content, debug) {
    if (typeof hTyp === "function") hTyp();

    // 合并后端的完整 debug 数据（后端在 response 中发送最终版本）
    if (debug && debug.pipeline) {
      _pendingDebug = debug;
      _sanitizePipeline(_pendingDebug);
      _parallelFlow = false;
      _pipelineInsertCounter = (_pendingDebug.pipeline || []).length;
      _normalizePipelineOrder();
    }

    if (typeof addMsg === "function") {
      const sources = _extractSources(debug);
      addMsg("assistant", content, { sources });
    }

    window.DD = _pendingDebug || debug;
    if (typeof updDr === "function") updDr();
    // 自动打开 drawer 展示过程
    if (window.DD && window.DD.pipeline && window.DD.pipeline.length > 0) {
      document.getElementById("drTog")?.classList.add("has");
    }

    _pendingDebug = null;
    _parallelFlow = false;
    _pipelineInsertCounter = 0;
  }

  // ═══════════════════════════════════════
  //  Override 前端函数
  // ═══════════════════════════════════════

  function _syncUserBadge() {
    if (typeof window.setSessUser === "function") {
      window.setSessUser(USER_ID);
      return;
    }
    const el = document.getElementById("sessUserId");
    if (el) el.textContent = `user: ${USER_ID}`;
  }

  async function refreshSessions(options = {}) {
    if (!_connected) return false;
    _syncUserBadge();

    const preferCurrent = options.preferCurrent !== false;
    const preferredId = options.preferredId ? String(options.preferredId) : "";

    let sessions = await fetchSessions();
    if (!Array.isArray(sessions)) return false;

    // 保证每个用户至少有一个会话，避免空界面
    if (sessions.length === 0) {
      const created = await createSession("新对话");
      sessions = created ? [created] : [];
    }

    const mapped = sessions.map(s => ({
      id: String(s.session_id),
      nm: s.name,
      dt: _fmtDate(s.updated_at),
      ms: [],
      _backend: true,
    }));

    window.SS = mapped;
    if (typeof rSess === "function") rSess();

    const currentId = String(window.aS || _activeSessionId || "");
    const hasCurrent = mapped.some(s => s.id === currentId);
    const hasPreferred = mapped.some(s => s.id === preferredId);

    let targetId = "";
    if (hasPreferred) {
      targetId = preferredId;
    } else if (preferCurrent && hasCurrent) {
      targetId = currentId;
    } else if (mapped.length > 0) {
      targetId = mapped[0].id;
    }

    if (targetId && typeof loadS === "function") {
      await loadS(targetId);
    } else {
      window.aS = "";
      const msgEl = document.getElementById("mCtn");
      if (msgEl) msgEl.innerHTML = "";
      const titleEl = document.getElementById("cTitle");
      if (titleEl) titleEl.textContent = "小银 · Silver Pilot";
    }

    return true;
  }

  async function initWithBackend() {
    _syncUserBadge();
    const ok = await checkBackend();
    if (!ok) return;

    await refreshSessions({ preferCurrent: false });

    const reminders = await fetchReminders();
    if (reminders) {
      const remEl = document.getElementById("remList");
      if (remEl) {
        remEl.innerHTML = reminders.map((r, i) =>
          `<div class="rem-item" style="animation:msgIn .3s ease ${i * 0.04}s both${r.done ? ";opacity:.55" : ""}">
            <div class="rem-dot" style="background:${r.done ? "var(--text-hint)" : "var(--accent)"}"></div>
            <div style="flex:1;min-width:0">
              <div class="rem-msg"${r.done ? ' style="text-decoration:line-through"' : ""}>${r.message}</div>
              <div class="rem-sub">${r.repeat}${r.done ? " · 已完成" : ""}</div>
            </div>
            <div class="rem-time">${r.time}</div>
          </div>`
        ).join("");
      }
    }
  }

  // Override: loadS
  const _origLoadS = window.loadS;
  window.loadS = async function (id) {
    const sid = String(id);
    _activeSessionId = sid;

    if (!_connected) { if (_origLoadS) _origLoadS(sid); return; }

    const messages = await fetchMessages(sid);
    if (messages) {
      const session = (window.SS || []).find(s => String(s.id) === sid);
      if (session) {
        session.ms = messages.map(m => {
          let img = false, aud = false, url = "";
          if (m.metadata) {
            if (m.metadata.image_url) {
              img = true;
              url = m.metadata.image_url;
            } else if (m.metadata.image_path) {
              img = true;
              url = _toPublicUploadUrl(m.metadata.image_path, sid);
            }
            if (m.metadata.audio_path) aud = true;
          }
          return {
            r: m.role === "user" ? "u" : "a", c: m.content,
            t: _fmtTime(m.timestamp), s: m.sources,
            img: img, aud: aud, url: url
          };
        });
      }
    }
    if (_origLoadS) _origLoadS(sid);
    connectChat(sid);
  };

  // Override: newSess
  const _origNewSess = window.newSess;
  window.newSess = async function () {
    if (!_connected) { if (_origNewSess) _origNewSess(); return; }
    const result = await createSession("新对话");
    if (result && result.session_id) {
      const sid = String(result.session_id);
      await refreshSessions({ preferredId: sid, preferCurrent: false });
      if (typeof tSess === "function") tSess();
    }
  };

  // Override: delSess（删除会话并刷新列表）
  const _origDelSess = window.delSess;
  window.delSess = async function (ev, id) {
    if (ev) {
      ev.stopPropagation();
      ev.preventDefault();
    }

    const sid = String(id || "").trim();
    if (!sid) return;

    const confirmed = typeof window.confirm === "function"
      ? window.confirm("确认删除该会话吗？")
      : true;
    if (!confirmed) return;

    if (!_connected) {
      if (_origDelSess) _origDelSess(ev, sid, true);
      return;
    }

    const current = Array.isArray(window.SS) ? window.SS.map((s) => String(s.id)) : [];
    const wasActive = String(window.aS || "") === sid;
    const nextId = wasActive ? (current.find((x) => x !== sid) || "") : String(window.aS || "");

    const ok = await deleteSession(sid);
    if (!ok) {
      if (typeof addMsg === "function") {
        addMsg("assistant", "⚠️ 删除会话失败，请刷新后重试。", { noSave: true });
      }
      return;
    }

    await refreshSessions({ preferredId: nextId, preferCurrent: !wasActive });
  };

  // Override: go（进入对话页时同步对应 user 的会话与消息）
  const _origGo = window.go;
  window.go = function (i) {
    if (_origGo) _origGo(i);
    if (i === 1 && _connected) {
      refreshSessions({ preferCurrent: true }).catch((err) => {
        console.error("[API] refreshSessions failed:", err);
      });
    }
  };



  // Override: resHTL
  const _origResHTL = window.resHTL;
  window.resHTL = function (id, ok) {
    if (!_connected) {
      if (_origResHTL) _origResHTL(id, ok);
      return;
    }

    const card = document.getElementById(id);
    if (card) {
      card.style.borderColor = ok ? "var(--green)" : "var(--text-hint)";
      const acts = card.querySelector(".hitl-acts");
      if (acts) {
        acts.innerHTML = `<span style="font-size:13px;color:${ok ? "var(--green)" : "var(--text-sub)"}">${ok ? "✅ 已确认执行" : "❌ 已取消"}</span>`;
      }
    }

    if (window.DD && window.DD.tools && window.DD.tools[0]) {
      window.DD.tools[0].needs_confirmation = false;
      window.DD.tools[0].result = { status: ok ? "confirmed" : "cancelled" };
      if (typeof updDr === "function") updDr();
    }

    if (!sendHITL(!!ok)) {
      _setInputLock(false);
      if (typeof addMsg === "function") {
        addMsg("assistant", "⚠️ 确认结果发送失败，请重试。", {});
      }
    }
  };

  // ── Utilities ──

  function _fmtDate(ts) {
    const diff = Date.now() / 1000 - ts;
    if (diff < 3600) return "刚刚";
    if (diff < 86400) return "今天";
    if (diff < 172800) return "昨天";
    return `${Math.floor(diff / 86400)}天前`;
  }

  function _fmtTime(ts) {
    const d = new Date(ts * 1000);
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  function _extractSources(debug) {
    if (!debug) return [];
    const s = new Set();
    if (debug.rag) {
      (debug.rag.graph_results || []).forEach(r => s.add(r.source));
      (debug.rag.vector_results || []).forEach(r => s.add(r.source));
    }
    return [...s];
  }

  function _toPublicUploadUrl(pathOrUrl, sessionId) {
    const raw = String(pathOrUrl || "").trim();
    if (!raw) return "";
    if (raw.startsWith("/upload/")) return raw;
    if (/^https?:\/\//i.test(raw)) return raw;

    const parts = raw.split(/[\\/]/).filter(Boolean);
    const filename = parts.length ? parts[parts.length - 1] : "";
    if (!filename || !filename.includes(".")) return "";

    const uid = encodeURIComponent(USER_ID);
    const sid = encodeURIComponent(String(sessionId || _activeSessionId || "default_session"));
    return `/upload/${uid}/${sid}/${encodeURIComponent(filename)}`;
  }

  function _nodeColor(name) {
    return {
      "Perception": "var(--n-per)", "Supervisor": "var(--n-sup)",
      "Medical Agent": "var(--n-med)", "Device Agent": "var(--n-dev)",
      "Chat Agent": "var(--yellow)", "Emergency Agent": "var(--red)",
      "Synthesizer": "var(--accent)", "Output Guard": "var(--n-grd)",
      "Memory Writer": "var(--text-hint)",
    }[name] || "var(--text-sub)";
  }

  function _isRuntimeMetaNode(nodeName) {
    return typeof nodeName === "string" && nodeName.startsWith("__");
  }

  function _formatDurationMs(ms) {
    if (!Number.isFinite(ms) || ms < 0) return "...";
    if (ms < 1000) return `${Math.round(Math.max(ms, 1))}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  function _sanitizePipeline(debugObj) {
    if (!debugObj || !Array.isArray(debugObj.pipeline)) return;
    debugObj.pipeline = debugObj.pipeline
      .filter((n) => n && !_isRuntimeMetaNode(n.name))
      .map((n) => {
        const node = { ...n };
        let ms = Number.isFinite(node.duration_ms) ? Number(node.duration_ms) : null;
        if (ms === null && Number.isFinite(node.start_ms) && Number.isFinite(node.end_ms)) {
          ms = Math.max(Number(node.end_ms) - Number(node.start_ms), 0);
        }
        if (ms !== null) {
          node.duration_ms = ms;
          if (!node.time || node.time === "..." || node.time === "0ms") {
            node.time = _formatDurationMs(ms);
          }
        }
        return node;
      });
  }

  // ── Global API ──
  window.SilverPilotAPI = {
    checkBackend, fetchSessions, createSession, deleteSession,
    fetchMessages, fetchProfile, fetchReminders,
    connectChat, sendMessage, sendHITL, refreshSessions,
    getUserId: () => USER_ID,
    isConnected: () => _connected,

    uploadFile: async (ev, type) => {
      const file = ev.target.files[0];
      if (!file) return;
      ev.target.value = "";

      if (!_connected) {
        if (typeof addMsg === "function") addMsg("assistant", "⚠️ 后端未连接，无法上传文件。");
        return;
      }

      const sid = String(window.aS || _activeSessionId || _wsSessionId || "default_session");
      const uid = String(typeof USER_ID !== "undefined" ? USER_ID : "default_user");

      const fd = new FormData();
      fd.append("file", file);
      fd.append("user_id", uid);
      fd.append("session_id", sid);
      try {
        const resp = await fetch("/api/upload", { method: "POST", body: fd });
        if (!resp.ok) throw new Error("上传请求失败");
        const data = await resp.json();

        window.PendingUploads[type] = {
           local_path: data.local_path,
           url: data.url
        };
        window.renderBuffer();
      } catch (err) {
        if (typeof addMsg === "function") addMsg("assistant", "⚠️ " + err.message);
      }
    },
    removeUpload: (type) => {
      window.PendingUploads[type] = null;
      window.renderBuffer();
    }
  };

  // --- OVERRIDES ---
  window.PendingUploads = { image: null, audio: null };
  window.renderBuffer = function() {
      const el = document.getElementById("upBuf");
      if (!el) return;
      let html = "";
      if (PendingUploads.image) {
        html += `<div class="buf-item">
          <img src="${window.PendingUploads.image.url}">
          <div class="buf-text">
            <span class="buf-title">图片附件</span>
            <span class="buf-sub">已准备发送</span>
          </div>
          <div class="buf-del" onclick="SilverPilotAPI.removeUpload('image')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg></div>
        </div>`;
      }
      if (PendingUploads.audio) {
        html += `<div class="buf-item">
          <div class="buf-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg></div>
          <div class="buf-text">
            <span class="buf-title">语音附件</span>
            <span class="buf-sub">已准备发送</span>
          </div>
          <div class="buf-del" onclick="SilverPilotAPI.removeUpload('audio')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg></div>
        </div>`;
      }
      el.innerHTML = html;
      if (html) el.classList.add("has-items");
      else el.classList.remove("has-items");
      if (typeof window.updBtn === "function") window.updBtn();
  };

  const _origHandleSend = window.handleSend;
  window.handleSend = async function () {
    if (_hitlPending) {
      if (typeof addMsg === "function") {
        addMsg("assistant", "⚠️ 当前有待确认操作，请先点击弹窗中的“确认执行”或“取消”。", { noSave: true });
      }
      return;
    }

    const field = document.getElementById("iField");
    const text = field ? field.value.trim() : "";
    const hasUploads = window.PendingUploads.image || window.PendingUploads.audio;
    if (!text && !hasUploads) return;

    if (!_connected) {
      if (_origHandleSend) _origHandleSend.call(window);
      return;
    }

    const sid = String(window.aS || _activeSessionId || _wsSessionId || "");
    if (!sid) {
      if (typeof addMsg === "function") addMsg("assistant", "⚠️当前无可用会话，请先创建。");
      return;
    }

    const img_path = window.PendingUploads.image ? window.PendingUploads.image.local_path : "";
    const img_url = window.PendingUploads.image ? window.PendingUploads.image.url : "";
    const aud_path = window.PendingUploads.audio ? window.PendingUploads.audio.local_path : "";

    let dispText = text || "[发送了文件]";
    if (typeof addMsg === "function") addMsg("user", dispText, { isImage: !!img_path, isAudio: !!aud_path, imgUrl: img_url });
    if (typeof window.sTyp === "function") window.sTyp();

    if (field) field.value = "";
    window.PendingUploads.image = null;
    window.PendingUploads.audio = null;
    window.renderBuffer();
    if (typeof window.updBtn === "function") window.updBtn();

    try {
        const ready = await _ensureChatConnected(sid);
        if (!ready) throw new Error("无法连接到会话");

        _pendingDebug = { pipeline: [], intents: [], entities: [], rag: null, tools: [], perception: null };
        _parallelFlow = false;
        _pipelineInsertCounter = 0;
        window.DD = _pendingDebug;
        if (typeof tDr === "function" && typeof dT === "function") dT("pipe");

        const mod = { text: !!text, audio: !!aud_path, image: !!img_path };
        if (!sendMessage(dispText, mod, img_path, aud_path, img_url)) {
          throw new Error("WS 发送失败");
        }
    } catch (err) {
        if (typeof window.hTyp === "function") window.hTyp();
        if (typeof addMsg === "function") addMsg("assistant", "⚠️ 发送失败: " + err.message);
    }
  };

  const _origClrChat = window.clrChat;
  window.clrChat = async function () {
    if (!_connected) {
       if (_origClrChat) _origClrChat();
       return;
    }
    const sid = String(window.aS || _activeSessionId || "");
    if (!sid) return;

    try {
       await fetch(`${API_BASE}/api/sessions/${encodeURIComponent(sid)}/messages`, { method: "DELETE" });
       await refreshSessions({ preferCurrent: true });
    } catch (e) {
       console.error("Clear chat failed", e);
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(initWithBackend, 500));
  } else {
    setTimeout(initWithBackend, 500);
  }

  console.log("[Silver Pilot API v2] Connector loaded.");
})();
