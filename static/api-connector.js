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

  // ── 配置 ──
  const API_BASE = window.location.origin;
  const WS_BASE = API_BASE.replace("http", "ws");
  const USER_ID = "default_user";

  let _ws = null;          // 当前 WebSocket 连接
  let _wsSessionId = null;  // 当前 WS 连接的 session ID
  let _connected = false;   // 后端是否可用
  let _onResponse = null;   // 回调：收到最终响应
  let _onHITL = null;       // 回调：收到 HITL 请求

  // ── 检测后端是否可用 ──
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
      return await resp.json();
    } catch (e) {
      console.error("[API] fetchSessions failed:", e);
      return null;
    }
  }

  async function createSession(name) {
    if (!_connected) return null;
    try {
      const resp = await fetch(`${API_BASE}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name || "新对话", user_id: USER_ID }),
      });
      return await resp.json();
    } catch (e) {
      console.error("[API] createSession failed:", e);
      return null;
    }
  }

  async function deleteSession(sessionId) {
    if (!_connected) return false;
    try {
      await fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: "DELETE" });
      return true;
    } catch {
      return false;
    }
  }

  async function fetchMessages(sessionId) {
    if (!_connected) return null;
    try {
      const resp = await fetch(`${API_BASE}/api/sessions/${sessionId}/messages`);
      return await resp.json();
    } catch (e) {
      console.error("[API] fetchMessages failed:", e);
      return null;
    }
  }

  // ═══════════════════════════════════════
  //  REST: Profile / Health / Reminders
  // ═══════════════════════════════════════

  async function fetchProfile() {
    if (!_connected) return null;
    try {
      const resp = await fetch(`${API_BASE}/api/profile/${USER_ID}`);
      return await resp.json();
    } catch {
      return null;
    }
  }

  async function fetchHealth() {
    if (!_connected) return null;
    try {
      const resp = await fetch(`${API_BASE}/api/health/${USER_ID}`);
      return await resp.json();
    } catch {
      return null;
    }
  }

  async function fetchReminders() {
    if (!_connected) return null;
    try {
      const resp = await fetch(`${API_BASE}/api/reminders/${USER_ID}`);
      return await resp.json();
    } catch {
      return null;
    }
  }

  // ═══════════════════════════════════════
  //  WebSocket: 对话
  // ═══════════════════════════════════════

  function connectChat(sessionId) {
    // 如果已连接到同一个 session，复用
    if (_ws && _ws.readyState === WebSocket.OPEN && _wsSessionId === sessionId) {
      return _ws;
    }

    // 关闭旧连接
    if (_ws) {
      _ws.close();
      _ws = null;
    }

    _wsSessionId = sessionId;
    _ws = new WebSocket(`${WS_BASE}/ws/chat/${sessionId}`);

    _ws.onopen = () => {
      console.log(`[WS] Connected to session ${sessionId}`);
    };

    _ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        _handleWSMessage(msg);
      } catch (e) {
        console.error("[WS] Parse error:", e);
      }
    };

    _ws.onclose = () => {
      console.log("[WS] Disconnected");
      _ws = null;
      _wsSessionId = null;
    };

    _ws.onerror = (e) => {
      console.error("[WS] Error:", e);
    };

    return _ws;
  }

  function sendMessage(content, modality, imagePath, audioPath) {
    if (!_ws || _ws.readyState !== WebSocket.OPEN) {
      console.warn("[WS] Not connected, cannot send");
      return false;
    }

    _ws.send(JSON.stringify({
      type: "message",
      content: content,
      modality: modality || { text: true, audio: false, image: false },
      image_path: imagePath || "",
      audio_path: audioPath || "",
    }));
    return true;
  }

  function sendHITLResponse(confirmed) {
    if (!_ws || _ws.readyState !== WebSocket.OPEN) return false;
    _ws.send(JSON.stringify({
      type: "hitl_response",
      confirmed: confirmed,
    }));
    return true;
  }

  // ── WS 消息处理 ──
  function _handleWSMessage(msg) {
    switch (msg.type) {
      case "node_start":
        // 触发前端 Pipeline 动画
        if (window._onNodeStart) window._onNodeStart(msg.node);
        break;

      case "node_end":
        if (window._onNodeEnd) window._onNodeEnd(msg.node, msg.data, msg.duration_ms);
        break;

      case "hitl_request":
        // 显示 HITL 确认卡片
        if (window._onHITLRequest) window._onHITLRequest(msg.data);
        if (_onHITL) _onHITL(msg.data);
        break;

      case "response":
        // 最终响应 + debug 数据
        if (window._onAgentResponse) window._onAgentResponse(msg.content, msg.debug);
        if (_onResponse) _onResponse(msg.content, msg.debug);
        break;

      case "error":
        console.error("[Agent Error]", msg.message);
        if (window._onAgentError) window._onAgentError(msg.message);
        break;
    }
  }

  // ═══════════════════════════════════════
  //  Override 前端函数（对接真实后端）
  // ═══════════════════════════════════════

  /**
   * 初始化时加载真实数据。
   * 如果后端可用，替换 mock sessions/reminders/profile。
   */
  async function initWithBackend() {
    const ok = await checkBackend();
    if (!ok) {
      console.log("[API] Backend offline, keeping mock data");
      return;
    }

    // 加载会话列表
    const sessions = await fetchSessions();
    if (sessions && sessions.length > 0 && typeof SS !== "undefined") {
      // 将后端 sessions 同步到前端全局变量 SS
      window.SS = sessions.map((s) => ({
        id: s.session_id,
        nm: s.name,
        dt: _formatDate(s.updated_at),
        ms: [], // 消息稍后按需加载
        _backend: true,
      }));
      if (typeof rSess === "function") rSess();
      if (typeof loadS === "function" && window.SS.length > 0) {
        loadS(window.SS[0].id);
      }
    }

    // 加载提醒
    const reminders = await fetchReminders();
    if (reminders && typeof REMS !== "undefined") {
      window.REMS = reminders.map((r) => ({
        t: r.time,
        m: r.message,
        r: r.repeat,
        d: r.done ? 1 : 0,
        c: r.done ? "var(--text-hint)" : "var(--accent)",
      }));
      if (typeof renderReminders === "function") renderReminders();
      // 兼容压缩后的函数名
      const remEl = document.getElementById("remList");
      if (remEl && window.REMS) {
        remEl.innerHTML = window.REMS.map((r, i) =>
          `<div class="rem-item" style="animation:msgIn .3s ease ${i * 0.04}s both${r.d ? ";opacity:.55" : ""}">
            <div class="rem-dot" style="background:${r.d ? "var(--text-hint)" : r.c}"></div>
            <div style="flex:1;min-width:0">
              <div class="rem-msg"${r.d ? ' style="text-decoration:line-through"' : ""}>${r.m}</div>
              <div class="rem-sub">${r.r}${r.d ? " · 已完成" : ""}</div>
            </div>
            <div class="rem-time">${r.t}</div>
          </div>`
        ).join("");
      }
    }
  }

  /**
   * Override: 加载 session 时从后端拉取消息。
   */
  const _origLoadS = window.loadS;
  window.loadS = async function (id) {
    if (!_connected) {
      if (_origLoadS) _origLoadS(id);
      return;
    }

    // 从后端加载消息
    const messages = await fetchMessages(id);
    if (messages) {
      const session = (window.SS || []).find((s) => s.id === id);
      if (session) {
        session.ms = messages.map((m) => ({
          r: m.role === "user" ? "u" : "a",
          c: m.content,
          t: _formatTime(m.timestamp),
          s: m.sources,
        }));
      }
    }

    // 调用原始加载函数
    if (_origLoadS) _origLoadS(id);

    // 建立 WebSocket 连接
    connectChat(id);
  };

  /**
   * Override: 创建新会话时同步到后端。
   */
  const _origNewSess = window.newSess;
  window.newSess = async function () {
    if (!_connected) {
      if (_origNewSess) _origNewSess();
      return;
    }

    const result = await createSession("新对话");
    if (result) {
      window.SS = window.SS || [];
      window.SS.unshift({
        id: result.session_id,
        nm: result.name,
        dt: "刚刚",
        ms: [{ r: "a", c: "您好！我是小银，有什么可以帮您的吗？", t: _formatTime(Date.now() / 1000) }],
        _backend: true,
      });
      if (typeof loadS === "function") loadS(result.session_id);
      if (typeof tSess === "function") tSess();
    }
  };

  /**
   * Override: 发送消息时通过 WebSocket。
   */
  const _origHandleSend = window.handleSend;
  window.handleSend = function () {
    const field = document.getElementById("iField");
    const text = field ? field.value.trim() : "";
    if (!text) return;

    // 添加用户消息到 UI
    if (typeof addMsg === "function") addMsg("user", text);
    if (field) field.value = "";
    if (typeof updBtn === "function") updBtn();

    // 通过 WS 发送
    if (_connected && _ws && _ws.readyState === WebSocket.OPEN) {
      if (typeof sTyp === "function") sTyp();

      // 设置响应回调
      _onResponse = (content, debug) => {
        if (typeof hTyp === "function") hTyp();
        if (typeof addMsg === "function") {
          addMsg("assistant", content, { sources: _extractSources(debug) });
        }
        // 更新 drawer
        if (typeof DD !== "undefined") window.DD = _transformDebug(debug);
        if (typeof updDr === "function") updDr();
        _onResponse = null;
      };

      // 设置 HITL 回调
      _onHITL = (data) => {
        if (typeof hTyp === "function") hTyp();
        // 显示 HITL 卡片（复用前端现有的 showHTL）
        if (typeof showHTL === "function") {
          showHTL({
            debug: { tools: [data] },
            response_confirmed: "✅ 操作已执行",
            response_cancelled: "好的，已取消。",
          });
        }
        _onHITL = null;
      };

      // Pipeline 动画回调
      window._onNodeStart = (node) => {
        // 可以在这里驱动 drawer 的 pipeline 动画
        console.log(`[Pipeline] ${node} started`);
      };

      window._onNodeEnd = (node, data, duration) => {
        console.log(`[Pipeline] ${node} done (${duration}ms)`);
      };

      sendMessage(text, { text: true, audio: false, image: false });
    } else {
      // WS 不可用，走 mock
      if (_origHandleSend) _origHandleSend.call(window);
    }
  };

  // ── Override HITL 确认 ──
  const _origResHTL = window.resHTL;
  window.resHTL = function (id, ok) {
    // 更新 UI
    if (_origResHTL) _origResHTL(id, ok);

    // 通过 WS 发送确认
    if (_connected) {
      sendHITLResponse(!!ok);
    }
  };

  // ═══════════════════════════════════════
  //  Utilities
  // ═══════════════════════════════════════

  function _formatDate(timestamp) {
    const now = Date.now() / 1000;
    const diff = now - timestamp;
    if (diff < 3600) return "刚刚";
    if (diff < 86400) return "今天";
    if (diff < 172800) return "昨天";
    return `${Math.floor(diff / 86400)}天前`;
  }

  function _formatTime(timestamp) {
    const d = new Date(timestamp * 1000);
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  function _extractSources(debug) {
    if (!debug) return [];
    const sources = new Set();
    if (debug.rag) {
      (debug.rag.graph_results || []).forEach((r) => sources.add(r.source));
      (debug.rag.vector_results || []).forEach((r) => sources.add(r.source));
    }
    return [...sources];
  }

  function _transformDebug(debug) {
    // 将后端 debug 格式转为前端 DD 格式
    if (!debug) return null;
    return {
      pipeline: (debug.pipeline || []).map((n) => ({
        name: n.name,
        color: _nodeColor(n.name),
        time: n.time || "...",
        status: "done",
      })),
      intents: (debug.intents || []).map((i) => ({
        type: i.type,
        query: i.query,
        priority: i.priority,
        color: _intentColor(i.type),
      })),
      entities: debug.entities || [],
      rag: debug.rag || null,
      tools: debug.tools || [],
      perception: debug.perception || null,
    };
  }

  function _nodeColor(name) {
    const map = {
      Perception: "var(--n-per)",
      Supervisor: "var(--n-sup)",
      "Medical Agent": "var(--n-med)",
      "Device Agent": "var(--n-dev)",
      "Chat Agent": "var(--yellow)",
      "Output Guard": "var(--n-grd)",
    };
    return map[name] || "var(--text-sub)";
  }

  function _intentColor(type) {
    const map = {
      MEDICAL_QUERY: "var(--n-med)",
      DEVICE_CONTROL: "var(--n-dev)",
      CHITCHAT: "var(--yellow)",
      EMERGENCY: "var(--red)",
    };
    return map[type] || "var(--text-sub)";
  }

  // ═══════════════════════════════════════
  //  暴露全局 API + 自动初始化
  // ═══════════════════════════════════════

  window.SilverPilotAPI = {
    checkBackend,
    fetchSessions,
    createSession,
    deleteSession,
    fetchMessages,
    fetchProfile,
    fetchHealth,
    fetchReminders,
    connectChat,
    sendMessage,
    sendHITLResponse,
    isConnected: () => _connected,
  };

  // 页面加载完成后自动尝试连接后端
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      setTimeout(initWithBackend, 500); // 等前端 init() 先执行
    });
  } else {
    setTimeout(initWithBackend, 500);
  }

  console.log("[Silver Pilot API] Connector loaded. Use window.SilverPilotAPI to access.");
})();
