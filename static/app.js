const API_BASE_URL = (window.API_BASE_URL || "").replace(/\/$/, "");
const sessionList = document.querySelector("#sessions");
const messages = document.querySelector("#messages");
const title = document.querySelector("#chat-title");
const status = document.querySelector("#status");
const form = document.querySelector("#message-form");
const input = document.querySelector("#message-input");
const newButton = document.querySelector("#new-session");
const clearDatabaseButton = document.querySelector("#clear-database");
let sessions = [];
let currentSessionId = null;
let busy = false;

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    const message = typeof detail === "string"
      ? detail
      : detail?.message || (Array.isArray(detail) ? detail.map((item) => item.msg).join("; ") : "");
    throw new Error(message || "Backend не смог выполнить запрос");
  }
  return response.status === 204 ? null : response.json();
}

function setBusy(value) {
  busy = value;
  input.disabled = value;
  form.querySelector("button").disabled = value;
  newButton.disabled = value;
  clearDatabaseButton.disabled = value;
  if (value) status.textContent = "Агент отвечает…";
}

function renderSessions() {
  sessionList.replaceChildren(...sessions.map((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = session.id === currentSessionId ? "session active" : "session";
    button.textContent = session.title;
    button.addEventListener("click", () => openSession(session.id));
    return button;
  }));
}

function renderMessages(items) {
  if (!items.length) {
    messages.innerHTML = '<div class="empty"><span>✦</span><h2>Чем помочь?</h2><p>Каждая сессия имеет собственную историю диалога.</p></div>';
    return;
  }
  messages.replaceChildren(...items.map((message) => {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;
    const label = document.createElement("span");
    label.textContent = message.role === "user" ? "Вы" : "Агент";
    const content = document.createElement("p");
    content.textContent = message.content;
    article.append(label, content);
    return article;
  }));
  messages.scrollTop = messages.scrollHeight;
}

async function openSession(sessionId) {
  if (busy) return;
  try {
    const session = await api(`/api/chat/sessions/${sessionId}`);
    currentSessionId = session.id;
    title.textContent = session.title;
    renderSessions();
    renderMessages(session.messages);
    status.textContent = "Готов";
    input.focus();
  } catch (error) {
    status.textContent = error.message;
  }
}

async function createSession() {
  if (busy) return;
  try {
    const session = await api("/api/chat/sessions", { method: "POST" });
    sessions.unshift(session);
    await openSession(session.id);
  } catch (error) {
    status.textContent = error.message;
  }
}

async function loadSessions() {
  try {
    sessions = await api("/api/chat/sessions");
    if (sessions.length) await openSession(sessions[0].id);
    else await createSession();
  } catch (error) {
    status.textContent = error.message;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || !currentSessionId || busy) return;
  setBusy(true);
  try {
    const result = await api(`/api/chat/sessions/${currentSessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    });
    input.value = "";
    const session = await api(`/api/chat/sessions/${currentSessionId}`);
    sessions = sessions.filter((item) => item.id !== result.session.id);
    sessions.unshift(result.session);
    title.textContent = result.session.title;
    renderSessions();
    renderMessages(session.messages);
    status.textContent = "Готов";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

newButton.addEventListener("click", createSession);
clearDatabaseButton.addEventListener("click", async () => {
  if (busy || !window.confirm("Удалить все чат-сессии и сообщения без возможности восстановления?")) return;

  setBusy(true);
  try {
    await api("/api/chat/sessions", { method: "DELETE" });
    sessions = [];
    currentSessionId = null;
    title.textContent = "Новый чат";
    renderSessions();
    renderMessages([]);
    status.textContent = "История очищена";
  } catch (error) {
    status.textContent = error.message;
    return;
  } finally {
    setBusy(false);
  }
  await createSession();
});

loadSessions();
