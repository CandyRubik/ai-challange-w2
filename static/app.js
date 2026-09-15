const API_BASE_URL = (window.API_BASE_URL || "").replace(/\/$/, "");
const sessionList = document.querySelector("#sessions");
const messages = document.querySelector("#messages");
const title = document.querySelector("#chat-title");
const status = document.querySelector("#status");
const form = document.querySelector("#message-form");
const input = document.querySelector("#message-input");
const submitButton = form.querySelector("button[type='submit']");
const newButton = document.querySelector("#new-session");
const clearDatabaseButton = document.querySelector("#clear-database");
const profileSelect = document.querySelector("#profile-select");
const newProfileButton = document.querySelector("#new-profile");
const editProfileButton = document.querySelector("#edit-profile");
const activeProfileName = document.querySelector("#active-profile-name");
const activeProfileStatus = document.querySelector("#active-profile-status");
const activeProfileSummary = document.querySelector("#active-profile-summary");
const activeProfileConstraints = document.querySelector("#active-profile-constraints");
const profileDialog = document.querySelector("#profile-dialog");
const profileForm = document.querySelector("#profile-form");
const profileDialogTitle = document.querySelector("#profile-dialog-title");
const profileIdInput = document.querySelector("#profile-id");
const profileNameInput = document.querySelector("#profile-name");
const profileDescriptionInput = document.querySelector("#profile-description");
const profileLanguageInput = document.querySelector("#profile-language");
const profileToneInput = document.querySelector("#profile-tone");
const profileDetailInput = document.querySelector("#profile-detail-level");
const profileFormatInput = document.querySelector("#profile-response-format");
const profileConstraintsInput = document.querySelector("#profile-constraints-input");
const closeProfileDialogButton = document.querySelector("#close-profile-dialog");
const cancelProfileButton = document.querySelector("#cancel-profile");
const deleteProfileButton = document.querySelector("#delete-profile");
const commandMenu = document.querySelector("#command-menu");
const followupQueue = document.querySelector("#followup-queue");
const queuedMessagesContainer = document.querySelector("#queued-messages");
const queueCount = document.querySelector("#queue-count");
const shortTermMemory = document.querySelector("#short-term-memory");
const workingMemory = document.querySelector("#working-memory");
const longTermMemory = document.querySelector("#long-term-memory");
const shortTermCount = document.querySelector("#short-term-count");
const workingCount = document.querySelector("#working-count");
const longTermCount = document.querySelector("#long-term-count");
let sessions = [];
let profiles = [];
let currentProfileId = null;
let currentSessionId = null;
let currentMessages = [];
let memorySnapshot = { working: [], long_term: [] };
let busy = false;
let activeCommandIndex = 0;
let queuedMessages = [];

const memoryCommands = [
  { name: "/goal", layer: "working", category: "goal", description: "цель текущей задачи" },
  { name: "/constraint", layer: "working", category: "constraint", description: "ограничение текущей задачи" },
  { name: "/decision", layer: "working", category: "decision", description: "решение текущей задачи" },
  { name: "/profile", layer: "long_term", category: "profile", description: "устойчивый факт профиля" },
  { name: "/preference", layer: "long_term", category: "preference", description: "предпочтение пользователя" },
  { name: "/knowledge", layer: "long_term", category: "knowledge", description: "знание для будущих чатов" },
];

const profileLabels = {
  tone: {
    neutral: "нейтральный",
    friendly: "дружелюбный",
    formal: "формальный",
    technical: "технический",
  },
  detail_level: {
    brief: "кратко",
    balanced: "сбалансированно",
    detailed: "подробно",
  },
  response_format: {
    plain: "обычный текст",
    bullets: "списки",
    steps: "пошагово",
  },
  language: { ru: "русский", en: "English" },
};

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
  submitButton.textContent = value ? "В очередь" : "Отправить";
  newButton.disabled = value || activeProfile()?.onboarding_complete === false;
  clearDatabaseButton.disabled = value;
  profileSelect.disabled = value;
  newProfileButton.disabled = value;
  editProfileButton.disabled = value;
  if (value) status.textContent = "Агент отвечает…";
}

function activeProfile() {
  return profiles.find((profile) => profile.id === currentProfileId) || null;
}

function renderActiveProfile() {
  const profile = activeProfile();
  if (!profile) return;
  activeProfileName.textContent = profile.name;
  newButton.disabled = busy || !profile.onboarding_complete;
  newButton.title = profile.onboarding_complete
    ? "Создать новый чат"
    : "Сначала завершите интервью";
  editProfileButton.disabled = busy;
  activeProfileStatus.textContent = profile.onboarding_complete
    ? "PROFILE · READY"
    : `INTERVIEW · ${profile.onboarding_step}/3`;
  const settings = [
    profileLabels.language[profile.language] || profile.language,
    profileLabels.tone[profile.tone] || profile.tone,
    profileLabels.detail_level[profile.detail_level] || profile.detail_level,
    profileLabels.response_format[profile.response_format] || profile.response_format,
  ];
  activeProfileSummary.textContent = profile.onboarding_complete
    ? (profile.description
      ? `${profile.description} · ${settings.join(" · ")}`
      : settings.join(" · "))
    : "Агент ещё собирает профиль из диалога";
  activeProfileConstraints.replaceChildren(...profile.constraints.map((constraint) => {
    const tag = document.createElement("small");
    tag.textContent = constraint;
    return tag;
  }));
}

function renderProfiles() {
  profileSelect.replaceChildren(...profiles.map((profile) => {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.onboarding_complete
      ? profile.name
      : `${profile.name} · интервью`;
    return option;
  }));
  if (currentProfileId) profileSelect.value = currentProfileId;
  renderActiveProfile();
}

function rememberSelectedProfile() {
  try {
    localStorage.setItem("rubik-active-profile", currentProfileId);
  } catch (_) {
    // Persistence is optional when storage is blocked by the browser.
  }
}

function openProfileDialog(profile) {
  if (busy) return;
  profileDialogTitle.textContent = profile.onboarding_complete
    ? "Проверка профиля"
    : "Интервью не завершено";
  profileIdInput.value = profile.id;
  profileNameInput.value = profile.name;
  profileDescriptionInput.value = profile.description;
  profileLanguageInput.value = profile.language;
  profileToneInput.value = profile.tone;
  profileDetailInput.value = profile.detail_level;
  profileFormatInput.value = profile.response_format;
  profileConstraintsInput.value = profile.constraints.join("\n");
  deleteProfileButton.hidden = profile.id === "default";
  profileForm.querySelector("button[type='submit']").disabled = !profile.onboarding_complete;
  profileDialog.showModal();
  profileNameInput.focus();
}

function closeProfileDialog() {
  profileDialog.close();
  profileForm.reset();
}

function profilePayload() {
  return {
    name: profileNameInput.value.trim(),
    description: profileDescriptionInput.value.trim(),
    language: profileLanguageInput.value,
    tone: profileToneInput.value,
    detail_level: profileDetailInput.value,
    response_format: profileFormatInput.value,
    constraints: profileConstraintsInput.value
      .split("\n")
      .map((constraint) => constraint.trim())
      .filter(Boolean),
  };
}

function renderQueue() {
  queueCount.textContent = String(queuedMessages.length);
  followupQueue.hidden = queuedMessages.length === 0;
  queuedMessagesContainer.replaceChildren(...queuedMessages.map((queued, index) => {
    const item = document.createElement("article");
    const position = document.createElement("span");
    position.textContent = String(index + 1);
    const content = document.createElement("p");
    content.textContent = queued.content;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.title = "Убрать из очереди";
    remove.setAttribute("aria-label", `Убрать из очереди: ${queued.content}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      queuedMessages = queuedMessages.filter((item) => item.id !== queued.id);
      renderQueue();
      status.textContent = queuedMessages.length
        ? `В очереди: ${queuedMessages.length}`
        : "Очередь очищена";
    });
    item.append(position, content, remove);
    return item;
  }));
}

function enqueueMessage(content) {
  queuedMessages.push({
    id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    content,
  });
  renderQueue();
  status.textContent = `Агент отвечает · в очереди: ${queuedMessages.length}`;
}

function commandMatches() {
  const value = input.value;
  if (!value.startsWith("/") || /\s/.test(value)) return [];
  return memoryCommands.filter((command) => command.name.startsWith(value.toLowerCase()));
}

function hideCommandMenu() {
  commandMenu.hidden = true;
  commandMenu.replaceChildren();
  input.removeAttribute("aria-activedescendant");
}

function selectCommand(command) {
  input.value = `${command.name} `;
  hideCommandMenu();
  input.focus();
}

function renderCommandMenu() {
  const matches = commandMatches();
  if (!matches.length) {
    hideCommandMenu();
    return;
  }

  activeCommandIndex = Math.min(activeCommandIndex, matches.length - 1);
  const options = matches.map((command, index) => {
    const option = document.createElement("button");
    option.type = "button";
    option.id = `memory-command-${index}`;
    option.className = index === activeCommandIndex ? "command-option active" : "command-option";
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", String(index === activeCommandIndex));

    const name = document.createElement("strong");
    name.textContent = command.name;
    const description = document.createElement("span");
    description.textContent = command.description;
    const layer = document.createElement("small");
    layer.className = command.layer;
    layer.textContent = command.layer === "working" ? "WORKING" : "LONG-TERM";
    option.append(name, description, layer);
    option.addEventListener("mousedown", (event) => {
      event.preventDefault();
      selectCommand(command);
    });
    return option;
  });

  commandMenu.replaceChildren(...options);
  commandMenu.hidden = false;
  input.setAttribute("aria-activedescendant", options[activeCommandIndex].id);
}

function parsedMemoryCommand(content) {
  const command = memoryCommands.find((candidate) => (
    content === candidate.name || content.startsWith(`${candidate.name} `)
  ));
  if (!command) return null;
  return { ...command, content: content.slice(command.name.length).trim() };
}

async function executeMemoryCommand(command) {
  if (!command.content) {
    status.textContent = `Добавь текст после ${command.name}`;
    input.focus();
    return;
  }

  setBusy(true);
  status.textContent = "Сохраняю команду в память…";
  try {
    await api("/api/memory", {
      method: "POST",
      body: JSON.stringify({
        layer: command.layer,
        category: command.category,
        content: command.content,
        session_id: command.layer === "working" ? currentSessionId : null,
        profile_id: command.layer === "long_term" ? currentProfileId : null,
        source_session_id: currentSessionId,
        source_text: `${command.name} ${command.content}`,
      }),
    });
    hideCommandMenu();
    const [session, updatedSessions, memory] = await Promise.all([
      api(`/api/chat/sessions/${currentSessionId}`),
      api(`/api/chat/sessions?profile_id=${encodeURIComponent(currentProfileId)}`),
      api(`/api/memory?session_id=${encodeURIComponent(currentSessionId)}`),
    ]);
    sessions = updatedSessions;
    memorySnapshot = memory;
    title.textContent = session.title;
    renderSessions();
    renderMessages(session.messages);
    status.textContent = command.layer === "working"
      ? `Сохранено в рабочую память: ${command.category}`
      : `Сохранено в долговременную память: ${command.category}`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    setBusy(false);
    input.focus();
    void drainQueue();
  }
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

function renderMessages(items, updateState = true) {
  if (updateState) {
    currentMessages = items;
    renderMemory();
  }
  if (!items.length) {
    messages.innerHTML = '<div class="empty"><span>✦</span><h2>Чем помочь?</h2><p>Напишите первую задачу. Перед работой агент коротко познакомится с вами.</p></div>';
    return;
  }
  messages.replaceChildren(...items.map((message) => {
    const article = document.createElement("article");
    const kind = message.kind || "message";
    article.className = `message ${message.role} ${kind}`;
    const label = document.createElement("span");
    label.textContent = kind === "command"
      ? "Команда памяти"
      : (message.role === "user" ? "Вы" : "Агент");
    let content;
    if (kind === "pending") {
      content = document.createElement("div");
      content.className = "typing-indicator";
      content.setAttribute("aria-label", "Агент формирует ответ");
      content.append(
        document.createElement("i"),
        document.createElement("i"),
        document.createElement("i"),
      );
    } else {
      content = document.createElement("p");
      content.textContent = message.content;
    }
    article.append(label, content);
    return article;
  }));
  messages.scrollTop = messages.scrollHeight;
}

function emptyMemory(text) {
  const item = document.createElement("p");
  item.className = "memory-empty";
  item.textContent = text;
  return item;
}

function renderShortTermMemory() {
  const shortTermMessages = currentMessages.filter((message) => (
    (message.kind || "message") === "message"
  ));
  shortTermCount.textContent = String(shortTermMessages.length);
  const recent = shortTermMessages.slice(-6);
  if (!recent.length) {
    shortTermMemory.replaceChildren(emptyMemory("Диалог пока пуст."));
    return;
  }
  shortTermMemory.replaceChildren(...recent.map((message) => {
    const item = document.createElement("article");
    item.className = "memory-item compact";
    const category = document.createElement("strong");
    category.textContent = message.role === "user" ? "user" : "assistant";
    const content = document.createElement("p");
    content.textContent = message.content;
    item.append(category, content);
    return item;
  }));
}

function memoryEntryElement(entry) {
  const item = document.createElement("article");
  item.className = "memory-item";
  const heading = document.createElement("div");
  const category = document.createElement("strong");
  category.textContent = entry.category;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "memory-delete";
  remove.title = "Удалить запись";
  remove.setAttribute("aria-label", `Удалить ${entry.category}`);
  remove.textContent = "×";
  remove.addEventListener("click", () => deleteMemory(entry));
  heading.append(category, remove);
  const content = document.createElement("p");
  content.textContent = entry.content;
  item.append(heading, content);
  return item;
}

function renderMemoryList(container, entries, emptyText) {
  if (!entries.length) {
    container.replaceChildren(emptyMemory(emptyText));
    return;
  }
  container.replaceChildren(...entries.map(memoryEntryElement));
}

function renderMemory() {
  renderShortTermMemory();
  workingCount.textContent = String(memorySnapshot.working.length);
  longTermCount.textContent = String(memorySnapshot.long_term.length);
  renderMemoryList(workingMemory, memorySnapshot.working, "Нет данных задачи.");
  renderMemoryList(longTermMemory, memorySnapshot.long_term, "Нет общих воспоминаний.");
}

async function loadMemory(sessionId) {
  memorySnapshot = await api(`/api/memory?session_id=${encodeURIComponent(sessionId)}`);
  renderMemory();
}

async function deleteMemory(entry) {
  try {
    await api(`/api/memory/${entry.layer}/${entry.id}`, { method: "DELETE" });
    await loadMemory(currentSessionId);
    status.textContent = "Запись памяти удалена";
  } catch (error) {
    status.textContent = error.message;
  }
}

async function openSession(sessionId) {
  if (busy) return;
  try {
    const [session, memory] = await Promise.all([
      api(`/api/chat/sessions/${sessionId}`),
      api(`/api/memory?session_id=${encodeURIComponent(sessionId)}`),
    ]);
    currentSessionId = session.id;
    memorySnapshot = memory;
    title.textContent = session.title;
    renderSessions();
    renderMessages(session.messages);
    status.textContent = "Готов";
    input.focus();
  } catch (error) {
    status.textContent = error.message;
  }
}

async function sendChatMessage(content) {
  setBusy(true);
  const optimisticMessages = [
    ...currentMessages,
    { role: "user", kind: "message", content },
    { role: "assistant", kind: "pending", content: "" },
  ];
  renderMessages(optimisticMessages, false);
  try {
    const result = await api(`/api/chat/sessions/${currentSessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    });
    const [session, memory, loadedProfiles] = await Promise.all([
      api(`/api/chat/sessions/${currentSessionId}`),
      api(`/api/memory?session_id=${encodeURIComponent(currentSessionId)}`),
      api("/api/profiles"),
    ]);
    profiles = loadedProfiles;
    memorySnapshot = memory;
    sessions = sessions.filter((item) => item.id !== result.session.id);
    sessions.unshift(result.session);
    title.textContent = result.session.title;
    renderSessions();
    renderProfiles();
    renderMessages(session.messages);
    status.textContent = "Готов";
  } catch (error) {
    renderMessages(currentMessages);
    status.textContent = error.message;
  } finally {
    setBusy(false);
    input.focus();
    void drainQueue();
  }
}

async function dispatchContent(content) {
  const memoryCommand = parsedMemoryCommand(content);
  if (memoryCommand) {
    await executeMemoryCommand(memoryCommand);
  } else {
    await sendChatMessage(content);
  }
}

async function drainQueue() {
  if (busy || !queuedMessages.length || !currentSessionId) return;
  const [next] = queuedMessages.splice(0, 1);
  renderQueue();
  await dispatchContent(next.content);
}

async function createSession() {
  if (busy) return;
  try {
    const session = await api("/api/chat/sessions", {
      method: "POST",
      body: JSON.stringify({ profile_id: currentProfileId }),
    });
    sessions.unshift(session);
    await openSession(session.id);
  } catch (error) {
    status.textContent = error.message;
  }
}

async function loadSessions() {
  try {
    sessions = await api(`/api/chat/sessions?profile_id=${encodeURIComponent(currentProfileId)}`);
    if (sessions.length) await openSession(sessions[0].id);
    else await createSession();
  } catch (error) {
    status.textContent = error.message;
  }
}

async function activateProfile(profileId) {
  currentProfileId = profileId;
  currentSessionId = null;
  currentMessages = [];
  memorySnapshot = { working: [], long_term: [] };
  queuedMessages = [];
  rememberSelectedProfile();
  renderProfiles();
  renderQueue();
  renderMessages([]);
  title.textContent = "Новый чат";
  await loadSessions();
}

async function loadProfiles() {
  try {
    profiles = await api("/api/profiles");
    let savedProfileId = null;
    try {
      savedProfileId = localStorage.getItem("rubik-active-profile");
    } catch (_) {
      // Use the default profile when storage is unavailable.
    }
    const selected = profiles.some((profile) => profile.id === savedProfileId)
      ? savedProfileId
      : profiles[0]?.id;
    if (!selected) throw new Error("Не удалось создать основной профиль");
    await activateProfile(selected);
  } catch (error) {
    status.textContent = error.message;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || !currentSessionId) return;
  const memoryCommand = parsedMemoryCommand(content);
  if (memoryCommand && !memoryCommand.content) {
    await executeMemoryCommand(memoryCommand);
    return;
  }
  input.value = "";
  hideCommandMenu();
  if (busy) {
    enqueueMessage(content);
    input.focus();
    return;
  }
  await dispatchContent(content);
});

input.addEventListener("keydown", (event) => {
  const matches = commandMatches();
  if (!commandMenu.hidden && matches.length) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      activeCommandIndex = (activeCommandIndex + direction + matches.length) % matches.length;
      renderCommandMenu();
      return;
    }
    if (event.key === "Tab" || event.key === "Enter") {
      event.preventDefault();
      selectCommand(matches[activeCommandIndex]);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      hideCommandMenu();
      return;
    }
  }
  if (
    event.key === "Enter"
    && !event.shiftKey
    && !event.isComposing
  ) {
    event.preventDefault();
    if (input.value.trim()) form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  activeCommandIndex = 0;
  renderCommandMenu();
});

newButton.addEventListener("click", createSession);
profileSelect.addEventListener("change", async () => {
  if (busy) {
    profileSelect.value = currentProfileId;
    return;
  }
  await activateProfile(profileSelect.value);
});
newProfileButton.addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  status.textContent = "Создаю автопрофиль…";
  try {
    const created = await api("/api/profiles/auto", { method: "POST" });
    profiles = await api("/api/profiles");
    setBusy(false);
    await activateProfile(created.id);
    status.textContent = "Напишите первую задачу — агент начнёт интервью";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    setBusy(false);
  }
});
editProfileButton.addEventListener("click", () => openProfileDialog(activeProfile()));
closeProfileDialogButton.addEventListener("click", closeProfileDialog);
cancelProfileButton.addEventListener("click", closeProfileDialog);
deleteProfileButton.addEventListener("click", async () => {
  const profile = activeProfile();
  if (!profile || profile.id === "default" || busy) return;
  const confirmed = window.confirm(
    `Удалить профиль «${profile.name}» вместе со всеми его чатами и памятью? Это действие нельзя отменить.`,
  );
  if (!confirmed) return;

  deleteProfileButton.disabled = true;
  setBusy(true);
  status.textContent = "Удаляю профиль…";
  try {
    await api(`/api/profiles/${encodeURIComponent(profile.id)}`, { method: "DELETE" });
    closeProfileDialog();
    profiles = await api("/api/profiles");
    const fallbackProfile = profiles[0];
    if (!fallbackProfile) throw new Error("После удаления не осталось профилей");
    setBusy(false);
    await activateProfile(fallbackProfile.id);
    status.textContent = "Профиль и связанные данные удалены";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    deleteProfileButton.disabled = false;
    setBusy(false);
  }
});
profileDialog.addEventListener("click", (event) => {
  if (event.target === profileDialog) closeProfileDialog();
});
profileForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const profileId = profileIdInput.value;
  const saveButton = profileForm.querySelector("button[type='submit']");
  saveButton.disabled = true;
  status.textContent = "Сохраняю профиль…";
  try {
    const saved = await api(`/api/profiles/${encodeURIComponent(profileId)}`, {
      method: "PUT",
      body: JSON.stringify(profilePayload()),
    });
    profiles = await api("/api/profiles");
    closeProfileDialog();
    await activateProfile(saved.id);
    status.textContent = "Профиль скорректирован";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    saveButton.disabled = false;
  }
});
clearDatabaseButton.addEventListener("click", async () => {
  if (busy || !window.confirm("Удалить чаты активного профиля и их рабочую память? Долговременная память сохранится.")) return;

  setBusy(true);
  try {
    await api(`/api/chat/sessions?profile_id=${encodeURIComponent(currentProfileId)}`, { method: "DELETE" });
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

renderMemory();
renderQueue();
loadProfiles();
