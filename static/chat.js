let currentChatId = null;
let currentMessages = [];

const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const chatHistory = document.getElementById("chatHistory");
const typingIndicator = document.getElementById("typingIndicator");
const chatList = document.getElementById("chatList");
const aboutInfo = document.getElementById("aboutInfo");

const temperatureSlider = document.getElementById("temperatureSlider");
const temperatureValue = document.getElementById("temperatureValue");
const contextSlider = document.getElementById("contextSlider");
const contextValue = document.getElementById("contextValue");

const themeToggle = document.getElementById("themeToggle");
const clearChatBtn = document.getElementById("clearChat");
const deleteChatBtn = document.getElementById("deleteChat");
const exportChatBtn = document.getElementById("exportChat");
const newChatBtn = document.getElementById("newChat");

temperatureSlider.addEventListener("input", () => {
  temperatureValue.textContent = temperatureSlider.value;
});

contextSlider.addEventListener("input", () => {
  contextValue.textContent = contextSlider.value;
});

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function showTyping(show) {
  typingIndicator.classList.toggle("hidden", !show);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function nowIso() {
  return new Date().toISOString();
}

function getSourceFilename(source, fallbackIndex) {
  return (
    source.doc_id ||
    source.id ||
    source.source ||
    source.filename ||
    `Document ${fallbackIndex}`
  );
}

function getSourceScore(source) {
  const candidates = [
    source.score,
    source.similarity,
    source.similarity_score,
    source.rerank_score,
    source.distance,
  ];

  for (const value of candidates) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value.toFixed(3);
    }
    if (typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value))) {
      return Number(value).toFixed(3);
    }
  }

  return "N/A";
}

function getSourceText(source) {
  return (
    source.text ||
    source.content ||
    source.chunk ||
    source.preview ||
    ""
  ).trim();
}

function buildSourcesHtml(sources) {
  if (!sources || sources.length === 0) {
    return "";
  }

  const cards = sources.map((source, index) => {
    const filename = escapeHtml(getSourceFilename(source, index + 1));
    const score = escapeHtml(getSourceScore(source));
    const fullText = escapeHtml(getSourceText(source));
    const preview = escapeHtml(getSourceText(source).slice(0, 260));

    return `
      <div class="source-card">
        <div class="source-card-header">
          <div class="source-filename">📄 ${filename}</div>
          <div class="source-score">Score: ${score}</div>
        </div>

        <div class="source-preview">${preview || "No preview available."}${fullText.length > 260 ? "..." : ""}</div>

        <details class="source-details">
          <summary>View full context</summary>
          <div class="source-fulltext">${fullText || "No full context available."}</div>
        </details>
      </div>
    `;
  }).join("");

  return `
    <div class="message-sources">
      <details>
        <summary>📚 Sources (${sources.length})</summary>
        <div class="sources-list">
          ${cards}
        </div>
      </details>
    </div>
  `;
}

function renderMessages(messages) {
  currentMessages = messages || [];
  chatHistory.innerHTML = "";

  if (!messages || messages.length === 0) {
    chatHistory.innerHTML = `
      <div class="empty-state" id="emptyState">
        <div class="empty-state-icon">💬</div>
        <h2>Start a conversation</h2>
        <p>Ask about product specifications, setup guides, or troubleshooting steps.</p>
      </div>
    `;
    return;
  }

  messages.forEach((msg) => {
    const wrapper = document.createElement("div");
    wrapper.className = `message-row ${msg.role === "user" ? "user-row" : "assistant-row"}`;

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${msg.role === "user" ? "user-bubble" : "assistant-bubble"}`;

    const roleLabel = msg.role === "user" ? "You" : "Assistant";
    const textHtml = escapeHtml(msg.content).replace(/\n/g, "<br>");
    const timeHtml = escapeHtml(formatTime(msg.timestamp));
    const sourcesHtml = msg.role === "assistant" ? buildSourcesHtml(msg.sources || []) : "";

    bubble.innerHTML = `
      <div class="message-role">${roleLabel}</div>
      <div class="message-text">${textHtml}</div>
      ${sourcesHtml}
      <div class="message-time">${timeHtml}</div>
    `;

    wrapper.appendChild(bubble);
    chatHistory.appendChild(wrapper);
  });

  scrollToBottom();
}

function renderChatList(chats) {
  chatList.innerHTML = "";

  if (!chats || chats.length === 0) {
    chatList.innerHTML = `<div class="history-empty">No chats yet</div>`;
    return;
  }

  chats.forEach((chat) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `chat-list-item ${chat.chat_id === currentChatId ? "active" : ""}`;
    item.innerHTML = `
      <div class="chat-list-title">${escapeHtml(chat.title || "New Chat")}</div>
      <div class="chat-list-meta">${chat.message_count || 0} messages</div>
    `;
    item.addEventListener("click", () => loadChat(chat.chat_id));
    chatList.appendChild(item);
  });
}

async function loadAbout() {
  try {
    const response = await fetch("/about");
    const data = await response.json();

    aboutInfo.innerHTML = `
      <p><strong>App:</strong> ${escapeHtml(data.app || "")}</p>
      <p><strong>Author:</strong> ${escapeHtml(data.author || "")}</p>
      <p><strong>Course:</strong> ${escapeHtml(data.course || "")}</p>
      <p><strong>Version:</strong> ${escapeHtml(data.version || "")}</p>
      <p>${escapeHtml(data.description || "")}</p>
    `;
  } catch (_error) {
    aboutInfo.innerHTML = `<p>Unable to load author details.</p>`;
  }
}

async function loadChats() {
  try {
    const response = await fetch("/chats");
    const chats = await response.json();
    renderChatList(chats);
  } catch (error) {
    console.error("Failed to load chats", error);
  }
}

async function createNewChat() {
  const response = await fetch("/chats/new", { method: "POST" });
  const data = await response.json();
  currentChatId = data.chat_id;
  renderMessages([]);
  await loadChats();
  questionInput.focus();
}

async function loadChat(chatId) {
  try {
    const response = await fetch(`/chats/${chatId}`);
    if (!response.ok) return;

    const chat = await response.json();
    currentChatId = chat.chat_id;

    const messages = (chat.messages || []).map((msg) => ({
      ...msg,
      timestamp: msg.timestamp || nowIso(),
      sources: msg.sources || [],
    }));

    renderMessages(messages);
    await loadChats();
  } catch (error) {
    console.error("Failed to load chat", error);
  }
}

function appendMessage(role, content, sources = []) {
  const updated = [
    ...currentMessages,
    {
      role,
      content,
      timestamp: nowIso(),
      sources,
    },
  ];
  renderMessages(updated);
}

async function sendMessage(question) {
  if (!question.trim()) return;

  if (!currentChatId) {
    await createNewChat();
  }

  appendMessage("user", question);
  showTyping(true);

  try {
    const response = await fetch("/rag", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        n_context_docs: Number(contextSlider.value),
        temperature: Number(temperatureSlider.value),
        chat_id: currentChatId,
      }),
    });

    const data = await response.json();

    showTyping(false);

    if (!response.ok) {
      appendMessage("assistant", `Error: ${data.detail || "Something went wrong."}`, []);
      return;
    }

    appendMessage("assistant", data.answer || "No answer returned.", data.context || []);
    await loadChats();
  } catch (_error) {
    showTyping(false);
    appendMessage("assistant", "Network or server error.", []);
  }
}

async function clearCurrentChat() {
  if (!currentChatId) return;

  try {
    await fetch(`/chats/${currentChatId}/clear`, {
      method: "POST",
    });
    renderMessages([]);
    await loadChats();
  } catch (error) {
    console.error("Failed to clear chat", error);
  }
}

async function deleteCurrentChat() {
  if (!currentChatId) return;

  try {
    await fetch(`/chats/${currentChatId}`, {
      method: "DELETE",
    });
    currentChatId = null;
    renderMessages([]);
    await loadChats();
  } catch (error) {
    console.error("Failed to delete chat", error);
  }
}

function exportCurrentChat() {
  if (!currentMessages || currentMessages.length === 0) return;

  const lines = currentMessages.map((m) => {
    const prefix = m.role === "user" ? "You" : "Assistant";
    const time = formatTime(m.timestamp);
    const sourcesBlock =
      m.role === "assistant" && m.sources && m.sources.length > 0
        ? `\nSources:\n${m.sources
            .map((s, index) => {
              const filename = getSourceFilename(s, index + 1);
              const score = getSourceScore(s);
              const text = getSourceText(s).slice(0, 300);
              return `- ${filename} (score: ${score})\n  ${text}`;
            })
            .join("\n")}`
        : "";

    return `[${time}] ${prefix}: ${m.content}${sourcesBlock}`;
  });

  const blob = new Blob([lines.join("\n\n")], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "chat-export.txt";
  a.click();

  URL.revokeObjectURL(url);
}

function applyTheme(theme) {
  document.body.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
}

function toggleTheme() {
  const current = document.body.getAttribute("data-theme") || "light";
  applyTheme(current === "light" ? "dark" : "light");
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = "";
  await sendMessage(question);
});

questionInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    questionInput.value = "";
    await sendMessage(question);
  }
});

newChatBtn.addEventListener("click", createNewChat);
clearChatBtn.addEventListener("click", clearCurrentChat);
deleteChatBtn.addEventListener("click", deleteCurrentChat);
exportChatBtn.addEventListener("click", exportCurrentChat);
themeToggle.addEventListener("click", toggleTheme);

window.addEventListener("DOMContentLoaded", async () => {
  const savedTheme = localStorage.getItem("theme") || "light";
  applyTheme(savedTheme);

  await loadAbout();
  await loadChats();
  await createNewChat();
});