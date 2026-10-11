// AI Roleplayer — front-end. Vanilla JS, no build step.
// Talks to the local FastAPI backend, which proxies to Ollama.

const $ = (sel) => document.querySelector(sel);

const state = {
  characters: [],
  stories: [],
  story: null,        // the currently open story (full object with messages)
  model: null,
  streaming: false,
  abort: null,
  editingCharId: null,
};

// ---------- persistence of small prefs ----------
const prefs = {
  get playerName() { return localStorage.getItem("rp_player") || ""; },
  set playerName(v) { try { localStorage.setItem("rp_player", v); } catch {} },
  get temp() { return parseFloat(localStorage.getItem("rp_temp") || "0.8"); },
  set temp(v) { try { localStorage.setItem("rp_temp", v); } catch {} },
  get model() { return localStorage.getItem("rp_model") || ""; },
  set model(v) { try { localStorage.setItem("rp_model", v); } catch {} },
};

// ---------- API ----------
const api = {
  async models() { return (await fetch("/api/models")).json(); },
  async characters() { return (await fetch("/api/characters")).json(); },
  async saveCharacter(c) {
    const url = c.id ? `/api/characters/${c.id}` : "/api/characters";
    const method = c.id ? "PUT" : "POST";
    return (await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(c) })).json();
  },
  async deleteCharacter(id) { return fetch(`/api/characters/${id}`, { method: "DELETE" }); },
  async stories() { return (await fetch("/api/stories")).json(); },
  async getStory(id) { return (await fetch(`/api/stories/${id}`)).json(); },
  async createStory(s) {
    return (await fetch("/api/stories", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(s) })).json();
  },
  async saveStory(s) {
    return (await fetch(`/api/stories/${s.id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(s) })).json();
  },
  async deleteStory(id) { return fetch(`/api/stories/${id}`, { method: "DELETE" }); },
};

// ---------- system prompt ----------
function buildSystemPrompt(char) {
  const player = prefs.playerName.trim() || "the user";
  const lines = [
    `You are ${char.name}, a character in an immersive roleplay with ${player}.`,
  ];
  if (char.description) lines.push(`\nAbout you: ${char.description}`);
  if (char.personality) lines.push(`Your personality: ${char.personality}`);
  if (char.scenario) lines.push(`Scenario: ${char.scenario}`);
  lines.push(
    `\nStay fully in character as ${char.name} at all times. Write vivid, immersive prose.`,
    `Describe your actions, expressions, and the surroundings. Wrap actions and narration in *asterisks* and keep spoken dialogue in plain text.`,
    `Advance the story and react naturally to ${player}. Never break character, never mention that you are an AI, and never speak or act for ${player} — only play your own role and the world around them.`
  );
  return lines.join("\n");
}

// ---------- rendering ----------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
// Render *actions* as emphasised text (after escaping).
function formatContent(s) {
  return escapeHtml(s).replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function addMessageBubble(role, content) {
  const empty = $("#empty-state");
  if (empty) empty.remove();
  const char = currentCharacter();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "user" ? "user" : "ai"}`;
  const who = role === "user" ? "🧑" : (char ? char.avatar : "🎭");
  wrap.innerHTML = `<div class="who">${escapeHtml(who)}</div><div class="bubble">${formatContent(content)}</div>`;
  $("#messages").appendChild(wrap);
  scrollToBottom();
  return wrap.querySelector(".bubble");
}

function scrollToBottom() {
  const m = $("#messages");
  m.scrollTop = m.scrollHeight;
}

function renderMessages() {
  const box = $("#messages");
  box.innerHTML = "";
  if (!state.story || state.story.messages.length === 0) {
    box.innerHTML = `<div class="empty-state"><div class="big">${currentCharacter()?.avatar || "🎭"}</div><h2>The scene is set</h2><p>Say or do something to begin.</p></div>`;
    return;
  }
  for (const m of state.story.messages) addMessageBubble(m.role, m.content);
}

function currentCharacter() {
  if (!state.story) return null;
  return state.characters.find((c) => c.id === state.story.character_id) || null;
}

// ---------- sidebar lists ----------
function renderCharacters() {
  const box = $("#character-list");
  box.innerHTML = "";
  if (state.characters.length === 0) {
    box.innerHTML = `<div class="list-item" style="color:var(--muted);cursor:default">No characters yet</div>`;
    return;
  }
  for (const c of state.characters) {
    const el = document.createElement("div");
    el.className = "list-item";
    el.innerHTML = `
      <span class="emoji">${escapeHtml(c.avatar || "🎭")}</span>
      <div class="meta"><div class="title">${escapeHtml(c.name)}</div>
      <div class="sub">${escapeHtml(c.description || "")}</div></div>
      <button class="del" title="Edit">✎</button>`;
    el.querySelector(".del").onclick = (e) => { e.stopPropagation(); openCharModal(c); };
    el.onclick = () => startStoryWith(c);
    box.appendChild(el);
  }
}

function renderStories() {
  const box = $("#story-list");
  box.innerHTML = "";
  if (state.stories.length === 0) {
    box.innerHTML = `<div class="list-item" style="color:var(--muted);cursor:default">No saved stories</div>`;
    return;
  }
  for (const s of state.stories) {
    const el = document.createElement("div");
    el.className = "list-item" + (state.story && s.id === state.story.id ? " active" : "");
    el.innerHTML = `
      <span class="emoji">📖</span>
      <div class="meta"><div class="title">${escapeHtml(s.title)}</div>
      <div class="sub">${escapeHtml(s.character_name || "")} · ${s.turns} msgs</div></div>
      <button class="del" title="Delete">🗑</button>`;
    el.querySelector(".del").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${s.title}"? This can't be undone.`)) return;
      await api.deleteStory(s.id);
      if (state.story && state.story.id === s.id) { state.story = null; renderMessages(); updateHeader(); $("#composer").classList.add("hidden"); }
      await refreshStories();
    };
    el.onclick = () => openStory(s.id);
    box.appendChild(el);
  }
}

function updateHeader() {
  const char = currentCharacter();
  $("#active-avatar").textContent = char ? char.avatar : "🎭";
  $("#active-name").textContent = char ? char.name : (state.story ? state.story.title : "Pick a character to begin");
  $("#edit-active-char").classList.toggle("hidden", !char);
}

// ---------- story flow ----------
async function startStoryWith(char) {
  const messages = [];
  if (char.greeting) messages.push({ role: "assistant", content: char.greeting });
  const created = await api.createStory({
    title: char.name,
    character_id: char.id,
    character_name: char.name,
    model: state.model,
    messages,
  });
  state.story = created;
  await refreshStories();
  renderMessages();
  updateHeader();
  $("#composer").classList.remove("hidden");
  $("#input").focus();
}

async function openStory(id) {
  state.story = await api.getStory(id);
  renderStories();
  renderMessages();
  updateHeader();
  $("#composer").classList.remove("hidden");
}

async function sendMessage(text) {
  if (!state.story || !state.model || state.streaming) return;
  state.story.messages.push({ role: "user", content: text });
  addMessageBubble("user", text);

  const bubble = addMessageBubble("assistant", "");
  bubble.classList.add("cursor");
  let full = "";

  state.streaming = true;
  toggleSendStop(true);
  state.abort = new AbortController();

  const char = currentCharacter();
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.model,
        system: char ? buildSystemPrompt(char) : null,
        temperature: prefs.temp,
        messages: state.story.messages.filter((m) => m.role === "user" || m.role === "assistant"),
      }),
      signal: state.abort.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.replace(/^data: /, "").trim();
        if (!line) continue;
        let obj;
        try { obj = JSON.parse(line); } catch { continue; }
        if (obj.token) {
          full += obj.token;
          bubble.innerHTML = formatContent(full);
          scrollToBottom();
        } else if (obj.error) {
          full += (full ? "\n\n" : "") + "⚠️ " + obj.error;
          bubble.innerHTML = formatContent(full);
        } else if (obj.done) {
          // stream finished
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      full += (full ? "\n\n" : "") + "⚠️ " + e.message;
      bubble.innerHTML = formatContent(full);
    }
  } finally {
    bubble.classList.remove("cursor");
    state.streaming = false;
    state.abort = null;
    toggleSendStop(false);

    if (full.trim()) {
      state.story.messages.push({ role: "assistant", content: full });
      await api.saveStory(state.story);
      refreshStories();
    }
  }
}

function toggleSendStop(streaming) {
  $("#send-btn").classList.toggle("hidden", streaming);
  $("#stop-btn").classList.toggle("hidden", !streaming);
}

// ---------- character modal ----------
function openCharModal(char) {
  state.editingCharId = char ? char.id : null;
  $("#char-modal-title").textContent = char ? `Edit ${char.name}` : "New Character";
  $("#c-avatar").value = char?.avatar || "🎭";
  $("#c-name").value = char?.name || "";
  $("#c-description").value = char?.description || "";
  $("#c-personality").value = char?.personality || "";
  $("#c-scenario").value = char?.scenario || "";
  $("#c-greeting").value = char?.greeting || "";
  $("#char-delete").classList.toggle("hidden", !char);
  $("#char-modal").classList.remove("hidden");
  $("#c-name").focus();
}
function closeCharModal() { $("#char-modal").classList.add("hidden"); }

async function saveCharFromModal() {
  const name = $("#c-name").value.trim();
  if (!name) { alert("Give your character a name."); return; }
  const char = {
    id: state.editingCharId,
    avatar: $("#c-avatar").value.trim() || "🎭",
    name,
    description: $("#c-description").value.trim(),
    personality: $("#c-personality").value.trim(),
    scenario: $("#c-scenario").value.trim(),
    greeting: $("#c-greeting").value.trim(),
  };
  await api.saveCharacter(char);
  closeCharModal();
  await refreshCharacters();
}

// ---------- models ----------
async function refreshModels() {
  const res = await api.models();
  const sel = $("#model-select");
  $("#ollama-banner").classList.toggle("hidden", res.ok);
  sel.innerHTML = "";
  if (!res.ok || res.models.length === 0) {
    sel.innerHTML = `<option value="">— no models —</option>`;
    state.model = null;
    return;
  }
  for (const m of res.models) {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  }
  const want = prefs.model && res.models.includes(prefs.model) ? prefs.model : res.models[0];
  sel.value = want;
  state.model = want;
}

async function refreshCharacters() { state.characters = await api.characters(); renderCharacters(); updateHeader(); }
async function refreshStories() { state.stories = await api.stories(); renderStories(); }

// ---------- wiring ----------
function init() {
  // prefs
  $("#player-name").value = prefs.playerName;
  $("#temp-slider").value = prefs.temp;
  $("#temp-val").textContent = prefs.temp;

  $("#player-name").addEventListener("input", (e) => { prefs.playerName = e.target.value; });
  $("#temp-slider").addEventListener("input", (e) => { prefs.temp = e.target.value; $("#temp-val").textContent = e.target.value; });
  $("#model-select").addEventListener("change", (e) => { state.model = e.target.value; prefs.model = e.target.value; });

  $("#new-story-btn").addEventListener("click", () => {
    alert("Pick a character from the list to start a new story. ✎ edits one; + creates one.");
  });
  $("#new-char-btn").addEventListener("click", () => openCharModal(null));
  $("#edit-active-char").addEventListener("click", () => { const c = currentCharacter(); if (c) openCharModal(c); });

  // modal
  $("#char-cancel").addEventListener("click", closeCharModal);
  $("#char-save").addEventListener("click", saveCharFromModal);
  $("#char-delete").addEventListener("click", async () => {
    if (!state.editingCharId) return;
    if (!confirm("Delete this character? Saved stories with it will remain.")) return;
    await api.deleteCharacter(state.editingCharId);
    closeCharModal();
    await refreshCharacters();
  });
  $("#char-modal").addEventListener("click", (e) => { if (e.target.id === "char-modal") closeCharModal(); });

  // composer
  const input = $("#input");
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    if (!state.model) { alert("No model selected. Pull one in Ollama first (e.g. `ollama pull llama3.1`)."); return; }
    input.value = ""; input.style.height = "auto";
    sendMessage(text);
  });
  $("#stop-btn").addEventListener("click", () => { if (state.abort) state.abort.abort(); });

  $("#retry-ollama").addEventListener("click", (e) => { e.preventDefault(); refreshModels(); });

  // initial load
  refreshModels();
  refreshCharacters();
  refreshStories();
}

init();
