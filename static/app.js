// AI Roleplayer — front-end. Vanilla JS, no build step.
// The server owns story state and assembles the prompt; this client sends
// actions (send/regenerate/swipe/continue), streams the reply, and re-syncs
// from the server after each turn so the two never clobber each other.

const $ = (s) => document.querySelector(s);

const state = {
  characters: [], stories: [], personas: [], lorebooks: [],
  story: null, model: null,
  streaming: false, abort: null,
  editingCharId: null, editingPersonaId: null, editingLoreId: null,
  editingCharImage: null, loreDraft: null, tts: false,
};

const prefs = {
  get model() { return localStorage.getItem("rp_model") || ""; },
  set model(v) { try { localStorage.setItem("rp_model", v); } catch {} },
};

// ---------- API ----------
async function jget(u) { const r = await fetch(u); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); }
async function jsend(u, method, body) {
  const r = await fetch(u, { method, headers: { "Content-Type": "application/json" }, body: body != null ? JSON.stringify(body) : undefined });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
const api = {
  models: () => jget("/api/models"),
  presets: (m) => jget("/api/presets?model=" + encodeURIComponent(m || "")),
  characters: () => jget("/api/characters"),
  saveCharacter: (c) => jsend(c.id ? `/api/characters/${c.id}` : "/api/characters", c.id ? "PUT" : "POST", c),
  deleteCharacter: (id) => fetch(`/api/characters/${id}`, { method: "DELETE" }),
  importCharacter: (filename, data_b64) => jsend("/api/characters/import", "POST", { filename, data_b64 }),
  setAvatar: (cid, data_b64, ext) => jsend(`/api/characters/${cid}/avatar`, "POST", { data_b64, ext }),
  clearAvatarImg: (cid) => fetch(`/api/characters/${cid}/avatar`, { method: "DELETE" }),
  stories: () => jget("/api/stories"),
  getStory: (id) => jget(`/api/stories/${id}`),
  createStory: (s) => jsend("/api/stories", "POST", s),
  importStory: (filename, data_b64) => jsend("/api/stories/import", "POST", { filename, data_b64 }),
  deleteStory: (id) => fetch(`/api/stories/${id}`, { method: "DELETE" }),
  storyMeta: (id, patch) => jsend(`/api/stories/${id}/meta`, "PUT", patch),
  editMessage: (id, idx, content) => jsend(`/api/stories/${id}/message/${idx}`, "PUT", { content }),
  selectSwipe: (id, idx, swipe_id) => jsend(`/api/stories/${id}/message/${idx}/swipe`, "PUT", { swipe_id }),
  deleteFrom: (id, idx) => fetch(`/api/stories/${id}/message/${idx}`, { method: "DELETE" }),
  personas: () => jget("/api/personas"),
  savePersona: (p) => jsend(p.id ? `/api/personas/${p.id}` : "/api/personas", p.id ? "PUT" : "POST", p),
  deletePersona: (id) => fetch(`/api/personas/${id}`, { method: "DELETE" }),
  lorebooks: () => jget("/api/lorebooks"),
  saveLorebook: (b) => jsend(b.id ? `/api/lorebooks/${b.id}` : "/api/lorebooks", b.id ? "PUT" : "POST", b),
  deleteLorebook: (id) => fetch(`/api/lorebooks/${id}`, { method: "DELETE" }),
};

// ---------- helpers ----------
function escapeHtml(s) { return (s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function formatContent(s) { return escapeHtml(s).replace(/\*([^*]+)\*/g, "<em>$1</em>"); }
function stripThink(s) {
  s = (s || "").replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, "");
  const low = s.toLowerCase();
  if (low.includes("<think>") && !low.includes("</think>")) s = s.slice(0, low.indexOf("<think>"));
  return s;
}
function estTokens(s) { return Math.ceil((s || "").length / 4); }
function currentCharacter() { return state.story ? state.characters.find((c) => c.id === state.story.character_id) : null; }
function avatarHtml(entity, fallback = "🎭") {
  if (entity && entity.avatar_image) return `<img class="av" src="${escapeHtml(entity.avatar_image)}" alt="" />`;
  return escapeHtml((entity && entity.avatar) || fallback);
}
function speak(text) {
  if (!("speechSynthesis" in window) || !text) return;
  const clean = text.replace(/\*([^*]+)\*/g, "$1").replace(/[*_`#>~]/g, "");
  try { window.speechSynthesis.cancel(); window.speechSynthesis.speak(new SpeechSynthesisUtterance(clean)); } catch {}
}

// ---------- rendering: messages ----------
function renderMessages() {
  const box = $("#messages");
  box.innerHTML = "";
  const msgs = state.story ? state.story.messages : [];
  if (!msgs || msgs.length === 0) {
    box.innerHTML = `<div class="empty-state"><div class="big">${currentCharacter()?.avatar || "🎭"}</div><h2>The scene is set</h2><p>Say or do something to begin.</p></div>`;
    return;
  }
  const char = currentCharacter();
  msgs.forEach((m, i) => box.appendChild(buildBubble(m, i, char, i === msgs.length - 1)));
  scrollToBottom();
}

function buildBubble(m, i, char, isLast) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${m.role === "user" ? "user" : "ai"}`;
  const who = document.createElement("div");
  who.className = "who";
  who.innerHTML = m.role === "user" ? "🧑" : avatarHtml(char);
  const col = document.createElement("div");
  col.className = "col";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = formatContent(m.content);
  col.appendChild(bubble);
  col.appendChild(buildControls(m, i, isLast, bubble));
  wrap.appendChild(who);
  wrap.appendChild(col);
  return wrap;
}

function buildControls(m, i, isLast, bubble) {
  const ctl = document.createElement("div");
  ctl.className = "msg-controls";
  const btn = (label, title, fn) => { const b = document.createElement("button"); b.textContent = label; b.title = title; b.onclick = fn; return b; };

  const isAssistant = m.role === "assistant";
  if (isAssistant && isLast) {
    const swipes = m.swipes || [m.content];
    const nav = document.createElement("span");
    nav.className = "swipe-nav";
    const left = btn("‹", "Previous version", () => swipe(i, -1));
    left.disabled = (m.swipe_id || 0) <= 0;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `${(m.swipe_id || 0) + 1}/${swipes.length}`;
    const right = btn("›", "Next version / generate new", () => swipe(i, 1));
    nav.append(left, count, right);
    ctl.appendChild(nav);
    ctl.appendChild(btn("↻", "Regenerate", () => chatAction("regenerate")));
    ctl.appendChild(btn("↳", "Continue", () => chatAction("continue")));
  }
  if (isAssistant && "speechSynthesis" in window) ctl.appendChild(btn("🔊", "Read aloud", () => speak(m.content)));
  ctl.appendChild(btn("✎", "Edit", () => startEdit(i, bubble)));
  ctl.appendChild(btn("🗑", "Delete from here", () => deleteFrom(i)));
  return ctl;
}

function scrollToBottom() { const m = $("#messages"); m.scrollTop = m.scrollHeight; }

function updateHeader() {
  const char = currentCharacter();
  $("#active-avatar").innerHTML = char ? avatarHtml(char) : "🎭";
  $("#active-name").textContent = char ? char.name : (state.story ? state.story.title : "Pick a character to begin");
  const open = !!state.story;
  $("#edit-active-char").classList.toggle("hidden", !char);
  $("#settings-btn").classList.toggle("hidden", !open);
  $("#composer").classList.toggle("hidden", !open);
  $("#an-bar").classList.toggle("hidden", !open);
  if (open) $("#an-input").value = (state.story.authors_note || {}).text || "";
}

// ---------- inline edit ----------
function startEdit(i, bubble) {
  if (state.streaming) return;
  const m = state.story.messages[i];
  const ta = document.createElement("textarea");
  ta.value = m.content;
  ta.rows = Math.min(12, Math.max(2, m.content.split("\n").length));
  const save = document.createElement("button");
  save.className = "primary"; save.textContent = "Save";
  const cancel = document.createElement("button");
  cancel.className = "ghost"; cancel.textContent = "Cancel";
  const bar = document.createElement("div");
  bar.className = "msg-controls"; bar.style.opacity = 1;
  bar.append(save, cancel);
  bubble.replaceWith(ta);
  ta.after(bar);
  ta.focus();
  cancel.onclick = () => renderMessages();
  save.onclick = async () => {
    await api.editMessage(state.story.id, i, ta.value);
    await reloadStory();
  };
}

// ---------- chat ----------
async function chatAction(action, userText) {
  if (!state.story || state.streaming) return;
  if (!state.story.model) {
    await api.storyMeta(state.story.id, { model: state.model });
    state.story.model = state.model;
  }
  if (!state.story.model) { alert("No model available. Pull one in Ollama first, e.g. `ollama pull mistral-nemo`."); return; }

  // Optimistic UI for a freshly sent user line.
  if (action === "send") {
    state.story.messages.push({ role: "user", content: userText });
  }
  renderMessages();

  // Temp assistant bubble to stream into.
  const tmp = document.createElement("div");
  tmp.className = "msg ai";
  const char = currentCharacter();
  tmp.innerHTML = `<div class="who">${char ? avatarHtml(char) : "🎭"}</div><div class="col"><div class="bubble cursor"></div></div>`;
  if ($("#empty-state")) renderMessages();
  $("#messages").appendChild(tmp);
  const tmpBubble = tmp.querySelector(".bubble");
  scrollToBottom();

  state.streaming = true; toggleSendStop(true);
  state.abort = new AbortController();
  let full = "";
  let errorMsg = null;
  let aborted = false;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ story_id: state.story.id, action, user_message: userText || null, settings: {} }),
      signal: state.abort.signal,
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n"); buf = parts.pop();
      for (const part of parts) {
        const line = part.replace(/^data: /, "").trim();
        if (!line) continue;
        let obj; try { obj = JSON.parse(line); } catch { continue; }
        if (obj.token) { full += obj.token; tmpBubble.innerHTML = formatContent(stripThink(full)); scrollToBottom(); }
        else if (obj.error) { errorMsg = obj.error; tmpBubble.classList.remove("cursor"); tmpBubble.innerHTML = formatContent("⚠️ " + obj.error); }
        else if (obj.done) { updateCtxMeter(obj.prompt_tokens, obj.num_ctx); }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") aborted = true;
    else { errorMsg = e.message; tmpBubble.innerHTML = formatContent("⚠️ " + e.message); }
  } finally {
    state.streaming = false; state.abort = null; toggleSendStop(false);
    await reloadStory();          // re-sync from the server (source of truth)
    refreshStories();
    if (errorMsg) showToast("⚠️ " + errorMsg);
    // The server persists nothing on a failed OR stopped send, so give the
    // user their typed text back either way.
    if ((errorMsg || aborted) && action === "send" && userText) {
      const inp = $("#input"); inp.value = userText; inp.dispatchEvent(new Event("input"));
    }
    // Read the finished reply aloud if TTS is on.
    if (state.tts && !errorMsg && !aborted && state.story) {
      const last = state.story.messages[state.story.messages.length - 1];
      if (last && last.role === "assistant") speak(last.content);
    }
  }
}

function showToast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => t.classList.add("hidden"), 6000);
}

function updateTtsBtn() {
  const b = $("#tts-btn");
  if (b) b.textContent = state.tts ? "🔊 On" : "🔈 Off";
}

function swipe(i, dir) {
  if (state.streaming) return;
  const m = state.story.messages[i];
  const swipes = m.swipes || [m.content];
  let id = m.swipe_id || 0;
  if (dir > 0 && id >= swipes.length - 1) { chatAction("swipe"); return; }  // generate a new alternate
  id = Math.max(0, Math.min(swipes.length - 1, id + dir));
  m.swipe_id = id; m.content = swipes[id];
  renderMessages();
  api.selectSwipe(state.story.id, i, id).catch(() => reloadStory());  // resync if the server rejected it
}

async function deleteFrom(i) {
  if (state.streaming) return;
  if (!confirm("Delete this message and everything after it?")) return;
  await api.deleteFrom(state.story.id, i);
  await reloadStory();
  refreshStories();
}

async function reloadStory() {
  if (!state.story) return;
  state.story = await api.getStory(state.story.id);
  renderMessages();
  updateHeader();
}

function toggleSendStop(streaming) {
  $("#send-btn").classList.toggle("hidden", streaming);
  $("#stop-btn").classList.toggle("hidden", !streaming);
  $("#input").disabled = streaming;
}

function updateCtxMeter(prompt_tokens, num_ctx) {
  const el = $("#ctx-meter");
  if (prompt_tokens == null || !num_ctx) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  el.textContent = `ctx ${prompt_tokens} / ${num_ctx}`;
  const ratio = prompt_tokens / num_ctx;
  el.classList.toggle("warn", ratio >= 0.75 && ratio < 0.95);
  el.classList.toggle("over", ratio >= 0.95);
}

// ---------- story lifecycle ----------
async function startStoryWith(char) {
  const alts = (char.alternate_greetings || []).filter((g) => g && g.trim());
  if (alts.length) { openGreetChooser(char, alts); return; }
  await createStoryWithGreeting(char, char.greeting);
}

async function createStoryWithGreeting(char, greeting) {
  const messages = [];
  if (greeting && greeting.trim()) messages.push({ role: "assistant", content: greeting });
  const created = await api.createStory({
    title: char.name, character_id: char.id, character_name: char.name,
    model: state.model, messages,
  });
  state.story = created;
  await refreshStories();
  renderMessages(); updateHeader();
  $("#ctx-meter").classList.add("hidden");
  $("#input").focus();
}

function openGreetChooser(char, alts) {
  const box = $("#greet-options");
  box.innerHTML = "";
  const options = [{ label: "Default greeting", text: char.greeting || "" }];
  alts.forEach((g, i) => options.push({ label: `Alternate ${i + 1}`, text: g }));
  for (const o of options) {
    const el = document.createElement("div");
    el.className = "greet-option";
    const preview = o.text ? (o.text.length > 400 ? o.text.slice(0, 400) + "…" : o.text) : "(empty opening)";
    el.innerHTML = `<div class="label">${escapeHtml(o.label)}</div>${formatContent(preview)}`;
    el.onclick = () => { $("#greet-modal").classList.add("hidden"); createStoryWithGreeting(char, o.text); };
    box.appendChild(el);
  }
  $("#greet-modal").classList.remove("hidden");
}

async function openStory(id) {
  state.story = await api.getStory(id);
  renderStories(); renderMessages(); updateHeader();
  $("#ctx-meter").classList.add("hidden");
}

// ---------- sidebar lists ----------
function listItem({ emoji, emojiHtml, title, sub, onClick, actions }) {
  const el = document.createElement("div");
  el.className = "list-item";
  const mark = emojiHtml != null ? emojiHtml : escapeHtml(emoji);
  el.innerHTML = `<span class="emoji">${mark}</span><div class="meta"><div class="title">${escapeHtml(title)}</div>${sub ? `<div class="sub">${escapeHtml(sub)}</div>` : ""}</div>`;
  if (onClick) el.onclick = onClick;
  (actions || []).forEach(({ label, title: t, cls, fn }) => {
    const b = document.createElement("button");
    b.className = "act " + (cls || ""); b.textContent = label; b.title = t;
    b.onclick = (e) => { e.stopPropagation(); fn(); };
    el.appendChild(b);
  });
  return el;
}

function renderStories() {
  const box = $("#story-list"); box.innerHTML = "";
  if (!state.stories.length) { box.innerHTML = `<div class="hint">No saved stories yet.</div>`; return; }
  for (const s of state.stories) {
    box.appendChild(listItem({
      emoji: "📖", title: s.title, sub: `${s.character_name || ""} · ${s.turns} msgs`,
      onClick: () => openStory(s.id),
      actions: [
        { label: "⤓", title: "Export", fn: () => window.open(`/api/stories/${s.id}/export`, "_blank") },
        { label: "🗑", title: "Delete", cls: "del", fn: async () => {
          if (!confirm(`Delete "${s.title}"?`)) return;
          await api.deleteStory(s.id);
          if (state.story && state.story.id === s.id) { state.story = null; renderMessages(); updateHeader(); }
          await refreshStories();
        } },
      ],
    }));
    if (state.story && s.id === state.story.id) box.lastChild.classList.add("active");
  }
}

function renderCharacters() {
  const box = $("#character-list"); box.innerHTML = "";
  if (!state.characters.length) { box.innerHTML = `<div class="hint">No characters yet.</div>`; return; }
  for (const c of state.characters) {
    box.appendChild(listItem({
      emojiHtml: avatarHtml(c), title: c.name, sub: c.description || "",
      onClick: () => startStoryWith(c),
      actions: [{ label: "✎", title: "Edit", fn: () => openCharModal(c) }],
    }));
  }
}

function renderPersonas() {
  const box = $("#persona-list"); box.innerHTML = "";
  if (!state.personas.length) { box.innerHTML = `<div class="hint">None. Create one to tell the AI who you are.</div>`; return; }
  for (const p of state.personas) {
    box.appendChild(listItem({ emoji: "🧑", title: p.name, sub: p.description || "", onClick: () => openPersonaModal(p) }));
  }
}

function renderLorebooks() {
  const box = $("#lore-list"); box.innerHTML = "";
  if (!state.lorebooks.length) { box.innerHTML = `<div class="hint">None. World facts that surface on keywords.</div>`; return; }
  for (const b of state.lorebooks) {
    box.appendChild(listItem({ emoji: "📚", title: b.name, sub: `${(b.entries || []).length} entries`, onClick: () => openLoreModal(b) }));
  }
}

// ---------- character modal ----------
function openCharModal(char) {
  state.editingCharId = char ? char.id : null;
  $("#char-modal-title").textContent = char ? `Edit ${char.name}` : "New Character";
  const g = (k, d = "") => (char && char[k] != null ? char[k] : d);
  $("#c-avatar").value = g("avatar", "🎭");
  state.editingCharImage = (char && char.avatar_image) || null;
  renderAvatarPreview();
  $("#c-altgreet").value = ((char && char.alternate_greetings) || []).join("\n---\n");
  $("#c-name").value = g("name");
  $("#c-description").value = g("description");
  $("#c-personality").value = g("personality");
  $("#c-scenario").value = g("scenario");
  $("#c-greeting").value = g("greeting");
  $("#c-example").value = g("example_dialogue");
  $("#c-system").value = g("system_prompt");
  $("#c-phi").value = g("post_history_instructions");
  $("#c-pov").value = g("pov");
  const lbSel = $("#c-lorebook");
  lbSel.innerHTML = `<option value="">None</option>` + state.lorebooks.map((b) => `<option value="${b.id}">${escapeHtml(b.name)}</option>`).join("");
  lbSel.value = g("lorebook_id");
  $("#char-delete").classList.toggle("hidden", !char);
  $("#char-export").classList.toggle("hidden", !char);
  updatePermTokens();
  $("#char-modal").classList.remove("hidden");
  $("#c-name").focus();
}
function closeCharModal() { $("#char-modal").classList.add("hidden"); }
function renderAvatarPreview() {
  const p = $("#c-avatar-preview");
  if (state.editingCharImage) p.innerHTML = `<img src="${escapeHtml(state.editingCharImage)}" alt="" />`;
  else p.textContent = ($("#c-avatar").value || "").trim() || "🎭";
}
function parseAltGreetings(v) {
  const parts = (v || "").split(/\n-{3,}\n/).map((s) => s.trim()).filter(Boolean);
  return parts.length ? parts : null;
}
function updatePermTokens() {
  const t = estTokens($("#c-description").value) + estTokens($("#c-personality").value) + estTokens($("#c-scenario").value);
  const el = $("#perm-tokens");
  el.textContent = `~${t} permanent tokens`;
  el.classList.toggle("warn", t > 1200);
}
async function saveCharFromModal() {
  const name = $("#c-name").value.trim();
  if (!name) { alert("Give your character a name."); return; }
  const char = {
    id: state.editingCharId, avatar: $("#c-avatar").value.trim() || "🎭", name,
    description: $("#c-description").value.trim(), personality: $("#c-personality").value.trim(),
    scenario: $("#c-scenario").value.trim(), greeting: $("#c-greeting").value.trim(),
    example_dialogue: $("#c-example").value, system_prompt: $("#c-system").value,
    post_history_instructions: $("#c-phi").value, pov: $("#c-pov").value,
    lorebook_id: $("#c-lorebook").value || null,
    alternate_greetings: parseAltGreetings($("#c-altgreet").value),
    avatar_image: state.editingCharImage,
  };
  await api.saveCharacter(char);
  closeCharModal();
  await refreshCharacters();
}

// ---------- persona modal ----------
function openPersonaModal(p) {
  state.editingPersonaId = p ? p.id : null;
  $("#persona-modal-title").textContent = p ? `Edit ${p.name}` : "New Persona";
  $("#p-name").value = p ? p.name : "";
  $("#p-desc").value = p ? (p.description || "") : "";
  $("#persona-delete").classList.toggle("hidden", !p);
  $("#persona-modal").classList.remove("hidden");
  $("#p-name").focus();
}
async function savePersonaFromModal() {
  const name = $("#p-name").value.trim();
  if (!name) { alert("Name your persona."); return; }
  await api.savePersona({ id: state.editingPersonaId, name, description: $("#p-desc").value.trim() });
  $("#persona-modal").classList.add("hidden");
  await refreshPersonas();
}

// ---------- lorebook modal ----------
function openLoreModal(book) {
  state.editingLoreId = book ? book.id : null;
  state.loreDraft = book ? JSON.parse(JSON.stringify(book)) : { name: "", entries: [] };
  if (!state.loreDraft.entries) state.loreDraft.entries = [];
  $("#lore-modal-title").textContent = book ? `Edit ${book.name}` : "New Lorebook";
  $("#l-name").value = state.loreDraft.name || "";
  $("#lore-delete").classList.toggle("hidden", !book);
  renderLoreEntries();
  $("#lore-modal").classList.remove("hidden");
  $("#l-name").focus();
}
function renderLoreEntries() {
  const box = $("#lore-entries"); box.innerHTML = "";
  state.loreDraft.entries.forEach((e, idx) => {
    const row = document.createElement("div");
    row.className = "lore-entry";
    row.innerHTML = `
      <div class="lore-head">
        <input data-k="keys" placeholder="Trigger keywords, comma-separated (e.g. castle, keep)" value="${escapeHtml((e.keys || []).join(', '))}" />
        <button class="act del" title="Remove">🗑</button>
      </div>
      <textarea data-k="content" rows="2" placeholder="What the AI should know when a keyword appears…">${escapeHtml(e.content || "")}</textarea>
      <div class="opts">
        <label class="inline"><input type="checkbox" data-k="constant" ${e.constant ? "checked" : ""}/> Always on</label>
        <label class="inline">Position
          <select data-k="position">
            <option value="before" ${e.position !== "after" ? "selected" : ""}>Top (context)</option>
            <option value="after" ${e.position === "after" ? "selected" : ""}>Near end (stronger)</option>
          </select>
        </label>
        <label class="inline">Priority <input type="number" data-k="order" value="${escapeHtml(String(e.order != null ? e.order : 100))}" style="width:70px"/></label>
      </div>`;
    row.querySelector(".del").onclick = () => { state.loreDraft.entries.splice(idx, 1); renderLoreEntries(); };
    row.querySelectorAll("[data-k]").forEach((inp) => {
      inp.onchange = () => {
        const k = inp.dataset.k;
        if (k === "keys") e.keys = inp.value.split(",").map((s) => s.trim()).filter(Boolean);
        else if (k === "constant") e.constant = inp.checked;
        else if (k === "order") e.order = parseInt(inp.value) || 0;
        else e[k] = inp.value;
      };
    });
    box.appendChild(row);
  });
}
async function saveLoreFromModal() {
  const name = $("#l-name").value.trim();
  if (!name) { alert("Name your lorebook."); return; }
  state.loreDraft.name = name;
  await api.saveLorebook({ id: state.editingLoreId, ...state.loreDraft });
  $("#lore-modal").classList.add("hidden");
  await refreshLorebooks();
}

// ---------- settings drawer ----------
async function openSettings() {
  if (!state.story) return;
  const model = state.story.model || state.model || "";
  const preset = (await api.presets(model)).settings;
  const s = { ...preset, ...(state.story.settings || {}) };
  const modelSel = $("#s-model");
  modelSel.innerHTML = $("#model-select").innerHTML;
  modelSel.value = model;
  const perSel = $("#s-persona");
  perSel.innerHTML = `<option value="">None</option>` + state.personas.map((p) => `<option value="${p.id}">${escapeHtml(p.name)}</option>`).join("");
  perSel.value = state.story.persona_id || "";
  const set = (id, v) => { $(id).value = v; };
  set("#s-temp", s.temperature); $("#s-temp-val").textContent = (+s.temperature).toFixed(2);
  set("#s-ctx", s.num_ctx); $("#s-ctx-val").textContent = s.num_ctx;
  set("#s-pred", s.num_predict); $("#s-pred-val").textContent = s.num_predict;
  set("#s-minp", s.min_p); set("#s-topp", s.top_p); set("#s-topk", s.top_k);
  set("#s-reppen", s.repeat_penalty); set("#s-repn", s.repeat_last_n);
  $("#s-rag").checked = !!s.rag_enabled;
  $("#s-embed").value = s.embed_model || "nomic-embed-text";
  $("#settings-modal").classList.remove("hidden");
}
async function saveSettings() {
  const settings = {
    temperature: parseFloat($("#s-temp").value), num_ctx: parseInt($("#s-ctx").value),
    num_predict: parseInt($("#s-pred").value), min_p: parseFloat($("#s-minp").value),
    top_p: parseFloat($("#s-topp").value), top_k: parseInt($("#s-topk").value),
    repeat_penalty: parseFloat($("#s-reppen").value), repeat_last_n: parseInt($("#s-repn").value),
    rag_enabled: $("#s-rag").checked, embed_model: $("#s-embed").value.trim() || "nomic-embed-text",
  };
  const patch = { model: $("#s-model").value, persona_id: $("#s-persona").value || null, settings };
  state.story = await api.storyMeta(state.story.id, patch);
  $("#settings-modal").classList.add("hidden");
  updateHeader();
}

// ---------- models ----------
async function refreshModels() {
  const res = await api.models();
  $("#ollama-banner").classList.toggle("hidden", res.ok);
  const sel = $("#model-select");
  sel.innerHTML = "";
  if (!res.ok || !res.models.length) {
    sel.innerHTML = `<option value="">— no models —</option>`;
    state.model = null;
    $("#model-hint").textContent = res.ok ? "No models yet. Run: ollama pull mistral-nemo" : "Ollama offline.";
    return;
  }
  for (const m of res.models) { const o = document.createElement("option"); o.value = m; o.textContent = m; sel.appendChild(o); }
  const want = prefs.model && res.models.includes(prefs.model) ? prefs.model : res.models[0];
  sel.value = want; state.model = want;
  $("#model-hint").textContent = `${res.models.length} model(s) available.`;
}

// ---------- import ----------
function handleImportFile(file) {
  const reader = new FileReader();
  reader.onload = async () => {
    const b64 = String(reader.result).split(",").pop();
    try {
      const char = await api.importCharacter(file.name, b64);
      await refreshCharacters(); await refreshLorebooks();
      openCharModal(char);
    } catch (e) { alert("Import failed: " + e.message); }
  };
  reader.readAsDataURL(file);
}

// ---------- refreshers ----------
async function refreshModelsAll() { await refreshModels(); }
async function refreshCharacters() { state.characters = await api.characters(); renderCharacters(); updateHeader(); }
async function refreshStories() { state.stories = await api.stories(); renderStories(); }
async function refreshPersonas() { state.personas = await api.personas(); renderPersonas(); }
async function refreshLorebooks() { state.lorebooks = await api.lorebooks(); renderLorebooks(); }

// ---------- wiring ----------
function bindModal(id, closeIds) {
  closeIds.forEach((c) => $(c) && ($(c).onclick = () => $(id).classList.add("hidden")));
  $(id).addEventListener("click", (e) => { if (e.target.id === id.slice(1)) $(id).classList.add("hidden"); });
}

function init() {
  $("#model-select").addEventListener("change", (e) => { state.model = e.target.value; prefs.model = e.target.value; });
  $("#retry-ollama").addEventListener("click", (e) => { e.preventDefault(); refreshModels(); });

  // text-to-speech toggle
  try { state.tts = localStorage.getItem("rp_tts") === "1"; } catch { state.tts = false; }
  updateTtsBtn();
  if (!("speechSynthesis" in window)) $("#tts-btn").classList.add("hidden");
  $("#tts-btn").onclick = () => {
    state.tts = !state.tts;
    try { localStorage.setItem("rp_tts", state.tts ? "1" : "0"); } catch {}
    if (!state.tts && "speechSynthesis" in window) window.speechSynthesis.cancel();
    updateTtsBtn();
  };

  // story import / full backup
  $("#import-story-btn").onclick = () => $("#import-story-file").click();
  $("#import-story-file").addEventListener("change", (e) => {
    const f = e.target.files[0]; e.target.value = "";
    if (!f) return;
    const reader = new FileReader();
    reader.onload = async () => {
      const b64 = String(reader.result).split(",").pop();
      try { const s = await api.importStory(f.name, b64); await refreshStories(); await openStory(s.id); }
      catch (err) { alert("Story import failed: " + err.message); }
    };
    reader.readAsDataURL(f);
  });
  $("#backup-btn").onclick = () => window.open("/api/backup", "_blank");

  // greeting chooser
  $("#greet-cancel").onclick = () => $("#greet-modal").classList.add("hidden");
  bindModal("#greet-modal", []);

  // character modal
  $("#new-char-btn").onclick = () => openCharModal(null);
  $("#edit-active-char").onclick = () => { const c = currentCharacter(); if (c) openCharModal(c); };
  $("#char-cancel").onclick = closeCharModal;
  $("#char-save").onclick = saveCharFromModal;
  ["#c-description", "#c-personality", "#c-scenario"].forEach((s) => $(s).addEventListener("input", updatePermTokens));
  $("#c-avatar").addEventListener("input", renderAvatarPreview);
  $("#c-avatar-upload").onclick = () => {
    if (!state.editingCharId) { alert("Save the character first, then add an image."); return; }
    $("#avatar-file").click();
  };
  $("#avatar-file").addEventListener("change", (e) => {
    const f = e.target.files[0]; e.target.value = "";
    if (!f || !state.editingCharId) return;
    const reader = new FileReader();
    reader.onload = async () => {
      const b64 = String(reader.result).split(",").pop();
      const ext = (f.name.split(".").pop() || "png").toLowerCase();
      try {
        const updated = await api.setAvatar(state.editingCharId, b64, ext);
        state.editingCharImage = updated.avatar_image; renderAvatarPreview(); await refreshCharacters();
      } catch (err) { alert("Avatar upload failed: " + err.message); }
    };
    reader.readAsDataURL(f);
  });
  $("#c-avatar-clear").onclick = async () => {
    if (state.editingCharId && state.editingCharImage) { try { await api.clearAvatarImg(state.editingCharId); } catch {} }
    state.editingCharImage = null; renderAvatarPreview(); await refreshCharacters();
  };
  $("#char-delete").onclick = async () => {
    if (!state.editingCharId || !confirm("Delete this character? Saved stories keep working.")) return;
    await api.deleteCharacter(state.editingCharId); closeCharModal(); await refreshCharacters();
  };
  $("#char-export").onclick = () => { if (state.editingCharId) window.open(`/api/characters/${state.editingCharId}/export`, "_blank"); };
  bindModal("#char-modal", []);

  // import
  $("#import-char-btn").onclick = () => $("#import-file").click();
  $("#import-file").addEventListener("change", (e) => { if (e.target.files[0]) handleImportFile(e.target.files[0]); e.target.value = ""; });

  // persona modal
  $("#new-persona-btn").onclick = () => openPersonaModal(null);
  $("#persona-cancel").onclick = () => $("#persona-modal").classList.add("hidden");
  $("#persona-save").onclick = savePersonaFromModal;
  $("#persona-delete").onclick = async () => {
    if (!state.editingPersonaId || !confirm("Delete this persona?")) return;
    await api.deletePersona(state.editingPersonaId); $("#persona-modal").classList.add("hidden"); await refreshPersonas();
  };
  bindModal("#persona-modal", []);

  // lorebook modal
  $("#new-lore-btn").onclick = () => openLoreModal(null);
  $("#lore-add-entry").onclick = () => { state.loreDraft.entries.push({ keys: [], content: "", order: 100, position: "before" }); renderLoreEntries(); };
  $("#lore-cancel").onclick = () => $("#lore-modal").classList.add("hidden");
  $("#lore-save").onclick = saveLoreFromModal;
  $("#lore-delete").onclick = async () => {
    if (!state.editingLoreId || !confirm("Delete this lorebook?")) return;
    await api.deleteLorebook(state.editingLoreId); $("#lore-modal").classList.add("hidden"); await refreshLorebooks();
  };
  bindModal("#lore-modal", []);

  // settings drawer
  $("#settings-btn").onclick = openSettings;
  $("#settings-cancel").onclick = () => $("#settings-modal").classList.add("hidden");
  $("#settings-save").onclick = saveSettings;
  $("#s-temp").addEventListener("input", (e) => $("#s-temp-val").textContent = (+e.target.value).toFixed(2));
  $("#s-ctx").addEventListener("input", (e) => $("#s-ctx-val").textContent = e.target.value);
  $("#s-pred").addEventListener("input", (e) => $("#s-pred-val").textContent = e.target.value);
  $("#s-reset-preset").onclick = async () => {
    const p = (await api.presets($("#s-model").value)).settings;
    $("#s-temp").value = p.temperature; $("#s-temp-val").textContent = (+p.temperature).toFixed(2);
    $("#s-ctx").value = p.num_ctx; $("#s-ctx-val").textContent = p.num_ctx;
    $("#s-pred").value = p.num_predict; $("#s-pred-val").textContent = p.num_predict;
    $("#s-minp").value = p.min_p; $("#s-topp").value = p.top_p; $("#s-topk").value = p.top_k;
    $("#s-reppen").value = p.repeat_penalty; $("#s-repn").value = p.repeat_last_n;
  };

  // author's note (save on blur)
  $("#an-input").addEventListener("change", async () => {
    if (!state.story) return;
    const an = state.story.authors_note || { depth: 4, role: "system" };
    an.text = $("#an-input").value;
    state.story = await api.storyMeta(state.story.id, { authors_note: an });
  });

  // composer
  const input = $("#input");
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = ""; input.style.height = "auto";
    chatAction("send", text);
  });
  $("#stop-btn").onclick = () => { if (state.abort) state.abort.abort(); };

  // initial load
  (async () => {
    await refreshModels();
    await Promise.all([refreshCharacters(), refreshStories(), refreshPersonas(), refreshLorebooks()]);
  })();
}

init();
