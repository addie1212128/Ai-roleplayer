# 🎭 AI Roleplayer

A roleplay / story bot that runs **entirely on your own computer**. The AI
"brain" is a local model served by [Ollama](https://ollama.com) — nothing you
type ever leaves your machine. You play through characters in a web page, with a
real prompt engine behind it: lorebooks, long-term memory, personas, swipes, and
Tavern character-card import.

![local](https://img.shields.io/badge/runs-100%25%20locally-8b6cff) ![no api key](https://img.shields.io/badge/no%20API%20key-needed-6c8bff)

---

## What it does

- 🎭 **Characters** — name, personality, scenario, greeting, plus advanced
  fields (example dialogue, system-prompt override, post-history instructions,
  narrative voice, alternate greetings). Ships with three examples; create or
  import your own. Give them **avatar images** (or emoji).
- 👥 **Group scenes** — put two or more characters in one story. A character
  replies automatically on your turn (mention a name to direct it), or tap a
  character to make them speak. Each reply is labelled with who's talking.
- 🧠 **Real memory** — a keyword **lorebook** (world facts that surface only when
  relevant), **rolling summarization** of older events, a floating **author's
  note**, and optional **semantic recall (RAG)** via local embeddings keep long
  stories coherent instead of goldfish-ing.
- 💬 **Immersive streaming** — in-character replies that type out live, actions
  in *italics*, with a scope-controlled prompt that stops the model speaking for
  you or breaking character. Optional **text-to-speech** reads replies aloud.
- 🔄 **Full control of each reply** — **swipe** for alternates, **regenerate**,
  **continue**, **edit** any message, or **delete from** a point.
- 🪄 **AI assists** — **impersonate** (let the model draft *your* next line when
  you're stuck), **✨ generate a character** from a one-line concept, a
  **light/dark theme**, and **story search**.
- 🧑 **Personas** — tell the AI who *you* are for better, more personal replies.
- 📥 **Import community characters** — load Tavern / SillyTavern **V2 & V3 cards**
  (`.json` or `.png`), so thousands of existing characters just work. Export too.
- 💾 **Export & backup** — download any story (JSON to re-import, or a readable
  **Markdown transcript** to keep/share), or grab a one-click zip of *all* your
  data so a lost machine never loses your saves.
- 🎚️ **Per-model tuned settings** — sensible sampler presets per model, a
  creativity slider, context size, reply length, and an advanced panel.
- 📊 **Live context meter** — see exactly how much of the model's memory you're
  using, so you know when to start a fresh scene.

Everything is stored as plain JSON under `data/`, fully local and private.

---

## Setup (one time)

### 1. Install Ollama — the AI engine
Download from **<https://ollama.com/download>** and install. It runs in the
background and uses your GPU automatically.

### 2. Pull a model
In a terminal, grab a model. Picks for a **GPU gaming PC**, by VRAM:

| Your GPU VRAM | Recommended | Command |
|---|---|---|
| 8 GB | Dolphin-Mistral 7B | `ollama pull dolphin-mistral` |
| **12 GB** | **Mistral-Nemo 12B** *(great default)* | `ollama pull mistral-nemo` |
| 12–16 GB | Mag-Mell 12B *(RP-tuned)* | `ollama run hf.co/mradermacher/MN-12B-Mag-Mell-R1-GGUF:Q4_K_M` |
| 16 GB | Gemma 3 12B | `ollama pull gemma3:12b-it-qat` |
| 24 GB | Cydonia 24B *(companion-grade)* | `ollama pull hf.co/bartowski/TheDrummer_Cydonia-24B-v4.3-GGUF:Q5_K_M` |

> Any chat model works — use whatever you like. Bigger = more coherent, needs
> more VRAM. The app auto-applies a good sampler preset per model.

> 🧠 **Want your *own* model?** You can fine-tune an open base model into a
> custom roleplay (and optionally teaching) brain that plugs right back into
> this app via Ollama. See **[TRAINING.md](TRAINING.md)** for the full,
> honest build plan — base models, datasets, cost, and the GGUF export steps.

### 3. Install Python
If you don't have it: **<https://www.python.org/downloads/>** (3.10+). On
Windows, tick **"Add Python to PATH"** during install.

---

## Run it

**Windows:** double-click **`start.bat`**
**macOS / Linux:** `./start.sh`
**Manual (any OS):** `pip install -r requirements.txt && python run.py`

Your browser opens at **<http://localhost:8000>**. Pick a character and play.

> **Playing from another device (phone, couch)?** By default the app binds to
> `127.0.0.1` (this machine only). If you set `HOST=0.0.0.0` to reach it over
> your Wi-Fi, the API would otherwise be open to everyone on the network with
> **no login** — so `run.py` automatically generates a one-time access **token**
> and prints a URL with it. Open that exact URL; keep the token private. (If you
> launch `uvicorn` directly instead of `run.py`, set `RP_TOKEN=<something>`
> yourself before binding beyond loopback.)

---

## How to play

1. **Pick a character** on the left to start a story (**✎** edits, **+** creates,
   **⬇** imports a card).
2. **Type an action or dialogue** and press **Enter**. Write in-character, e.g.
   *"I sit by the fire and order an ale."*
3. Hover any reply for controls: **‹ 1/2 ›** swipe to a different version (the
   **›** at the end generates a new one), **↻** regenerate, **↳** continue,
   **✎** edit, **🗑** delete from here.
4. **⚙ Settings** (per story): model, persona, creativity, context size, reply
   length, and advanced samplers.
5. The **author's note** bar above the composer is a private nudge to steer tone
   or what happens next — it's injected near the end of the prompt for max effect.

### Tips for a smart, immersive bot
- **The context size matters most.** The app defaults to 16K; raise it (Settings)
  for longer memory if your GPU has the VRAM, lower it if replies are slow.
- **Write a rich greeting.** The model copies its length, tone, and formatting —
  a short dry greeting yields short dry replies.
- **Add example dialogue** (advanced character field) to lock in a voice.
- **Use a lorebook** for world facts: they only enter the prompt when a keyword
  appears, so the bot "knows" a huge world without wasting memory every turn.
- **Group scenes** (👥 in the sidebar) shine with 2–3 characters; mention a name
  to pick who answers, or tap a character to make them speak.
- **Long-term memory (RAG)** is optional — pull an embedding model first
  (`ollama pull nomic-embed-text`), then enable it in a story's Settings. It
  helps most in very long stories; leave it off if you're unsure.

---

## How it works

```
Browser UI  ⇄  Python/FastAPI (localhost:8000)  ⇄  Ollama (localhost:11434)  ⇄  your GPU
```

The server assembles the prompt from typed components in a deliberate order
(system + character + persona → lorebook → summaries → budgeted history → author's
note & reminders near the end), streams the reply from Ollama, and owns story
persistence. See the exact prompt for any story at
`GET /api/debug/prompt?story_id=<id>`.

### Where your data lives
- `data/characters/`, `data/stories/`, `data/lorebooks/`, `data/personas/` — all
  local JSON, git-ignored.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Red **"Can't reach Ollama"** banner | Install Ollama and make sure it's running (`ollama serve`). Click **retry**. |
| Dropdown says **"no models"** | Pull one: `ollama pull mistral-nemo`, then retry. |
| Replies are slow | Lower the context size (Settings), use a smaller model, or confirm Ollama is using your GPU. |
| Bot forgets / loses the plot | Raise context size, and lean on a **lorebook** + let summaries build. |
| Replies repeat themselves | In Settings, nudge `repeat_penalty` to ~1.08 (but keep it ≤1.1; some 24B models want 1.0). |
| Model "thinks" out loud in replies | Reasoning blocks (`<think>…</think>`) are stripped automatically. |
| `python` not found (Windows) | Reinstall Python with **"Add to PATH"**, or use `py`. |
| Port 8000 in use | `PORT=8080 python run.py`. |

> ⚠️ Imported community cards can contain instructions in their text that try to
> steer the model. Skim a card's description before playing — it's just text, not
> code, but it does shape behavior.

---

## Developing / testing

```bash
pip install -r requirements.txt
python tests/selftest.py      # backend logic + fake-Ollama integration (no real Ollama needed)
python tests/browser_test.py  # headless-browser UI test (needs: pip install playwright)
```

## Tech
- **Backend:** Python, FastAPI, httpx · **Frontend:** vanilla HTML/CSS/JS (no build)
- **Model runtime:** Ollama (local) · **Card format:** Tavern Character Card V2/V3
