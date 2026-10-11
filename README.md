# 🎭 AI Roleplayer

A roleplay / story bot that runs **entirely on your own computer**. The AI
"brain" is a local model served by [Ollama](https://ollama.com) — nothing you
type is sent to the cloud. You chat with characters in a web page in your
browser, and every story is saved to disk so you can pick it back up later.

![local](https://img.shields.io/badge/runs-100%25%20locally-8b6cff) ![no api key](https://img.shields.io/badge/no%20API%20key-needed-6c8bff)

---

## What you get

- 🎭 **Characters** — give the AI a name, personality, scenario, and opening
  line. Ships with three examples (a tavern keeper, a starship captain, and an
  open-world Narrator/game master). Create and edit your own.
- 💬 **Immersive streaming chat** — in-character replies that type out live,
  with actions shown in *italics*.
- 💾 **Saved stories** — every roleplay is stored locally; resume or delete any
  time.
- 🧠 **Use any model** — pick from whatever models you've pulled into Ollama.
- 🎚️ **Controls** — creativity (temperature), your player name, per-character
  settings.

---

## Setup (one time)

### 1. Install Ollama — this is the AI engine
Download it from **<https://ollama.com/download>** and install. It runs quietly
in the background and uses your GPU automatically.

### 2. Pull a model
Open a terminal / command prompt and grab a model. For a gaming PC with a
decent GPU, good starting points:

```bash
ollama pull llama3.1        # 8B — fast, solid all-rounder (~5 GB)
# or, for richer writing if you have the VRAM:
ollama pull qwen2.5:14b     # larger, more detailed prose (~9 GB)
```

> Any chat model works — use whatever you like. Smaller = faster, bigger =
> more coherent storytelling.

### 3. Install Python
If you don't have it: **<https://www.python.org/downloads/>** (3.10 or newer).
On Windows, tick **"Add Python to PATH"** during install.

---

## Run it

**Windows:** double-click **`start.bat`**

**macOS / Linux:**
```bash
./start.sh
```

**Or manually (any OS):**
```bash
pip install -r requirements.txt
python run.py
```

Your browser opens at **<http://localhost:8000>**. That's it — pick a character
on the left and start playing.

> The first launch installs a few small Python packages and may take a minute.
> Later launches are instant.

---

## How to use it

1. **Pick a character** in the left sidebar to start a new story — or click
   **✎** to edit one, or **+** to create your own.
2. **Type your action or dialogue** and press **Enter** (Shift+Enter for a new
   line). Write in-character — e.g. *"I sit down and order an ale."*
3. The character responds and the story is **saved automatically**. Find past
   stories under **Your Stories**.
4. Tweak the **model**, **creativity**, and **your name** at the bottom of the
   sidebar.

### Making a good character
- **Description** — who they are in a sentence.
- **Personality** — how they talk and behave (this shapes the voice most).
- **Scenario** — the setting and the situation you're dropped into.
- **Greeting** — their first message; sets the scene. Put *actions* in asterisks.

---

## Where your data lives

Everything is on your machine:

- `data/characters/` — your characters (JSON files you can edit by hand).
- `data/stories/` — your saved roleplays.

The `data/` folder is git-ignored, so your stories stay private and local.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Red **"Can't reach Ollama"** banner | Make sure Ollama is installed and running. It usually auto-starts; otherwise run `ollama serve`. Click **retry**. |
| Dropdown says **"no models"** | You haven't pulled one yet: `ollama pull llama3.1`, then click retry. |
| Replies are slow | Use a smaller model (e.g. `llama3.1` instead of a 14B+), or make sure Ollama is using your GPU. |
| `python` not found (Windows) | Reinstall Python with **"Add to PATH"** checked, or use `py` instead of `python`. |
| Port 8000 in use | Run with a different port: `PORT=8080 python run.py`. |

---

## How it works

```
Browser (web UI)  ⇄  Python / FastAPI app (localhost:8000)  ⇄  Ollama (localhost:11434)  ⇄  local model on your GPU
```

The Python app serves the UI, stores characters & stories, and streams model
output from Ollama to your browser. No external services are involved.

---

## Tech

- **Backend:** Python, FastAPI, httpx
- **Frontend:** vanilla HTML/CSS/JS (no build step)
- **Model runtime:** Ollama (local)
