"""AI Roleplayer — a local roleplay/story bot that runs entirely on your machine.

The AI "brain" is a model served locally by Ollama (http://localhost:11434).
This FastAPI app serves the web UI and proxies streaming chat to Ollama, plus
stores your characters and saved stories as JSON files under ./data/.

Nothing is sent anywhere except your own local Ollama instance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import storage

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI(title="AI Roleplayer")

# Copy bundled example characters into data/ on first launch.
storage.seed_presets()


# --- schemas ------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    system: Optional[str] = None
    temperature: float = 0.8


class Character(BaseModel):
    id: Optional[str] = None
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    greeting: str = ""
    avatar: str = "🎭"


class Story(BaseModel):
    id: Optional[str] = None
    title: str = "Untitled"
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    model: Optional[str] = None
    messages: List[ChatMessage] = []


# --- helpers ------------------------------------------------------------------

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# --- Ollama -------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    """Return the models the user has pulled in Ollama, or a helpful error."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
        names = [m["name"] for m in data.get("models", [])]
        return {"ok": True, "models": names}
    except Exception as e:  # Ollama not running / unreachable
        return {"ok": False, "models": [], "error": str(e)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream a chat completion from Ollama back to the browser as SSE."""
    messages: List[dict] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.extend(m.model_dump() for m in req.messages)

    payload = {
        "model": req.model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": req.temperature},
    }

    async def generate():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        detail = body.decode("utf-8", errors="ignore")
                        yield _sse({"error": f"Ollama returned {r.status_code}: {detail}"})
                        return
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = obj.get("message", {}).get("content")
                        if content:
                            yield _sse({"token": content})
                        if obj.get("done"):
                            yield _sse({"done": True})
                            return
        except Exception as e:
            yield _sse({"error": f"Could not reach Ollama at {OLLAMA_URL}. Is it running? ({e})"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- characters ---------------------------------------------------------------

@app.get("/api/characters")
async def get_characters():
    return storage.list_characters()


@app.get("/api/characters/{cid}")
async def get_character(cid: str):
    char = storage.get_character(cid)
    if not char:
        raise HTTPException(404, "Character not found")
    return char


@app.post("/api/characters")
async def create_character(char: Character):
    return storage.save_character(char.model_dump())


@app.put("/api/characters/{cid}")
async def update_character(cid: str, char: Character):
    data = char.model_dump()
    data["id"] = cid
    return storage.save_character(data)


@app.delete("/api/characters/{cid}")
async def remove_character(cid: str):
    if not storage.delete_character(cid):
        raise HTTPException(404, "Character not found")
    return {"ok": True}


# --- stories ------------------------------------------------------------------

@app.get("/api/stories")
async def get_stories():
    return storage.list_stories()


@app.get("/api/stories/{sid}")
async def get_story(sid: str):
    story = storage.get_story(sid)
    if not story:
        raise HTTPException(404, "Story not found")
    return story


@app.post("/api/stories")
async def create_story(story: Story):
    return storage.save_story(story.model_dump())


@app.put("/api/stories/{sid}")
async def update_story(sid: str, story: Story):
    data = story.model_dump()
    data["id"] = sid
    return storage.save_story(data)


@app.delete("/api/stories/{sid}")
async def remove_story(sid: str):
    if not storage.delete_story(sid):
        raise HTTPException(404, "Story not found")
    return {"ok": True}


# --- static UI ----------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
