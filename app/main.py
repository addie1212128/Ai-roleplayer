"""AI Roleplayer — a local roleplay/story bot that runs entirely on your machine.

The AI "brain" is a model served locally by Ollama (http://localhost:11434).
This FastAPI app serves the web UI, assembles the prompt SERVER-SIDE (character
card + persona + lorebook + summaries + budgeted history + depth injections),
streams the reply from Ollama, and OWNS story persistence (atomic, per-story
locked) so client and server never clobber each other.

Nothing is sent anywhere except your own local Ollama instance.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from . import cards, context, rag, storage
from .samplers import DEFAULTS, build_options, preset_for

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI(title="AI Roleplayer")
storage.seed_presets()

# One lock per story serializes read-modify-write (incl. the whole generation),
# so two turns on the same story can't interleave. Different stories run free.
_story_locks: dict[str, asyncio.Lock] = {}


def _lock(sid: str) -> asyncio.Lock:
    lock = _story_locks.get(sid)
    if lock is None:
        lock = asyncio.Lock()
        _story_locks[sid] = lock
    return lock


# --- schemas (extra='allow' so forward-compatible fields survive) -------------

class Character(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: str
    avatar: str = "🎭"
    description: str = ""
    personality: str = ""
    scenario: str = ""
    greeting: str = ""
    example_dialogue: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    lorebook_id: Optional[str] = None
    pov: str = ""
    alternate_greetings: Optional[List[str]] = None
    avatar_image: Optional[str] = None


class StoryCreate(BaseModel):
    model_config = ConfigDict(extra="allow")
    title: str = "Untitled"
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    model: Optional[str] = None
    persona_id: Optional[str] = None
    messages: List[dict] = []
    settings: dict = {}


class StoryMeta(BaseModel):
    model_config = ConfigDict(extra="allow")
    title: Optional[str] = None
    model: Optional[str] = None
    persona_id: Optional[str] = None
    lorebook_id: Optional[str] = None
    authors_note: Optional[dict] = None
    settings: Optional[dict] = None


class ChatReq(BaseModel):
    model_config = ConfigDict(extra="allow")
    story_id: str
    action: str = "send"           # send | regenerate | swipe | continue
    user_message: Optional[str] = None
    settings: dict = {}


class MessageEdit(BaseModel):
    content: str
    swipe_id: Optional[int] = None


class SwipeSelect(BaseModel):
    swipe_id: int


class Lorebook(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: str = "Lorebook"
    scanDepth: int = 4
    tokenBudget: Optional[int] = None
    entries: List[dict] = []


class Persona(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: str
    description: str = ""


class ImportReq(BaseModel):
    filename: str = "card"
    data_b64: str


class AvatarReq(BaseModel):
    data_b64: str
    ext: str = "png"


_IMG_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_AVATAR_BYTES = 8_000_000


def _save_avatar(cid: str, raw: bytes, ext: str) -> str:
    """Write a character's avatar image; returns the servable path."""
    storage.AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    for p in storage.AVATAR_DIR.glob(f"{cid}.*"):  # drop any previous ext
        try:
            p.unlink()
        except OSError:
            pass
    (storage.AVATAR_DIR / f"{cid}.{ext}").write_bytes(raw)
    return f"/avatars/{cid}.{ext}"


# --- helpers ------------------------------------------------------------------

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def effective_settings(story: dict, override: dict) -> dict:
    """preset(model)  <-  story.settings  <-  per-call override."""
    model = (override or {}).get("model") or story.get("model") or ""
    s = preset_for(model)
    s.update(story.get("settings") or {})
    s.update(override or {})
    s["model"] = model
    return s


def _resolve_actors(story: dict):
    character = storage.get_character(story.get("character_id")) or {
        "name": story.get("character_name") or "Narrator"}
    persona = storage.get_persona(story["persona_id"]) if story.get("persona_id") else None
    lb_id = character.get("lorebook_id") or story.get("lorebook_id")
    lorebook = storage.get_lorebook(lb_id) if lb_id else None
    return character, persona, lorebook


async def _summarize_block(text: str, model: str) -> Optional[str]:
    payload = {
        "model": model, "stream": False, "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": context.SUMMARIZE_PROMPT},
            {"role": "user", "content": text},
        ],
        "options": {"temperature": 0.0, "num_predict": 256, "num_ctx": 8192},
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            out = r.json().get("message", {}).get("content", "")
        return context.strip_think(out).strip() or None
    except Exception:
        return None


async def _maybe_summarize(story: dict, model: str) -> bool:
    """Summarize older message ranges once enough raw turns have accumulated."""
    msgs = story.get("messages", [])
    summaries = story.setdefault("summaries", [])
    last = max((s["range"][1] for s in summaries if s.get("range")), default=0)
    KEEP_RAW, TRIGGER = 8, 24
    if len(msgs) - last < TRIGGER:
        return False
    end = len(msgs) - KEEP_RAW
    block = msgs[last:end]
    if len(block) < 12:
        return False
    lines = []
    for m in block:
        who = "User" if m.get("role") == "user" else (story.get("character_name") or "Character")
        lines.append(f"{who}: {m.get('content', '')}")
    summary = await _summarize_block("\n".join(lines), model)
    if summary:
        summaries.append({"range": [last, end], "text": summary})
        storage.save_story(story)
        return True
    return False


def _invalidate_summaries(story: dict, from_index: int) -> None:
    story["summaries"] = [s for s in story.get("summaries", [])
                          if s.get("range", [0, 0])[1] <= from_index]


# --- Ollama / models ----------------------------------------------------------

@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
        return {"ok": True, "models": [m["name"] for m in data.get("models", [])]}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


@app.get("/api/presets")
async def presets(model: str = ""):
    return {"defaults": DEFAULTS, "model": model, "settings": preset_for(model)}


# --- chat ---------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatReq):
    if not storage.get_story(req.story_id):
        raise HTTPException(404, "Story not found")

    async def generate():
        async with _lock(req.story_id):
            story = storage.get_story(req.story_id)
            if story is None:
                yield _sse({"error": "Story vanished."})
                return
            msgs = story.setdefault("messages", [])
            character, persona, lorebook = _resolve_actors(story)
            settings = effective_settings(story, req.settings)
            model = settings.get("model")
            if not model:
                yield _sse({"error": "No model selected. Pull one in Ollama (e.g. "
                                     "`ollama pull mistral-nemo`) and pick it."})
                return

            action = req.action
            # Choose the context to send WITHOUT persisting any destructive
            # mutation first. Nothing is written to disk until a clean reply
            # exists, so a failed, empty, or aborted generation never loses the
            # previous reply (regenerate) or leaves an orphaned user turn (send).
            if action == "send":
                if not (req.user_message or "").strip():
                    yield _sse({"error": "Empty message."})
                    return
                msgs.append({"role": "user", "content": req.user_message.strip()})
                ctx_msgs = msgs
            elif action == "regenerate":
                # Replace a trailing assistant reply, or generate one for a
                # dangling user turn. The pop happens only on success, below.
                ctx_msgs = msgs[:-1] if (msgs and msgs[-1].get("role") == "assistant") else msgs
            elif action == "swipe":
                if not (msgs and msgs[-1].get("role") == "assistant"):
                    yield _sse({"error": "Nothing to swipe yet."})
                    return
                ctx_msgs = msgs[:-1]       # identical context to the original reply
            elif action == "continue":
                if not (msgs and msgs[-1].get("role") == "assistant"):
                    yield _sse({"error": "Nothing to continue yet."})
                    return
                ctx_msgs = msgs
            else:
                yield _sse({"error": f"Unknown action: {action}"})
                return

            ctx_story = dict(story)
            ctx_story["messages"] = ctx_msgs

            # Optional semantic recall (degrades to "" if the embed model is absent).
            rag_block = ""
            if settings.get("rag_enabled"):
                embed_model = settings.get("embed_model") or rag.DEFAULT_EMBED_MODEL
                try:
                    await rag.update_store(ctx_story, embed_model, OLLAMA_URL)
                    last_user = next((m.get("content", "") for m in reversed(ctx_msgs)
                                      if m.get("role") == "user"), "")
                    rag_block = await rag.retrieve(ctx_story, last_user, embed_model, OLLAMA_URL)
                except Exception:
                    rag_block = ""

            built = context.assemble_prompt(ctx_story, character, persona, lorebook,
                                            settings, rag_block=rag_block)
            messages = built["messages"]
            if action == "continue":
                messages.append({"role": "system", "content": (
                    f"[Continue {character.get('name', 'the character')}'s previous reply "
                    f"from exactly where it left off. Do not repeat earlier text.]")})

            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "keep_alive": settings.get("keep_alive", "30m"),
                "options": build_options(settings),
            }

            full = ""
            prompt_tokens = eval_tokens = None
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                        if r.status_code != 200:
                            body = await r.aread()
                            yield _sse({"error": f"Ollama returned {r.status_code}: "
                                                 f"{body.decode('utf-8', 'ignore')}"})
                            return
                        async for line in r.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            tok = obj.get("message", {}).get("content")
                            if tok:
                                full += tok
                                yield _sse({"token": tok})
                            if obj.get("done"):
                                prompt_tokens = obj.get("prompt_eval_count")
                                eval_tokens = obj.get("eval_count")
            except Exception as e:
                yield _sse({"error": f"Could not reach Ollama at {OLLAMA_URL}. "
                                     f"Is it running? ({e})"})
                return

            clean = context.strip_think(full)
            if not clean.strip():
                yield _sse({"error": "The model returned an empty reply. Try again, "
                                     "lower the temperature, or pick another model."})
                return

            # Commit the result to the right place (first write to disk).
            if action == "send":
                msgs.append({"role": "assistant", "content": clean,
                             "swipes": [clean], "swipe_id": 0})
                idx = len(msgs) - 1
            elif action == "regenerate":
                if msgs and msgs[-1].get("role") == "assistant":
                    msgs.pop()               # drop old reply only now that we have a new one
                msgs.append({"role": "assistant", "content": clean,
                             "swipes": [clean], "swipe_id": 0})
                idx = len(msgs) - 1
            elif action == "swipe":
                last = msgs[-1]
                last.setdefault("swipes", [last.get("content", "")])
                last["swipes"].append(clean)
                last["swipe_id"] = len(last["swipes"]) - 1
                last["content"] = clean
                idx = len(msgs) - 1
            else:  # continue
                last = msgs[-1]
                sep = "" if last.get("content", "").endswith((" ", "\n")) else " "
                last["content"] = last.get("content", "") + sep + clean
                if last.get("swipes"):
                    last["swipes"][last.get("swipe_id", 0)] = last["content"]
                idx = len(msgs) - 1

            storage.save_story(story)
            summarized = False
            if action in ("send", "regenerate"):
                summarized = await _maybe_summarize(story, model)

            yield _sse({
                "done": True,
                "index": idx,
                "message": msgs[idx],
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens,
                "num_ctx": settings.get("num_ctx"),
                "summarized": summarized,
                "debug": built["debug"],
            })

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/debug/prompt")
async def debug_prompt(story_id: str):
    story = storage.get_story(story_id)
    if not story:
        raise HTTPException(404, "Story not found")
    character, persona, lorebook = _resolve_actors(story)
    settings = effective_settings(story, {})
    return context.assemble_prompt(story, character, persona, lorebook, settings)


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


@app.post("/api/characters/import")
async def import_character(req: ImportReq):
    try:
        raw = base64.b64decode(req.data_b64)
    except Exception:
        raise HTTPException(400, "Could not decode the uploaded file.")
    try:
        parsed = cards.parse_card_bytes(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    char = parsed["character"]
    book = parsed.get("character_book")
    if book:
        lb = cards.lorebook_from_character_book(book, f"{char['name']} — lorebook")
        if lb and lb.get("entries"):
            saved_book = storage.save_lorebook(lb)
            char["lorebook_id"] = saved_book["id"]
    saved = storage.save_character(char)
    # The card PNG itself is the character art — use it as the avatar.
    if raw[:8] == cards.PNG_SIG:
        saved["avatar_image"] = _save_avatar(saved["id"], raw, "png")
        saved = storage.save_character(saved)
    return saved


@app.post("/api/characters/{cid}/avatar")
async def set_avatar(cid: str, req: AvatarReq):
    char = storage.get_character(cid)
    if not char:
        raise HTTPException(404, "Character not found")
    ext = (req.ext or "png").lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in _IMG_EXTS:
        raise HTTPException(400, "Unsupported image type")
    try:
        raw = base64.b64decode(req.data_b64)
    except Exception:
        raise HTTPException(400, "Could not decode the image.")
    if len(raw) > MAX_AVATAR_BYTES:
        raise HTTPException(400, "Image too large (max 8 MB).")
    char["avatar_image"] = _save_avatar(cid, raw, ext)
    return storage.save_character(char)


@app.delete("/api/characters/{cid}/avatar")
async def clear_avatar(cid: str):
    char = storage.get_character(cid)
    if not char:
        raise HTTPException(404, "Character not found")
    for p in storage.AVATAR_DIR.glob(f"{cid}.*"):
        try:
            p.unlink()
        except OSError:
            pass
    char["avatar_image"] = None
    return storage.save_character(char)


@app.get("/api/characters/{cid}/export")
async def export_character(cid: str):
    char = storage.get_character(cid)
    if not char:
        raise HTTPException(404, "Character not found")
    card = cards.card_from_character(char)
    fname = (char.get("name") or "character").replace(" ", "_")
    # HTTP headers are Latin-1; a non-ASCII or emoji name (common on imported
    # cards) would raise. Send an ASCII fallback + an RFC 5987 UTF-8 filename*.
    ascii_name = "".join(c for c in fname if c.isascii() and c not in '"\\') or "character"
    return JSONResponse(card, headers={
        "Content-Disposition":
            f"attachment; filename=\"{ascii_name}.json\"; filename*=UTF-8''{quote(fname)}.json"})


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


@app.get("/api/stories/{sid}/export")
async def export_story(sid: str):
    story = storage.get_story(sid)
    if not story:
        raise HTTPException(404, "Story not found")
    fname = (story.get("title") or "story").replace(" ", "_")
    ascii_name = "".join(c for c in fname if c.isascii() and c not in '"\\') or "story"
    return JSONResponse(story, headers={
        "Content-Disposition":
            f"attachment; filename=\"{ascii_name}.json\"; filename*=UTF-8''{quote(fname)}.json"})


@app.post("/api/stories/import")
async def import_story(req: ImportReq):
    try:
        obj = json.loads(base64.b64decode(req.data_b64).decode("utf-8", "ignore"))
    except Exception:
        raise HTTPException(400, "Could not read the story file.")
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        raise HTTPException(400, "Not a valid story export.")
    obj.pop("id", None)  # assign a fresh id so imports never collide
    obj.setdefault("summaries", [])
    obj.setdefault("authors_note", {"text": "", "depth": 4, "role": "system"})
    obj["title"] = str(obj.get("title") or "Imported story")
    return storage.save_story(obj)


@app.get("/api/backup")
async def backup():
    """Download every piece of local data (characters, stories, lorebooks,
    personas, avatars) as a single zip, so a lost container never loses saves."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d in (storage.CHAR_DIR, storage.STORY_DIR, storage.LORE_DIR,
                  storage.PERSONA_DIR, storage.AVATAR_DIR):
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.is_file():
                    z.write(f, arcname=str(f.relative_to(storage.DATA)))
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="ai-roleplayer-backup.zip"'})


@app.post("/api/stories")
async def create_story(story: StoryCreate):
    data = story.model_dump()
    # Normalize the seeded greeting into the swipe-aware message shape.
    for m in data.get("messages", []):
        if m.get("role") == "assistant" and "swipes" not in m:
            m["swipes"] = [m.get("content", "")]
            m["swipe_id"] = 0
    data.setdefault("summaries", [])
    data.setdefault("authors_note", {"text": "", "depth": 4, "role": "system"})
    return storage.save_story(data)


@app.delete("/api/stories/{sid}")
async def remove_story(sid: str):
    if not storage.delete_story(sid):
        raise HTTPException(404, "Story not found")
    _story_locks.pop(sid, None)  # don't leak the per-story lock
    return {"ok": True}


@app.put("/api/stories/{sid}/meta")
async def update_story_meta(sid: str, meta: StoryMeta):
    async with _lock(sid):
        story = storage.get_story(sid)
        if not story:
            raise HTTPException(404, "Story not found")
        # exclude_unset so a client can explicitly clear persona_id/lorebook_id
        # (sent as null) while omitted fields stay untouched.
        patch = meta.model_dump(exclude_unset=True)
        story.update(patch)
        return storage.save_story(story)


@app.put("/api/stories/{sid}/message/{idx}")
async def edit_message(sid: str, idx: int, edit: MessageEdit):
    async with _lock(sid):
        story = storage.get_story(sid)
        if not story:
            raise HTTPException(404, "Story not found")
        msgs = story.get("messages", [])
        if idx < 0 or idx >= len(msgs):
            raise HTTPException(404, "Message index out of range")
        m = msgs[idx]
        m["content"] = edit.content
        if m.get("swipes"):
            sid_i = edit.swipe_id if edit.swipe_id is not None else m.get("swipe_id", 0)
            if 0 <= sid_i < len(m["swipes"]):
                m["swipes"][sid_i] = edit.content
                m["swipe_id"] = sid_i
        _invalidate_summaries(story, idx)
        storage.save_story(story)
        return m


@app.put("/api/stories/{sid}/message/{idx}/swipe")
async def select_swipe(sid: str, idx: int, sel: SwipeSelect):
    async with _lock(sid):
        story = storage.get_story(sid)
        if not story:
            raise HTTPException(404, "Story not found")
        msgs = story.get("messages", [])
        if idx < 0 or idx >= len(msgs):
            raise HTTPException(404, "Message index out of range")
        m = msgs[idx]
        swipes = m.get("swipes") or [m.get("content", "")]
        if not (0 <= sel.swipe_id < len(swipes)):
            raise HTTPException(400, "Swipe index out of range")
        m["swipes"] = swipes
        m["swipe_id"] = sel.swipe_id
        m["content"] = swipes[sel.swipe_id]
        storage.save_story(story)
        return m


@app.delete("/api/stories/{sid}/message/{idx}")
async def delete_from(sid: str, idx: int):
    """Truncate the story to messages before `idx` (inclusive delete of idx…end)."""
    async with _lock(sid):
        story = storage.get_story(sid)
        if not story:
            raise HTTPException(404, "Story not found")
        msgs = story.get("messages", [])
        if idx < 0 or idx > len(msgs):
            raise HTTPException(404, "Message index out of range")
        story["messages"] = msgs[:idx]
        _invalidate_summaries(story, idx)
        storage.save_story(story)
        return {"ok": True, "length": len(story["messages"])}


# --- lorebooks ----------------------------------------------------------------

@app.get("/api/lorebooks")
async def get_lorebooks():
    return storage.list_lorebooks()


@app.get("/api/lorebooks/{bid}")
async def get_lorebook(bid: str):
    book = storage.get_lorebook(bid)
    if not book:
        raise HTTPException(404, "Lorebook not found")
    return book


@app.post("/api/lorebooks")
async def create_lorebook(book: Lorebook):
    return storage.save_lorebook(book.model_dump())


@app.put("/api/lorebooks/{bid}")
async def update_lorebook(bid: str, book: Lorebook):
    data = book.model_dump()
    data["id"] = bid
    return storage.save_lorebook(data)


@app.delete("/api/lorebooks/{bid}")
async def remove_lorebook(bid: str):
    if not storage.delete_lorebook(bid):
        raise HTTPException(404, "Lorebook not found")
    return {"ok": True}


# --- personas -----------------------------------------------------------------

@app.get("/api/personas")
async def get_personas():
    return storage.list_personas()


@app.post("/api/personas")
async def create_persona(persona: Persona):
    return storage.save_persona(persona.model_dump())


@app.put("/api/personas/{pid}")
async def update_persona(pid: str, persona: Persona):
    data = persona.model_dump()
    data["id"] = pid
    return storage.save_persona(data)


@app.delete("/api/personas/{pid}")
async def remove_persona(pid: str):
    if not storage.delete_persona(pid):
        raise HTTPException(404, "Persona not found")
    return {"ok": True}


# --- static UI ----------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/avatars", StaticFiles(directory=str(storage.AVATAR_DIR)), name="avatars")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
