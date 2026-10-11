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
import copy
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from . import cards, context, rag, storage
from .samplers import DEFAULTS, build_options, pick_sampler_fields, preset_for

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# When set (run.py sets it automatically if bound beyond loopback), every /api/
# request must carry this token — so a non-loopback bind can't expose all data.
API_TOKEN = os.environ.get("RP_TOKEN", "").strip()
ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI(title="AI Roleplayer")
storage.seed_presets()


@app.middleware("http")
async def _require_token(request: Request, call_next):
    if API_TOKEN and request.url.path.startswith("/api/"):
        tok = request.headers.get("x-rp-token") or request.query_params.get("token")
        if tok != API_TOKEN:
            return JSONResponse({"detail": "Unauthorized — access token required."}, status_code=401)
    return await call_next(request)

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
    # Optional structured profile (SimsChat three-tier: pre-defined / personal / social)
    age: str = ""
    gender: str = ""
    career: str = ""
    aspiration: str = ""
    traits: str = ""
    skills: str = ""
    speech_style: str = ""
    relationships: str = ""
    living: str = ""


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
    mood: Optional[str] = None
    scene: Optional[str] = None


class ChatReq(BaseModel):
    model_config = ConfigDict(extra="allow")
    story_id: str
    action: str = "send"           # send | regenerate | swipe | continue | group_speak
    user_message: Optional[str] = None
    char_id: Optional[str] = None  # group_speak: which member should reply
    note: Optional[str] = None     # one-off steering for THIS turn (not persisted)
    settings: dict = {}


class MessageEdit(BaseModel):
    content: str
    swipe_id: Optional[int] = None


class SwipeSelect(BaseModel):
    swipe_id: int


class BranchCreate(BaseModel):
    name: Optional[str] = None


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


class SamplerPreset(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: str
    settings: dict = {}


class ImportReq(BaseModel):
    filename: str = "card"
    data_b64: str


class AvatarReq(BaseModel):
    data_b64: str
    ext: str = "png"


class GenCharReq(BaseModel):
    concept: str = ""
    model: str = ""


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


def _attachment_ascii(fname: str, fallback: str = "download") -> str:
    """ASCII + PRINTABLE download name for a Content-Disposition filename=.

    Control chars (CR/LF/NUL) are ASCII, so an unfiltered title would land a
    newline in the header — a crash under a strict ASGI server and an HTTP
    header-injection vector under a lax one. The full/original name is still
    conveyed via the RFC 5987 filename* (percent-encoded), so nothing is lost.
    """
    out = "".join(c for c in (fname or "") if c.isascii() and c.isprintable() and c not in '"\\')
    return out or fallback


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model reply (tolerates code fences /
    surrounding prose)."""
    if not text:
        return None
    t = text.strip()
    i = t.find("{")
    if i == -1:
        return None
    try:
        # raw_decode parses the FIRST complete object and ignores any trailing
        # text (e.g. a note the model appended), which slicing to the last '}'
        # would wrongly fold in and invalidate.
        obj, _ = json.JSONDecoder().raw_decode(t[i:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def effective_settings(story: dict, override: dict, char_defaults: Optional[dict] = None) -> dict:
    """preset(model)  <-  character defaults  <-  story.settings  <-  per-call override.

    A character's `default_settings` sit just above the model preset, so a story
    or an explicit override always wins. They can carry sampler knobs and the RP
    toggles but never the model (that is fixed from the story/override below)."""
    model = (override or {}).get("model") or story.get("model") or ""
    s = preset_for(model)
    if char_defaults:
        s.update(pick_sampler_fields(char_defaults))
    s.update(story.get("settings") or {})
    s.update(override or {})
    s["model"] = model
    return s


def _is_group(story: dict) -> bool:
    return bool(story.get("member_ids"))


def _member_chars(story: dict) -> list:
    out = []
    for cid in story.get("member_ids", []):
        c = storage.get_character(cid)
        if c:
            out.append(c)
    return out


def _group_active(story: dict, req: "ChatReq", members: list):
    """Pick which group member speaks for this turn."""
    if not members:
        return None
    by_id = {c["id"]: c for c in members}
    ids = [c["id"] for c in members]
    if req.action == "group_speak":
        return by_id.get(req.char_id) or members[0]
    if req.action in ("regenerate", "swipe", "continue"):
        for m in reversed(story.get("messages", [])):
            if m.get("role") == "assistant":
                return by_id.get(m.get("char_id")) or members[0]
        return members[0]
    # send: a named member replies; otherwise round-robin after the last speaker.
    text = (req.user_message or "").lower()
    for c in members:
        if re.search(r"\b" + re.escape((c.get("name") or "").lower()) + r"\b", text):
            return c
    last = next((m.get("char_id") for m in reversed(story.get("messages", []))
                 if m.get("role") == "assistant"), None)
    if last in ids:
        return members[(ids.index(last) + 1) % len(members)]
    return members[0]


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
        who = "User" if m.get("role") == "user" else (m.get("name") or story.get("character_name") or "Character")
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


MAX_BRANCHES = 50


def _branch_meta(b: dict) -> dict:
    """Lightweight view of a saved branch (no message bodies)."""
    return {"id": b.get("id"), "name": b.get("name", "Timeline"),
            "created": b.get("created"), "auto": bool(b.get("auto")),
            "turns": len(b.get("messages", []))}


def _story_response(story: dict) -> dict:
    """Story as sent to the UI: branches reduced to metadata so opening a story
    (and the post-turn re-sync) never ships every branch's full message history.
    The on-disk story and the JSON export keep the full branch bodies."""
    if not story.get("branches"):
        return story
    out = dict(story)
    out["branches"] = [_branch_meta(b) for b in story["branches"]]
    return out


def _snapshot(story: dict, name: str, auto: bool = False) -> dict:
    return {"id": storage.new_id(), "name": name[:120], "created": time.time(),
            "auto": auto, "messages": copy.deepcopy(story.get("messages", [])),
            "summaries": copy.deepcopy(story.get("summaries", []))}


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


@app.post("/api/generate_character")
async def generate_character(req: GenCharReq):
    """Draft a character card from a one-line concept using the local model."""
    concept = (req.concept or "").strip()
    if not concept:
        raise HTTPException(400, "Describe the character you want in a sentence or two.")
    if len(concept) > 2000:
        raise HTTPException(400, "Concept is too long.")
    model = (req.model or "").strip()
    if not model:
        raise HTTPException(400, "No model selected. Pick one first.")
    sys_prompt = (
        "You design characters for collaborative roleplay. Given a concept, respond with "
        "ONLY a JSON object (no prose, no markdown fence) with these string fields: "
        '"name", "description", "personality", "scenario", "greeting". '
        "The greeting is the character's immersive in-character opening message to the user "
        "(2-4 sentences, with *actions* in asterisks and \"speech\" in quotes). "
        "Make description and personality vivid but concise. Output valid JSON only."
    )
    payload = {
        "model": model, "stream": False, "keep_alive": "30m",
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": f"Concept: {concept}"}],
        "options": {"temperature": 0.9, "num_predict": 900, "num_ctx": 8192},
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            out = r.json().get("message", {}).get("content", "")
    except Exception as e:
        raise HTTPException(502, f"Could not reach Ollama at {OLLAMA_URL} ({e}).")
    data = _extract_json(context.strip_think(out))
    if not data:
        raise HTTPException(502, "The model didn't return usable character JSON — try again.")
    fields = {k: str(data.get(k, "") or "").strip()
              for k in ("name", "description", "personality", "scenario", "greeting")}
    if not fields["name"]:
        fields["name"] = concept[:40]
    return {"ok": True, "character": fields}


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
            settings = effective_settings(story, req.settings, character.get("default_settings"))
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
                if len(req.user_message) > 100_000:
                    yield _sse({"error": "Message is too long (max 100k characters)."})
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
            elif action == "group_speak":
                if not _is_group(story):
                    yield _sse({"error": "Not a group story."})
                    return
                ctx_msgs = msgs
            elif action == "impersonate":
                # Draft {{user}}'s next line in their voice. Read-only: nothing
                # is ever written to the story (the suggestion only fills the box).
                ctx_msgs = msgs
            else:
                yield _sse({"error": f"Unknown action: {action}"})
                return

            # Group scenes: one member speaks per turn. Pick the active speaker
            # and swap in their card; label past turns so speakers are distinct.
            user_name = (persona or {}).get("name") or settings.get("user_name") or "You"
            members = _member_chars(story) if _is_group(story) else []
            # Impersonation writes the USER's line, so don't select a group speaker
            # — but still keep the roster + labeled history so the draft fits the
            # scene and doesn't blur the members together.
            grp_imp = action == "impersonate" and bool(members)
            active = _group_active(story, req, members) if (members and action != "impersonate") else None
            if active:
                character = active
                lb_id = active.get("lorebook_id") or story.get("lorebook_id")
                lorebook = storage.get_lorebook(lb_id) if lb_id else None
                # Re-layer with the active speaker's own defaults (model stays fixed).
                settings = effective_settings(story, req.settings, active.get("default_settings"))

            ctx_story = dict(story)
            ctx_story["messages"] = (context.label_group_history(ctx_msgs)
                                     if (active or grp_imp) else ctx_msgs)

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

            if active:
                assemble_char = context.build_group_character(active, members)
            elif grp_imp:
                assemble_char = context.build_scene_character(character, members)
            else:
                assemble_char = character
            built = context.assemble_prompt(ctx_story, assemble_char, persona, lorebook,
                                            settings, rag_block=rag_block)
            messages = built["messages"]
            if action == "continue":
                messages.append({"role": "system", "content": (
                    f"[Continue {character.get('name', 'the character')}'s previous reply "
                    f"from exactly where it left off. Do not repeat earlier text.]")})
            if action == "impersonate":
                messages.append({"role": "system", "content": (
                    f"[For this turn only, write {user_name}'s next message, in first person "
                    f"as {user_name} — 1 to 3 sentences in their established voice. Output ONLY "
                    f"{user_name}'s own words and actions; do not write for "
                    f"{character.get('name', 'the character')} or narrate other characters.]")})
            # One-off steering for this turn only (guided regenerate / swipe). Goes
            # last so it's the strongest instruction, and is never persisted.
            note = (req.note or "").strip()[:2000]
            if note:
                messages.append({"role": "system", "content": (
                    f"[Director's note — apply this to the reply you are about to write; "
                    f"do not mention or quote this note: {note}]")})

            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "keep_alive": settings.get("keep_alive", "30m"),
                "options": build_options(settings),
            }

            full = ""
            prompt_tokens = eval_tokens = None
            # read timeout = max gap between tokens; a stalled Ollama then raises
            # (surfacing an error + releasing the per-story lock) instead of hanging forever.
            chat_timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=None)
            try:
                async with httpx.AsyncClient(timeout=chat_timeout) as client:
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
            if active and clean.startswith(active["name"] + ":"):
                clean = clean[len(active["name"]) + 1:].lstrip()  # model echoed the name
            if not clean.strip():
                yield _sse({"error": "The model returned an empty reply. Try again, "
                                     "lower the temperature, or pick another model."})
                return

            # Impersonation is read-only: return the drafted line, persist nothing.
            if action == "impersonate":
                yield _sse({"done": True, "impersonate": True, "text": clean,
                            "prompt_tokens": prompt_tokens, "eval_tokens": eval_tokens,
                            "num_ctx": settings.get("num_ctx")})
                return

            def _attach(msg):
                if active:
                    msg["char_id"] = active["id"]
                    msg["name"] = active["name"]
                return msg

            # Commit the result to the right place (first write to disk).
            if action in ("send", "group_speak"):
                msgs.append(_attach({"role": "assistant", "content": clean,
                                     "swipes": [clean], "swipe_id": 0}))
                idx = len(msgs) - 1
            elif action == "regenerate":
                if msgs and msgs[-1].get("role") == "assistant":
                    msgs.pop()               # drop old reply only now that we have a new one
                msgs.append(_attach({"role": "assistant", "content": clean,
                                     "swipes": [clean], "swipe_id": 0}))
                idx = len(msgs) - 1
            elif action == "swipe":
                last = msgs[-1]
                last.setdefault("swipes", [last.get("content", "")])
                last["swipes"].append(clean)
                if len(last["swipes"]) > 10:        # bound stored alternates
                    last["swipes"] = last["swipes"][-10:]
                last["swipe_id"] = len(last["swipes"]) - 1
                last["content"] = clean
                _attach(last)   # keep attribution matching the card actually used
                idx = len(msgs) - 1
            else:  # continue
                last = msgs[-1]
                sep = "" if last.get("content", "").endswith((" ", "\n")) else " "
                last["content"] = last.get("content", "") + sep + clean
                if last.get("swipes"):
                    last["swipes"][last.get("swipe_id", 0)] = last["content"]
                _attach(last)
                idx = len(msgs) - 1

            storage.save_story(story)
            summarized = False
            if action in ("send", "regenerate", "group_speak"):
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
    settings = effective_settings(story, {}, character.get("default_settings"))
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
    ascii_name = _attachment_ascii(fname, "character")
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
    return _story_response(story)


@app.get("/api/stories/{sid}/export")
async def export_story(sid: str):
    story = storage.get_story(sid)
    if not story:
        raise HTTPException(404, "Story not found")
    fname = (story.get("title") or "story").replace(" ", "_")
    ascii_name = _attachment_ascii(fname, "story")
    return JSONResponse(story, headers={
        "Content-Disposition":
            f"attachment; filename=\"{ascii_name}.json\"; filename*=UTF-8''{quote(fname)}.json"})


@app.get("/api/stories/{sid}/transcript")
async def export_transcript(sid: str):
    """Export a story as a readable Markdown transcript (human-friendly, unlike
    the JSON export which is for re-import)."""
    story = storage.get_story(sid)
    if not story:
        raise HTTPException(404, "Story not found")
    char = story.get("character_name") or "Character"
    out = [f"# {story.get('title', 'Untitled')}", "", f"*Character: {char}*", ""]
    for m in story.get("messages", []):
        role = m.get("role")
        if role == "user":
            who = "You"
        elif role == "assistant":
            who = m.get("name") or char
        else:
            continue
        content = (m.get("content") or "").strip()
        if content:
            out.append(f"**{who}:** {content}")
            out.append("")
    md = "\n".join(out)
    fname = (story.get("title") or "story").replace(" ", "_")
    ascii_name = _attachment_ascii(fname, "story")
    return Response(content=md, media_type="text/markdown", headers={
        "Content-Disposition":
            f"attachment; filename=\"{ascii_name}.md\"; filename*=UTF-8''{quote(fname)}.md"})


@app.post("/api/stories/import")
async def import_story(req: ImportReq):
    try:
        raw = base64.b64decode(req.data_b64)
    except Exception:
        raise HTTPException(400, "Could not read the story file.")
    if len(raw) > 25_000_000:
        raise HTTPException(400, "Story file is too large.")
    try:
        obj = json.loads(raw.decode("utf-8", "ignore"))
    except Exception:
        raise HTTPException(400, "Could not read the story file.")
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        raise HTTPException(400, "Not a valid story export.")
    msgs = obj["messages"]
    if not all(isinstance(m, dict) and m.get("role") in ("user", "assistant", "system") for m in msgs):
        raise HTTPException(400, "Story file has malformed messages.")
    for m in msgs:  # normalize assistant turns into swipe shape, like create_story
        if m.get("role") == "assistant" and "swipes" not in m:
            m["swipes"] = [m.get("content", "")]
            m["swipe_id"] = 0
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
    return _story_response(storage.save_story(data))


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
        return _story_response(storage.save_story(story))


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


# --- branches (saved alternate timelines) -------------------------------------

@app.get("/api/stories/{sid}/branches")
async def list_branches(sid: str):
    story = storage.get_story(sid)
    if not story:
        raise HTTPException(404, "Story not found")
    return [_branch_meta(b) for b in story.get("branches", [])]


@app.post("/api/stories/{sid}/branches")
async def create_branch(sid: str, req: BranchCreate):
    async with _lock(sid):
        story = storage.get_story(sid)
        if not story:
            raise HTTPException(404, "Story not found")
        branches = story.setdefault("branches", [])
        name = (req.name or "").strip() or f"Timeline {len(branches) + 1}"
        branch = _snapshot(story, name)
        branches.append(branch)
        if len(branches) > MAX_BRANCHES:        # bound file growth; drop oldest
            del branches[0:len(branches) - MAX_BRANCHES]
        storage.save_story(story)
        return {"ok": True, "branch": _branch_meta(branch), "count": len(branches)}


@app.post("/api/stories/{sid}/branches/{bid}/restore")
async def restore_branch(sid: str, bid: str):
    async with _lock(sid):
        story = storage.get_story(sid)
        if not story:
            raise HTTPException(404, "Story not found")
        branches = story.setdefault("branches", [])
        target = next((b for b in branches if b.get("id") == bid), None)
        if not target:
            raise HTTPException(404, "Branch not found")
        # Never destroy the live timeline: keep ONE rolling auto-snapshot of it
        # so an accidental restore is always reversible.
        branches[:] = [b for b in branches if not b.get("auto")]
        branches.append(_snapshot(story, "⏱ Before restore (auto)", auto=True))
        story["messages"] = copy.deepcopy(target.get("messages", []))
        story["summaries"] = copy.deepcopy(target.get("summaries", []))
        return _story_response(storage.save_story(story))


@app.delete("/api/stories/{sid}/branches/{bid}")
async def delete_branch(sid: str, bid: str):
    async with _lock(sid):
        story = storage.get_story(sid)
        if not story:
            raise HTTPException(404, "Story not found")
        branches = story.get("branches", [])
        kept = [b for b in branches if b.get("id") != bid]
        if len(kept) == len(branches):
            raise HTTPException(404, "Branch not found")
        story["branches"] = kept
        storage.save_story(story)
        return {"ok": True}


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


# --- sampler presets (named generation-setting bundles) -----------------------

@app.get("/api/sampler_presets")
async def get_sampler_presets():
    return storage.list_sampler_presets()


@app.post("/api/sampler_presets")
async def create_sampler_preset(preset: SamplerPreset):
    data = preset.model_dump()
    data["settings"] = pick_sampler_fields(data.get("settings") or {})
    return storage.save_sampler_preset(data)


@app.put("/api/sampler_presets/{pid}")
async def update_sampler_preset(pid: str, preset: SamplerPreset):
    data = preset.model_dump()
    data["id"] = pid
    data["settings"] = pick_sampler_fields(data.get("settings") or {})
    return storage.save_sampler_preset(data)


@app.delete("/api/sampler_presets/{pid}")
async def remove_sampler_preset(pid: str):
    if not storage.delete_sampler_preset(pid):
        raise HTTPException(404, "Preset not found")
    return {"ok": True}


# --- static UI ----------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/avatars", StaticFiles(directory=str(storage.AVATAR_DIR)), name="avatars")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
