"""Durable memory for the roleplay bot — the facts a character *keeps*.

Two scopes, both plain JSON living inside the documents they belong to (so they
ride along in exports and backups for free):

  • story memories   — story["memories"]      (this scene only)
  • character memories — character["memories"] (persist across every story with
    that character, so they genuinely "remember you")

Memories are captured two ways:
  • manual — the user adds/edits/pins them in the UI
  • automatic — every few turns the local model extracts NEW durable facts from
    the recent exchange (names, preferences, relationships, promises, secrets,
    appearance, major events) and merges them into BOTH scopes.

This is deliberately distinct from RAG (which recalls raw older message chunks)
and summaries (which compress message ranges): memories are short, curated,
always-available facts. Everything here degrades to a safe no-op if the model
or network is unavailable, so auto-capture can be left on.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import httpx

from . import storage

# How many NEW messages must accumulate before an auto-extraction runs. Kept
# above a couple of turns so extraction is occasional, not every reply.
MEMORY_EVERY = 8
# Only look this far back when extracting, so a long cold story can't send a
# giant transcript to the model in one call.
MEMORY_LOOKBACK = 40
STORY_CAP = 80           # bound how many memories each scope stores
CHARACTER_CAP = 120
MAX_PER_EXTRACTION = 8   # cap how many new facts one extraction can add
MAX_MEMORY_LEN = 280     # a memory is a short fact, not a paragraph


MEMORY_EXTRACT_PROMPT = (
    "You are a memory archivist for a roleplay. From the exchange below, extract "
    "ONLY durable, long-term facts worth remembering in future scenes: names and "
    "identities, physical appearance, personality traits revealed, preferences and "
    "dislikes, relationships and how people feel about each other, promises, threats "
    "or commitments, important possessions, locations, secrets, and major events or "
    "decisions. Ignore small talk and moment-to-moment action. Do NOT repeat anything "
    "under 'Already known'. Respond with ONLY a JSON array of short, self-contained "
    "factual strings in the present tense, e.g. [\"Sam is afraid of fire.\", \"Aria "
    "owns a silver dagger.\"]. If there is nothing new worth remembering, respond "
    "with exactly []."
)


# --- shaping / dedupe ---------------------------------------------------------

def _norm(text: str) -> str:
    """Normalize a memory for duplicate detection: lowercase, collapse
    whitespace, drop trailing punctuation."""
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .!?,;:")


def new_memory(text: str, source: str) -> dict:
    return {"id": storage.new_id(), "text": text[:MAX_MEMORY_LEN].strip(),
            "pinned": False, "source": source, "created": time.time()}


def enforce_cap(mems: list[dict], cap: int) -> None:
    """Keep the list within `cap`, dropping the OLDEST unpinned first. Pinned
    memories are always kept (even past the cap)."""
    if len(mems) <= cap:
        return
    pinned = [m for m in mems if m.get("pinned")]
    unpinned = [m for m in mems if not m.get("pinned")]
    room = max(0, cap - len(pinned))
    # newest unpinned kept; mems stays in insertion order otherwise
    keep_unpinned = set(id(m) for m in sorted(unpinned, key=lambda m: m.get("created", 0))[-room:]) if room else set()
    mems[:] = [m for m in mems if m.get("pinned") or id(m) in keep_unpinned]


def merge_memories(existing: list[dict], new_texts: list[str], source: str, cap: int) -> list[dict]:
    """Append each genuinely-new fact (not a near-duplicate of one already held),
    then enforce the cap. Returns the memories actually added."""
    seen = {_norm(m.get("text", "")) for m in existing}
    seen.discard("")
    added: list[dict] = []
    for t in new_texts:
        t = (t or "").strip()
        if not t:
            continue
        n = _norm(t)
        if not n or n in seen:
            continue
        # containment guard: skip "Sam is afraid" if "Sam is afraid of fire" exists (or vice-versa)
        if any(n in s or s in n for s in seen):
            continue
        seen.add(n)
        mem = new_memory(t, source)
        existing.append(mem)
        added.append(mem)
        if len(added) >= MAX_PER_EXTRACTION:
            break
    enforce_cap(existing, cap)
    # a capped-out memory shouldn't be reported as "added"
    return [m for m in added if m in existing]


def find(mems: list[dict], mid: str) -> Optional[dict]:
    return next((m for m in mems if m.get("id") == mid), None)


# --- model-driven extraction --------------------------------------------------

def _extract_json_array(text: str) -> Optional[list]:
    """Pull the first JSON array out of a model reply (tolerates prose / fences).
    Returns None if there's no parseable array."""
    if not text:
        return None
    i = text.find("[")
    if i == -1:
        return None
    try:
        arr, _ = json.JSONDecoder().raw_decode(text[i:])
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def _render_transcript(msgs: list[dict], char_name: str, user_name: str) -> str:
    lines = []
    for m in msgs:
        who = user_name if m.get("role") == "user" else (m.get("name") or char_name)
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{who}: {content}")
    return "\n".join(lines)


async def extract_memories(transcript: str, existing_texts: list[str], model: str,
                           ollama_url: str) -> Optional[list[str]]:
    """Ask the local model for NEW durable facts. Returns a (possibly empty) list
    of fact strings on success, or None on a hard failure (so the caller can
    retry on a later turn instead of skipping the block)."""
    if not transcript.strip() or not model:
        return []
    known = "\n".join(f"- {t}" for t in existing_texts[-60:])
    user = (f"Already known (do not repeat):\n{known or '(nothing yet)'}\n\n"
            f"Exchange:\n{transcript}")
    payload = {
        "model": model, "stream": False, "keep_alive": "30m",
        "messages": [{"role": "system", "content": MEMORY_EXTRACT_PROMPT},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0.0, "num_predict": 300, "num_ctx": 8192},
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            out = r.json().get("message", {}).get("content", "")
    except Exception:
        return None  # hard failure -> let a later turn try this block again
    # strip reasoning blocks the same way the main path does
    out = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", out, flags=re.DOTALL | re.IGNORECASE)
    arr = _extract_json_array(out)
    if arr is None:
        return []  # model answered but gave no usable array -> nothing new
    facts = []
    for item in arr:
        s = (item if isinstance(item, str) else str(item)).strip()
        if s:
            facts.append(s)
    return facts[:MAX_PER_EXTRACTION]


async def auto_extract(story: dict, character: Optional[dict], model: str,
                       ollama_url: str, user_name: str) -> bool:
    """Extract memories from the messages added since the last run and merge them
    into the story and (if there's a real character) that character's lasting
    bank. Advances story['memory_cursor'] and persists. Returns True iff new
    memories were stored. Safe to call every turn — it self-throttles."""
    msgs = [m for m in story.get("messages", []) if m.get("role") in ("user", "assistant")]
    cursor = int(story.get("memory_cursor") or 0)
    if len(msgs) - cursor < MEMORY_EVERY:
        return False

    char_name = (character or {}).get("name") or story.get("character_name") or "the character"
    block = msgs[max(cursor, len(msgs) - MEMORY_LOOKBACK):]
    transcript = _render_transcript(block, char_name, user_name)

    story_mems = story.setdefault("memories", [])
    char_id = (character or {}).get("id")
    char_fresh = storage.get_character(char_id) if char_id else None
    char_mems = (char_fresh or {}).get("memories", []) if char_fresh else []
    existing_texts = [m.get("text", "") for m in story_mems] + [m.get("text", "") for m in char_mems]

    new_facts = await extract_memories(transcript, existing_texts, model, ollama_url)
    if new_facts is None:
        return False  # hard failure: don't advance the cursor, retry next turn

    added = merge_memories(story_mems, new_facts, "auto", STORY_CAP)
    if char_fresh is not None and new_facts:
        cmems = char_fresh.setdefault("memories", [])
        merge_memories(cmems, new_facts, "auto", CHARACTER_CAP)
        storage.save_character(char_fresh)

    story["memory_cursor"] = len(msgs)
    storage.save_story(story)
    return bool(added)
