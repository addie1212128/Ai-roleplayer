"""Evolving relationship + mood between the character and {{user}}.

A character that treats you the same in turn 2 and turn 200 feels dead. This
tracks, per story, how the character currently regards the user — an **affinity**
score (-100 hostile … +100 devoted) and a short **status** label ("wary allies",
"budding trust") — plus the character's **current mood**, and lets all of it
drift with what actually happens in the scene.

Capture mirrors the memory system: every few turns the local model judges the
recent exchange and returns a small affinity delta, an updated status, and a
mood drawn from the UI's mood vocabulary. The user can also set affinity/status
by hand. The current standing is injected into the prompt so the character's
warmth and choices stay consistent with it. Everything degrades to a safe no-op
if the model/network is unavailable, and it is skipped entirely for group scenes.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import httpx

from . import storage

REL_EVERY = 6            # assess once this many new messages accumulate
REL_LOOKBACK = 24        # transcript window handed to the model
MAX_DELTA = 15           # most affinity can move in one assessment
HISTORY_CAP = 30

# Mirrors the mood <select> in index.html so an auto-set mood is a valid option
# the dropdown can display. A returned mood outside this set is ignored.
MOOD_OPTIONS = [
    "Happy", "Sad", "Angry", "Afraid", "Confident", "Nervous", "Flirty", "Playful",
    "Focused", "Tense", "Embarrassed", "Inspired", "Energized", "Calm",
    "Suspicious", "Heartbroken",
]


def default_relationship() -> dict:
    return {"affinity": 0, "status": "", "updated": time.time(), "history": []}


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def band(affinity: int) -> str:
    """A human- and prompt-friendly descriptor for an affinity score."""
    a = int(affinity or 0)
    if a <= -60:
        return "hostile and distrustful"
    if a <= -20:
        return "wary and guarded"
    if a < 20:
        return "neutral — still getting acquainted"
    if a < 60:
        return "warm and friendly"
    return "deeply attached and trusting"


RELATIONSHIP_PROMPT = (
    "You track the emotional arc between two people in a roleplay. Read the recent "
    "exchange and judge how CHARNAME now feels about USERNAME and whether it shifted. "
    "Respond with ONLY a JSON object, no prose:\n"
    '{"affinity_delta": <integer from -15 to 15 — how CHARNAME\'s regard for USERNAME '
    "moved across this exchange; 0 if unchanged>, "
    '"status": "<2 to 4 word label for where they stand now, e.g. wary allies, budding '
    'trust, old friends>", '
    '"mood": "<CHARNAME\'s current mood as ONE of: MOODLIST>", '
    '"reason": "<one short clause>"}'
)


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    i = text.find("{")
    if i == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else None
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


async def assess(transcript: str, char_name: str, user_name: str, current: dict,
                 model: str, ollama_url: str) -> Optional[dict]:
    """Ask the model for an affinity delta + status + mood. Returns a normalized
    dict, or None on a hard failure (so the caller can retry later)."""
    if not transcript.strip() or not model:
        return None
    sys = (RELATIONSHIP_PROMPT
           .replace("CHARNAME", char_name).replace("USERNAME", user_name)
           .replace("MOODLIST", ", ".join(MOOD_OPTIONS)))
    cur_aff = int((current or {}).get("affinity") or 0)
    cur_status = (current or {}).get("status") or "just met"
    user = (f"Current standing: affinity {cur_aff} ({band(cur_aff)}), status \"{cur_status}\".\n\n"
            f"Recent exchange:\n{transcript}")
    payload = {
        "model": model, "stream": False, "keep_alive": "30m",
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        "options": {"temperature": 0.2, "num_predict": 160, "num_ctx": 8192},
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            out = r.json().get("message", {}).get("content", "")
    except Exception:
        return None
    out = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", out, flags=re.DOTALL | re.IGNORECASE)
    obj = _extract_json_object(out)
    if obj is None:
        return None
    try:
        delta = clamp(int(round(float(obj.get("affinity_delta") or 0))), -MAX_DELTA, MAX_DELTA)
    except (TypeError, ValueError):
        delta = 0
    status = str(obj.get("status") or "").strip()[:60]
    mood = str(obj.get("mood") or "").strip().title()
    mood = mood if mood in MOOD_OPTIONS else ""
    reason = str(obj.get("reason") or "").strip()[:160]
    return {"affinity_delta": delta, "status": status, "mood": mood, "reason": reason}


async def auto_update(story: dict, character: Optional[dict], model: str,
                      ollama_url: str, user_name: str) -> Optional[dict]:
    """Evolve the story's relationship + mood from the latest exchange. Advances
    story['rel_cursor'] and persists. Returns the new relationship snapshot when
    it changed, else None. Self-throttles via the cursor."""
    msgs = [m for m in story.get("messages", []) if m.get("role") in ("user", "assistant")]
    cursor = int(story.get("rel_cursor") or 0)
    if len(msgs) - cursor < REL_EVERY:
        return None

    char_name = (character or {}).get("name") or story.get("character_name") or "the character"
    block = msgs[max(cursor, len(msgs) - REL_LOOKBACK):]
    transcript = _render_transcript(block, char_name, user_name)
    current = story.get("relationship") or default_relationship()

    res = await assess(transcript, char_name, user_name, current, model, ollama_url)
    if res is None:
        return None  # hard failure: leave cursor so a later turn retries

    new_aff = clamp(int(current.get("affinity") or 0) + res["affinity_delta"], -100, 100)
    hist = list(current.get("history") or [])
    hist.append({"t": time.time(), "affinity": new_aff,
                 "delta": res["affinity_delta"], "reason": res["reason"]})
    rel = {
        "affinity": new_aff,
        "status": res["status"] or current.get("status") or "",
        "updated": time.time(),
        "history": hist[-HISTORY_CAP:],
        "last_delta": res["affinity_delta"],
    }
    story["relationship"] = rel
    if res["mood"]:
        story["mood"] = res["mood"]          # the character's mood drifts too
    story["rel_cursor"] = len(msgs)
    storage.save_story(story)
    return rel
