"""JSON-file storage for characters, stories, lorebooks, and personas.

Everything lives under ./data/ (git-ignored) so the user's own content stays
local. Bundled example characters in ./presets/characters/ are copied in on
first run. Writes are atomic (temp file + os.replace) so a crash or a second
writer can't leave a half-written / corrupt file.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHAR_DIR = DATA / "characters"
STORY_DIR = DATA / "stories"
LORE_DIR = DATA / "lorebooks"
PERSONA_DIR = DATA / "personas"
PRESET_DIR = ROOT / "presets" / "characters"

_DIRS = (CHAR_DIR, STORY_DIR, LORE_DIR, PERSONA_DIR)


def _ensure_dirs() -> None:
    for d in _DIRS:
        d.mkdir(parents=True, exist_ok=True)


def seed_presets() -> None:
    _ensure_dirs()
    if any(CHAR_DIR.glob("*.json")):
        return
    if PRESET_DIR.exists():
        for f in PRESET_DIR.glob("*.json"):
            shutil.copy(f, CHAR_DIR / f.name)


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write(path: Path, obj: dict) -> None:
    """Atomic write: serialize to a temp file in the same dir, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX and Windows


def _list(dir_: Path) -> list[dict]:
    _ensure_dirs()
    return [o for f in sorted(dir_.glob("*.json")) if (o := _read(f))]


# ----- characters -------------------------------------------------------------

def list_characters() -> list[dict]:
    chars = _list(CHAR_DIR)
    chars.sort(key=lambda c: c.get("name", "").lower())
    return chars


def get_character(cid: str) -> Optional[dict]:
    return _read(CHAR_DIR / f"{cid}.json")


def save_character(char: dict) -> dict:
    _ensure_dirs()
    if not char.get("id"):
        char["id"] = new_id()
        char["created"] = time.time()
    char["updated"] = time.time()
    _write(CHAR_DIR / f"{char['id']}.json", char)
    return char


def delete_character(cid: str) -> bool:
    f = CHAR_DIR / f"{cid}.json"
    if f.exists():
        f.unlink()
        return True
    return False


# ----- stories ----------------------------------------------------------------

def list_stories() -> list[dict]:
    out = []
    for s in _list(STORY_DIR):
        out.append({
            "id": s.get("id"),
            "title": s.get("title", "Untitled"),
            "character_id": s.get("character_id"),
            "character_name": s.get("character_name"),
            "model": s.get("model"),
            "updated": s.get("updated", 0),
            "turns": len(s.get("messages", [])),
        })
    out.sort(key=lambda s: s.get("updated", 0), reverse=True)
    return out


def get_story(sid: str) -> Optional[dict]:
    return _read(STORY_DIR / f"{sid}.json")


def save_story(story: dict) -> dict:
    _ensure_dirs()
    if not story.get("id"):
        story["id"] = new_id()
        story["created"] = time.time()
    story["updated"] = time.time()
    _write(STORY_DIR / f"{story['id']}.json", story)
    return story


def delete_story(sid: str) -> bool:
    f = STORY_DIR / f"{sid}.json"
    if f.exists():
        f.unlink()
        return True
    return False


# ----- lorebooks --------------------------------------------------------------

def list_lorebooks() -> list[dict]:
    books = _list(LORE_DIR)
    books.sort(key=lambda b: b.get("name", "").lower())
    return books


def get_lorebook(bid: str) -> Optional[dict]:
    return _read(LORE_DIR / f"{bid}.json")


def save_lorebook(book: dict) -> dict:
    _ensure_dirs()
    if not book.get("id"):
        book["id"] = new_id()
        book["created"] = time.time()
    book["updated"] = time.time()
    _write(LORE_DIR / f"{book['id']}.json", book)
    return book


def delete_lorebook(bid: str) -> bool:
    f = LORE_DIR / f"{bid}.json"
    if f.exists():
        f.unlink()
        return True
    return False


# ----- personas ---------------------------------------------------------------

def list_personas() -> list[dict]:
    return _list(PERSONA_DIR)


def get_persona(pid: str) -> Optional[dict]:
    return _read(PERSONA_DIR / f"{pid}.json")


def save_persona(persona: dict) -> dict:
    _ensure_dirs()
    if not persona.get("id"):
        persona["id"] = new_id()
        persona["created"] = time.time()
    persona["updated"] = time.time()
    _write(PERSONA_DIR / f"{persona['id']}.json", persona)
    return persona


def delete_persona(pid: str) -> bool:
    f = PERSONA_DIR / f"{pid}.json"
    if f.exists():
        f.unlink()
        return True
    return False
