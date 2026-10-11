"""JSON-file storage for characters, stories, lorebooks, and personas.

Everything lives under ./data/ (git-ignored) so the user's own content stays
local. Bundled example characters in ./presets/characters/ are copied in on
first run. Writes are atomic (temp file + os.replace) so a crash or a second
writer can't leave a half-written / corrupt file.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

# IDs become filenames, so restrict them to a safe charset. This blocks path
# traversal (e.g. a client-supplied id of "../../tmp/x") on every store.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _safe_id(oid) -> bool:
    return isinstance(oid, str) and _ID_RE.match(oid) is not None

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHAR_DIR = DATA / "characters"
STORY_DIR = DATA / "stories"
LORE_DIR = DATA / "lorebooks"
PERSONA_DIR = DATA / "personas"
AVATAR_DIR = DATA / "avatars"
PRESET_DIR = ROOT / "presets" / "characters"

_DIRS = (CHAR_DIR, STORY_DIR, LORE_DIR, PERSONA_DIR, AVATAR_DIR)


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


# Generic, id-validated document CRUD shared by every store.

def _doc_get(dir_: Path, oid) -> Optional[dict]:
    if not _safe_id(oid):
        return None
    return _read(dir_ / f"{oid}.json")


def _doc_save(dir_: Path, doc: dict) -> dict:
    # A missing OR unsafe id is replaced with a fresh one, so a client can
    # never steer the write path outside its store.
    if not _safe_id(doc.get("id")):
        doc["id"] = new_id()
        doc["created"] = time.time()
    doc["updated"] = time.time()
    _write(dir_ / f"{doc['id']}.json", doc)
    return doc


def _doc_delete(dir_: Path, oid) -> bool:
    if not _safe_id(oid):
        return False
    f = dir_ / f"{oid}.json"
    if f.exists():
        f.unlink()
        return True
    return False


# ----- characters -------------------------------------------------------------

def list_characters() -> list[dict]:
    chars = _list(CHAR_DIR)
    chars.sort(key=lambda c: c.get("name", "").lower())
    return chars


def get_character(cid: str) -> Optional[dict]:
    return _doc_get(CHAR_DIR, cid)


def save_character(char: dict) -> dict:
    return _doc_save(CHAR_DIR, char)


def delete_character(cid: str) -> bool:
    return _doc_delete(CHAR_DIR, cid)


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
    return _doc_get(STORY_DIR, sid)


def save_story(story: dict) -> dict:
    return _doc_save(STORY_DIR, story)


def delete_story(sid: str) -> bool:
    return _doc_delete(STORY_DIR, sid)


# ----- lorebooks --------------------------------------------------------------

def list_lorebooks() -> list[dict]:
    books = _list(LORE_DIR)
    books.sort(key=lambda b: b.get("name", "").lower())
    return books


def get_lorebook(bid: str) -> Optional[dict]:
    return _doc_get(LORE_DIR, bid)


def save_lorebook(book: dict) -> dict:
    return _doc_save(LORE_DIR, book)


def delete_lorebook(bid: str) -> bool:
    return _doc_delete(LORE_DIR, bid)


# ----- personas ---------------------------------------------------------------

def list_personas() -> list[dict]:
    return _list(PERSONA_DIR)


def get_persona(pid: str) -> Optional[dict]:
    return _doc_get(PERSONA_DIR, pid)


def save_persona(persona: dict) -> dict:
    return _doc_save(PERSONA_DIR, persona)


def delete_persona(pid: str) -> bool:
    return _doc_delete(PERSONA_DIR, pid)
