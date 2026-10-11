"""Simple JSON-file storage for characters and story sessions.

Everything lives under ./data/ (git-ignored) so the user's own characters and
saved roleplays stay local to their machine. Bundled example characters in
./presets/characters/ are copied in on first run.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHAR_DIR = DATA / "characters"
STORY_DIR = DATA / "stories"
PRESET_DIR = ROOT / "presets" / "characters"


def _ensure_dirs() -> None:
    CHAR_DIR.mkdir(parents=True, exist_ok=True)
    STORY_DIR.mkdir(parents=True, exist_ok=True)


def seed_presets() -> None:
    """Copy bundled example characters into data/ the first time we run."""
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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ----- characters -------------------------------------------------------------

def list_characters() -> list[dict]:
    _ensure_dirs()
    chars = [c for f in sorted(CHAR_DIR.glob("*.json")) if (c := _read(f))]
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
    """Return lightweight metadata for each saved story (no full message log)."""
    _ensure_dirs()
    out = []
    for f in sorted(STORY_DIR.glob("*.json")):
        s = _read(f)
        if not s:
            continue
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
