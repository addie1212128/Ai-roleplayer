"""Tavern / SillyTavern character-card import & export (V1, V2, V3).

Lets the bot read the de-facto community card standard so thousands of existing
characters (chub.ai / characterhub, etc.) can be imported for free. Cards embed
JSON either as a plain .json file or inside a PNG's tEXt chunk keyed 'chara'
(V2, base64) or 'ccv3' (V3, base64, preferred when present).

Everything here treats card content as UNTRUSTED DATA: we only JSON-parse it and
map text fields. We never execute anything. (Downloaded cards can still carry
prompt-injection text in their description — that's a behavioral surface, not a
code-exec one, and is surfaced to the user in the editor before use.)
"""
from __future__ import annotations

import base64
import binascii
import json
import struct
import zlib
from typing import Optional

PNG_SIG = b"\x89PNG\r\n\x1a\n"
MAX_DECOMPRESS = 4_000_000   # hard cap (4 MB) on any single decompressed chunk
MAX_CARD_BYTES = 32_000_000  # reject uploads larger than 32 MB outright


def _bounded_inflate(data: bytes):
    """zlib-decompress with a hard output cap, to defuse zip bombs.

    Returns the bytes, or None if the output would exceed the cap or the data
    is not valid zlib.
    """
    try:
        d = zlib.decompressobj()
        out = d.decompress(data, MAX_DECOMPRESS)
        if d.unconsumed_tail:          # more output remained past the cap
            return None
        out += d.flush()
        return out if len(out) <= MAX_DECOMPRESS else None
    except zlib.error:
        return None


def _as_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# PNG text-chunk extraction
# ---------------------------------------------------------------------------

def _iter_png_text_chunks(data: bytes):
    """Yield (keyword, text) for tEXt / iTXt / zTXt chunks in a PNG."""
    if not data.startswith(PNG_SIG):
        return
    i = len(PNG_SIG)
    n = len(data)
    while i + 8 <= n:
        (length,) = struct.unpack(">I", data[i:i + 4])
        ctype = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + length]
        i += 12 + length  # 4 len + 4 type + data + 4 crc
        if ctype == b"tEXt":
            if b"\x00" in body:
                kw, text = body.split(b"\x00", 1)
                yield kw.decode("latin-1", "ignore"), text.decode("latin-1", "ignore")
        elif ctype == b"zTXt":
            if b"\x00" in body:
                kw, rest = body.split(b"\x00", 1)
                if rest:
                    raw = _bounded_inflate(rest[1:])  # rest[0] is the method byte
                    if raw is not None:
                        yield kw.decode("latin-1", "ignore"), raw.decode("latin-1", "ignore")
        elif ctype == b"iTXt":
            parts = body.split(b"\x00", 2)
            if len(parts) >= 3:
                kw = parts[0]
                comp_flag = parts[1][:1]
                rest = parts[2]
                # rest = language\0 translated\0 text
                sub = rest.split(b"\x00", 2)
                text_bytes = sub[2] if len(sub) == 3 else rest
                if comp_flag == b"\x01":
                    text_bytes = _bounded_inflate(text_bytes)
                    if text_bytes is None:
                        continue
                yield kw.decode("latin-1", "ignore"), text_bytes.decode("utf-8", "ignore")
        if ctype == b"IEND":
            break


def extract_png_card(data: bytes) -> Optional[dict]:
    """Pull the card JSON out of a PNG. Prefers V3 ('ccv3') over V2 ('chara')."""
    found = {}
    for kw, text in _iter_png_text_chunks(data):
        if kw.lower() in ("chara", "ccv3"):
            found[kw.lower()] = text
    for key in ("ccv3", "chara"):
        if key in found:
            try:
                decoded = base64.b64decode(found[key])
                return json.loads(decoded.decode("utf-8", "ignore"))
            except (binascii.Error, json.JSONDecodeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# card <-> our character mapping
# ---------------------------------------------------------------------------

def _card_data(obj: dict) -> dict:
    """Unwrap the V2/V3 {spec, data:{...}} envelope, or accept a flat V1 card."""
    if isinstance(obj, dict) and isinstance(obj.get("data"), dict) and \
            str(obj.get("spec", "")).startswith("chara_card"):
        return obj["data"]
    return obj  # flat V1


def character_from_card(obj: dict) -> dict:
    """Map a Tavern card (any version) onto our character dict."""
    d = _card_data(obj)
    greeting = d.get("first_mes") or d.get("greeting") or ""
    char = {
        "name": (d.get("name") or "Imported Character").strip(),
        "avatar": "🎭",
        "description": d.get("description", "") or "",
        "personality": d.get("personality", "") or "",
        "scenario": d.get("scenario", "") or "",
        "greeting": greeting,
        "example_dialogue": d.get("mes_example", "") or "",
        "system_prompt": d.get("system_prompt", "") or "",
        "post_history_instructions": d.get("post_history_instructions", "") or "",
        "alternate_greetings": d.get("alternate_greetings", []) or [],
        "tags": d.get("tags", []) or [],
        "creator_notes": d.get("creator_notes", "") or "",
        "pov": "",
    }
    # Embedded lorebook (character_book) is returned separately for the caller
    # to save as a lorebook and link.
    book = d.get("character_book")
    return {"character": char, "character_book": book}


def card_from_character(char: dict) -> dict:
    """Export our character as a chara_card_v2 envelope (widely compatible)."""
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": char.get("name", ""),
            "description": char.get("description", ""),
            "personality": char.get("personality", ""),
            "scenario": char.get("scenario", ""),
            "first_mes": char.get("greeting", ""),
            "mes_example": char.get("example_dialogue", ""),
            "system_prompt": char.get("system_prompt", ""),
            "post_history_instructions": char.get("post_history_instructions", ""),
            "alternate_greetings": char.get("alternate_greetings", []),
            "tags": char.get("tags", []),
            "creator": char.get("creator", ""),
            "character_version": char.get("character_version", ""),
            "creator_notes": char.get("creator_notes", ""),
            "extensions": {},
        },
    }


def lorebook_from_character_book(book: dict, name: str) -> Optional[dict]:
    """Convert a V2 character_book into our lorebook shape."""
    if not book or not isinstance(book, dict):
        return None
    entries_in = book.get("entries", [])
    # V2 entries can be a list or a dict keyed by index.
    if isinstance(entries_in, dict):
        entries_in = list(entries_in.values())
    entries = []
    for e in entries_in:
        if not isinstance(e, dict):
            continue
        entries.append({
            "keys": e.get("keys", []) or e.get("key", []) or [],
            "keysecondary": e.get("secondary_keys", []) or e.get("keysecondary", []) or [],
            "content": e.get("content", "") or "",
            "constant": bool(e.get("constant", False)),
            "selective": bool(e.get("selective", False)),
            "selectiveLogic": "AND_ANY",
            "order": _as_int(e.get("insertion_order", e.get("order", 100)), 100),
            "position": "before" if (e.get("position") in (None, "before_char", 0, "before")) else "after",
            "depth": e.get("depth", 4),
            "matchWholeWords": True,
            "enabled": e.get("enabled", True),
        })
    return {
        "name": name or book.get("name") or "Imported Lorebook",
        "scanDepth": book.get("scan_depth", 4) or 4,
        "tokenBudget": book.get("token_budget"),
        "entries": entries,
    }


def parse_card_bytes(data: bytes) -> dict:
    """Detect .json vs PNG and return the mapped character (+ optional book).

    Raises ValueError if no card can be found.
    """
    if len(data) > MAX_CARD_BYTES:
        raise ValueError("Card file is too large.")
    if data[:8] == PNG_SIG:
        obj = extract_png_card(data)
        if obj is None:
            raise ValueError("No character card found embedded in this PNG.")
        if not isinstance(obj, dict):
            raise ValueError("Embedded card is not a JSON object.")
        return character_from_card(obj)
    # Try JSON.
    try:
        obj = json.loads(data.decode("utf-8", "ignore"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Not a valid card file (expected PNG or JSON): {e}")
    if not isinstance(obj, dict):
        raise ValueError("Card JSON must be an object.")
    return character_from_card(obj)
