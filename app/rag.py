"""Optional semantic recall (RAG) for long stories.

Keyword lorebooks miss relevant old moments that use different words. This
embeds older messages + summaries with a local Ollama embedding model
(default nomic-embed-text), stores the vectors in a sidecar, and at generation
time retrieves the top-K most relevant chunks that aren't already in the live
window — injected as "[Relevant earlier moments]".

nomic-embed-text requires a task prefix: "search_document: " for stored chunks
and "search_query: " for queries. Ollama returns UNnormalized vectors, so we
L2-normalize in Python and cosine == dot product. Everything degrades to a
no-op (empty string) if the embed model isn't available, so the feature is
safe to leave on.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

import httpx

from . import storage

DEFAULT_EMBED_MODEL = "nomic-embed-text"
KEEP_RAW = 12          # most-recent messages considered "live" (not retrieved)
CHUNK_WORDS = 400
CHUNK_OVERLAP = 50


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def _normalize(vec):
    if not vec:
        return None
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return None
    return [x / norm for x in vec]


def cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))  # inputs are already normalized


def chunk_messages(msgs: list[dict]) -> list[str]:
    if not msgs:
        return []
    text = "\n".join(
        ("User" if m.get("role") == "user" else "Char") + ": " + (m.get("content") or "")
        for m in msgs
    )
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_WORDS]))
        if i + CHUNK_WORDS >= len(words):
            break
        i += step
    return chunks


async def embed_texts(texts: list[str], model: str, ollama_url: str) -> list[Optional[list]]:
    """Return an L2-normalized vector per text (None on failure)."""
    if not texts:
        return []
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{ollama_url}/api/embed", json={"model": model, "input": texts})
            r.raise_for_status()
            embs = r.json().get("embeddings") or []
        out = [_normalize(e) for e in embs]
        if len(out) != len(texts):  # defensive
            out += [None] * (len(texts) - len(out))
        return out
    except Exception:
        return [None] * len(texts)


def _target_texts(story: dict) -> dict:
    """{key: text} for everything that should be embedded (summaries + older msgs)."""
    msgs = story.get("messages", [])
    older = msgs[: max(0, len(msgs) - KEEP_RAW)]
    texts = []
    for s in story.get("summaries", []):
        t = (s.get("text") or "").strip()
        if t:
            texts.append(t)
    texts += chunk_messages(older)
    return {_key(t): t for t in texts if t.strip()}


async def update_store(story: dict, model: str, ollama_url: str) -> Optional[dict]:
    """Embed any new/changed chunks, prune stale ones, persist. Returns the store."""
    sid = story.get("id")
    if not sid:
        return None
    want = _target_texts(story)
    store = storage.get_vec(sid) or {"model": model, "chunks": []}
    if store.get("model") != model:
        store = {"model": model, "chunks": []}  # model changed -> rebuild
    store["chunks"] = [c for c in store["chunks"] if c["key"] in want]  # prune stale
    have = {c["key"] for c in store["chunks"]}
    missing = [(k, t) for k, t in want.items() if k not in have]
    if missing:
        vecs = await embed_texts(["search_document: " + t for _, t in missing], model, ollama_url)
        for (k, t), v in zip(missing, vecs):
            if v:
                store["chunks"].append({"key": k, "text": t, "vec": v})
        storage.save_vec(sid, store)
    return store


async def retrieve(story: dict, query_text: str, model: str, ollama_url: str, k: int = 3) -> str:
    """Return a '[Relevant earlier moments]' block, or '' if nothing applies."""
    store = storage.get_vec(story.get("id"))
    if not store or not store.get("chunks") or not (query_text or "").strip():
        return ""
    qv = await embed_texts(["search_query: " + query_text], model, ollama_url)
    if not qv or not qv[0]:
        return ""
    q = qv[0]
    recent = "\n".join(m.get("content", "") for m in story.get("messages", [])[-KEEP_RAW:])
    ranked = sorted(store["chunks"], key=lambda c: cosine(q, c.get("vec")), reverse=True)
    picked = []
    for c in ranked:
        if c["text"] in recent:   # already in the live window
            continue
        picked.append(c["text"])
        if len(picked) >= k:
            break
    if not picked:
        return ""
    return "[Relevant earlier moments]\n" + "\n---\n".join(picked)
