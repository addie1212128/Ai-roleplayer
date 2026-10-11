"""Shared, dependency-light helpers for the roleplay fine-tuning toolkit.

Pure Python + stdlib only — NO torch/unsloth/datasets imports here, so this
module (and the unit tests over it) run anywhere, including CI. The heavy
scripts import this for the conventions they all must agree on:

  * the canonical training record shape ({"messages": [{"role","content"}]}),
  * which chat template / response-mask markers go with which base model,
  * how a character + persona become a system prompt,
  * slop phrases to scrub and the exact/near dedup helpers.

Everything here was pinned against the Oct-2026 verified research in
training/README.md — see that file for sources and the "why" behind the
marker strings and template choices.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional


def safe_stdout() -> None:
    """Make stdout tolerant of non-ASCII (✅, ⚠, →) on Windows.

    Native Windows encodes redirected stdout with cp1252, so an emoji in a
    print() raises UnicodeEncodeError — which can make a SUCCESSFUL run (adapter
    already saved) look like a crash. Call this first thing in each entrypoint.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- chat templates -----------------------------------------------------------
# Unsloth's get_chat_template() name  ->  the train_on_responses_only markers
# that mask everything except the assistant turns. The markers MUST match the
# rendered template byte-for-byte or masking silently covers every assistant
# token and the run dies on Unsloth's all-labels-masked safety check.
#
# Verified (Oct 2026): mistral/Tekken markers are "[INST]"/"[/INST]" with NO
# surrounding spaces; ChatML markers include the trailing newline. For families
# where the exact markers are version-sensitive (llama-3.1), markers=None tells
# train_on_responses_only() to auto-detect them from the template.
TEMPLATE_PRESETS: dict[str, dict] = {
    "mistral": {
        "instruction_part": "[INST]",
        "response_part": "[/INST]",
    },
    "chatml": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "llama-3.1": {
        # Auto-detect: llama-3.1 header markers vary by Unsloth version, so let
        # train_on_responses_only() read them off the template instead.
        "instruction_part": None,
        "response_part": None,
    },
    "qwen-2.5": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "gemma-3": {
        "instruction_part": None,
        "response_part": None,
    },
}

# Base models the toolkit knows how to drive. `template` is the Unsloth
# get_chat_template() name; `ollama_family` is the registry tag whose known-good
# Ollama template you can copy via `ollama show --template <tag>`.
MODEL_PRESETS: dict[str, dict] = {
    # keeper (recommended): 12B, Apache-2.0, 128k context, native Tekken [INST]
    "nemo-instruct": {
        "model_name": "unsloth/Mistral-Nemo-Instruct-2407",
        # merge_base MUST be the SAME weights training froze (the Instruct model),
        # not the raw Base — merging Instruct-trained LoRA deltas into Base discards
        # all instruction tuning and inherits Base's untrained EOS (silent breakage).
        "merge_base": "mistralai/Mistral-Nemo-Instruct-2407",
        "template": "mistral",
        "ollama_family": "mistral-nemo",
    },
    # base variant — only for raw/Alpaca completion; its EOS is ~untrained
    "nemo-base": {
        "model_name": "unsloth/Mistral-Nemo-Base-2407",
        "merge_base": "mistralai/Mistral-Nemo-Base-2407",
        "template": "mistral",
        "ollama_family": "mistral-nemo",
    },
    # v0 plumbing run: small, fast, cheap (gated repo — accept license + login)
    "llama3.1-8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "merge_base": "meta-llama/Llama-3.1-8B-Instruct",
        "template": "llama-3.1",
        "ollama_family": "llama3.1",
    },
    # strong reasoning / good for the teaching angle
    "qwen2.5-14b": {
        "model_name": "unsloth/Qwen2.5-14B-Instruct",
        "merge_base": "Qwen/Qwen2.5-14B-Instruct",
        "template": "qwen-2.5",
        "ollama_family": "qwen2.5",
    },
    # end-of-month dual-GPU keeper (train via Axolotl FSDP2, not this script).
    # Uses ChatML to match axolotl_mistral_small_24b_fsdp2.yaml (chat_template: chatml);
    # export with `--template chatml --no-inherit` so the served template matches training.
    "mistral-small-24b": {
        "model_name": "unsloth/Mistral-Small-24B-Base-2501",
        "merge_base": "mistralai/Mistral-Small-24B-Base-2501",
        "template": "chatml",
        "ollama_family": "mistral-small",
    },
}

DEFAULT_MODEL = "nemo-instruct"

# Assistant-slop / "GPT-ism" phrases. Rows whose assistant turns lean on these
# pull the model toward the generic helpful-assistant persona ("Assistant Axis"
# drift), which is exactly the dry-tutor failure to avoid.
SLOP_PHRASES: tuple[str, ...] = (
    "as an ai",
    "as a language model",
    "i cannot fulfill",
    "i can't fulfill",
    "i'm just an ai",
    "i am just an ai",
    "i hope this helps",
    "let me know if you have any other questions",
    "is there anything else i can help",
    "i'm sorry, but i cannot",
    "i am sorry, but i cannot",
    "it's important to note that",
    "it is important to note that",
    "i don't have personal",
    "i do not have personal",
)

# In-character tutor wrapper so educational data installs TEACHING BEHAVIOUR as
# a mode of the character, not a context-free "helpful assistant".
TUTOR_SYSTEM_TEMPLATE = (
    "You are {char}, and you stay fully in character at all times. When the "
    "user wants to learn or understand something, teach it in your own voice: "
    "explain it clearly, give a short concrete example, then ask ONE question "
    "to check their understanding. Never drop character or announce that you "
    "are an AI or a language model."
)


def tutor_system_prompt(char: str = "the character") -> str:
    return TUTOR_SYSTEM_TEMPLATE.format(char=char)


# --- JSONL I/O ----------------------------------------------------------------

def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one dict per non-blank line; skip malformed lines rather than die."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write rows as JSONL; return the count written. Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


# --- message validation / normalization --------------------------------------

_VALID_ROLES = ("system", "user", "assistant")


def clean_messages(messages) -> list[dict]:
    """Coerce a messages list down to clean {role, content} dicts.

    Drops any extra keys (e.g. rotut's non-standard "reasoning" field, which
    makes apply_chat_template choke), stringifies content, and drops turns with
    an unknown role or empty content.
    """
    out: list[dict] = []
    if not isinstance(messages, list):
        return out
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in _VALID_ROLES:
            continue
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def messages_valid(messages) -> bool:
    """A record is trainable if it has >=1 assistant turn with real content."""
    msgs = messages if isinstance(messages, list) else []
    return any(m.get("role") == "assistant" and str(m.get("content", "")).strip()
               for m in msgs)


# ShareGPT {from,value} -> {role,content}. "from" spellings vary across mirrors,
# so map defensively and default unknown speakers to user.
ROLE_MAP: dict[str, str] = {
    "system": "system", "system_prompt": "system",
    "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
    "bot": "assistant", "model": "assistant", "char": "assistant",
}


def sharegpt_to_messages(conv, char: str = "the character",
                         user: str = "User") -> list[dict]:
    """Convert a ShareGPT conversation list into {role,content} messages.

    Substitutes {{char}}/{{user}} placeholders (LimaRP uses these) so the model
    learns real names, not the literal tokens.
    """
    out: list[dict] = []
    if not isinstance(conv, list):
        return out
    for t in conv:
        if not isinstance(t, dict):
            continue
        frm = str(t.get("from", "")).lower()
        value = t.get("value")
        if value is None:
            continue
        value = (str(value).replace("{{char}}", char).replace("{{user}}", user)
                 .strip())
        if not value:
            continue
        out.append({"role": ROLE_MAP.get(frm, "user"), "content": value})
    return out


# --- slop + dedup -------------------------------------------------------------

def assistant_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m.get("role") == "assistant")


def slop_hits(messages: list[dict], phrases: tuple[str, ...] = SLOP_PHRASES) -> int:
    """Count slop-phrase occurrences across a record's assistant turns."""
    hay = assistant_text(messages).lower()
    return sum(hay.count(p) for p in phrases)


def is_sloppy(messages: list[dict], max_hits: int = 0,
              phrases: tuple[str, ...] = SLOP_PHRASES) -> bool:
    """True if a record should be dropped for assistant-slop (default: any hit)."""
    return slop_hits(messages, phrases) > max_hits


def _render_for_hash(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}:{m['content']}" for m in messages)


def exact_key(messages: list[dict]) -> str:
    """Stable hash of the rendered conversation for exact-duplicate dropping."""
    return hashlib.sha1(_render_for_hash(messages).encode("utf-8")).hexdigest()


def dedup_exact(records: Iterable[dict], key: str = "messages") -> list[dict]:
    """Drop records whose rendered conversation hashes identically."""
    seen: set[str] = set()
    kept: list[dict] = []
    for r in records:
        h = exact_key(r.get(key, []))
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    return kept


def _shingles(text: str, k: int = 5) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def dedup_near(records: list[dict], threshold: float = 0.9, k: int = 5,
               key: str = "messages") -> list[dict]:
    """Pure-Python near-duplicate drop via Jaccard over word k-shingles.

    O(n^2); fine for tens of thousands of rows. For very large merges prefer the
    datasketch MinHash-LSH path in prepare_data.py. `threshold` ~0.9 matches the
    community RP-dataset fuzzy-dedup convention.
    """
    kept: list[dict] = []
    sigs: list[set[str]] = []
    for r in records:
        sig = _shingles(assistant_text(r.get(key, [])), k)
        if any(_jaccard(sig, s) >= threshold for s in sigs):
            continue
        kept.append(r)
        sigs.append(sig)
    return kept


# --- character + persona -> system prompt -------------------------------------

def _join(parts: Iterable[str]) -> str:
    return "\n\n".join(p for p in parts if p and p.strip())


def build_character_system_prompt(char: dict,
                                  persona: Optional[dict] = None) -> str:
    """Reconstruct a training system prompt from an app character (+ persona).

    Mirrors the spirit of app/context.py's system block without importing it:
    an explicit system_prompt override wins; otherwise compose from the card's
    name/description/personality/scenario and the optional structured profile.
    """
    name = (char.get("name") or "the character").strip()

    override = (char.get("system_prompt") or "").strip()
    if override:
        head = override
    else:
        bits = [f"You are {name}."]
        for field, label in (
            ("description", ""),
            ("personality", "Personality:"),
            ("scenario", "Scenario:"),
        ):
            val = (char.get(field) or "").strip()
            if val:
                bits.append(f"{label} {val}".strip())
        head = _join(bits)

    # Optional SimsChat-style structured profile fields, if the card has them.
    profile_bits = []
    for field, label in (
        ("age", "Age"), ("gender", "Gender"), ("career", "Career"),
        ("aspiration", "Aspiration"), ("traits", "Traits"),
        ("skills", "Skills"), ("speech_style", "Speech style"),
        ("relationships", "Relationships"), ("living", "Living situation"),
    ):
        val = (char.get(field) or "").strip()
        if val:
            profile_bits.append(f"{label}: {val}")
    profile = ("Profile:\n" + "\n".join(profile_bits)) if profile_bits else ""

    example = (char.get("example_dialogue") or "").strip()
    example_block = (f"Example dialogue:\n{example}") if example else ""

    persona_block = ""
    if persona:
        pname = (persona.get("name") or "the user").strip()
        pdesc = (persona.get("description") or "").strip()
        persona_block = (f"You are talking to {pname}."
                         + (f" {pdesc}" if pdesc else ""))

    return _join([head, profile, example_block, persona_block])
