"""Server-side prompt assembly — the backbone of the roleplay engine.

`assemble_prompt()` turns a story + character + persona + lorebook + settings
into the exact `messages` list sent to Ollama, in a fixed, deliberate order
with token budgeting done BEFORE depth-injected pieces are spliced in. This is
the single biggest difference between a real RP frontend and a flat chat loop.

Assembly order (stable → volatile):
  1. system block  (instruct framing + character card + persona + how-to-play)
  2. lorebook "before" + constant entries
  3. [Story so far] rolling summaries
  4. example dialogue (dropped first when budget is tight, unless pinned)
  5. chat history (newest kept, filling the remaining budget)
  6. depth-injected, AFTER budgeting: @depth lorebook, author's note,
     post-history instructions, and a final in-character reminder.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# token estimation (pre-call). Ground-truth counts come back from Ollama as
# prompt_eval_count and are used for the UI meter; budgeting uses this estimate.
# ---------------------------------------------------------------------------

def est_tokens(text: str) -> int:
    if not text:
        return 0
    # ASCII averages ~4 chars/token; CJK and other wide scripts are closer to
    # ~1 token per character, so count them heavier to avoid undercounting
    # (an undercount would silently overflow num_ctx).
    ascii_chars = sum(1 for c in text if ord(c) < 0x80)
    wide_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + wide_chars)


def _msgs_tokens(msgs: list[dict]) -> int:
    # +4 per message approximates role/format overhead.
    return sum(est_tokens(m.get("content", "")) + 4 for m in msgs)


# ---------------------------------------------------------------------------
# macros
# ---------------------------------------------------------------------------

_MACRO_RE = re.compile(r"\{\{char\}\}|\{\{user\}\}|<BOT>|<USER>")


def substitute(text: str, char_name: str, user_name: str) -> str:
    if not text:
        return text or ""
    # One pass over the original text so a macro appearing inside a replacement
    # value (e.g. a character literally named "{{user}}") is not re-substituted.
    c = char_name or "the character"
    u = user_name or "You"
    table = {"{{char}}": c, "<BOT>": c, "{{user}}": u, "<USER>": u}
    return _MACRO_RE.sub(lambda m: table[m.group(0)], text)


# ---------------------------------------------------------------------------
# reasoning-model <think> stripping
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove closed reasoning blocks so they never reach the RP log or history."""
    if not text:
        return text or ""
    cleaned = _THINK_RE.sub("", text)
    # Any opener still present (either spelling) is unclosed — a stream cut
    # mid-thought. Drop from it to the end.
    m = re.search(r"<think(?:ing)?>", cleaned, re.IGNORECASE)
    if m:
        cleaned = cleaned[: m.start()]
    return cleaned.strip()


# ---------------------------------------------------------------------------
# default system-prompt template (used when a character sets no system_prompt)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM = """You are {{char}}, a real person within this collaborative fiction, never an AI, assistant, or language model. Never break character, add disclaimers or content warnings, or mention being an AI. If unsure, stay in character and keep the scene going.

[CHARACTER]
{{char}} — {description}
Personality: {personality}{profile_block}

[SCENE]
{scenario}
{persona_block}[HOW TO PLAY]
You write only for {{char}}, other NPCs, and the surrounding world.{pov_line} Treat {{user}} as an autonomous person whose thoughts, speech, decisions, and actions are unknown until they state them; describe {{user}} only through what {{char}} can actually see or hear. Drive the story forward through {{char}}'s actions, the NPCs, and the environment. When the moment calls for {{user}} to react, end on a clear opening — an action, a question, an expectant beat — and stop. Never decide {{user}}'s choices, words, movements, or feelings, and never treat silence as agreement.

[PROSE]
Wrap actions and narration in *asterisks*; put spoken words in "quotation marks". Match the length and style of the opening message. Vary your sentence openings. Avoid tired clichés like "a shiver down the spine", "barely above a whisper", "eyes sparkling with mischief", and the "not X, but Y" construction."""


def _pov_line(pov: str) -> str:
    if pov == "first":
        return " Write in first person as {{char}}."
    if pov == "third":
        return " Maintain a third-person-limited perspective anchored to {{char}}."
    return ""  # empty: let the greeting set the voice (the PROSE line says to match it)


# Structured, three-tier profile fields (SimsChat-style). Only non-empty ones
# are rendered, so characters stay simple unless the author fills them in.
_PROFILE_FIELDS = [
    ("Age", "age"), ("Gender", "gender"), ("Career", "career"),
    ("Aspiration", "aspiration"), ("Traits", "traits"), ("Skills", "skills"),
    ("Speech style", "speech_style"), ("Relationships", "relationships"),
    ("Living situation", "living"),
]


def _profile_block(character: dict) -> str:
    lines = [f"{label}: {str(character.get(key)).strip()}"
             for label, key in _PROFILE_FIELDS
             if character.get(key) and str(character.get(key)).strip()]
    return ("\n" + "\n".join(lines)) if lines else ""


def build_system_block(character: dict, persona: Optional[dict]) -> str:
    desc = (character.get("description") or "").strip()
    personality = (character.get("personality") or "").strip()
    scenario = (character.get("scenario") or "").strip()
    persona_desc = ((persona or {}).get("description") or "").strip()

    persona_block = ""
    if persona_desc:
        persona_block = f"\n[WHO {{{{user}}}} IS]\n{persona_desc}\n\n"
    else:
        persona_block = "\n"

    fields = {
        "description": desc or "(no description)",
        "personality": personality or "(unspecified)",
        "scenario": scenario or "(open scene)",
        "persona_block": persona_block,
        "pov_line": _pov_line(character.get("pov", "")),
        "profile_block": _profile_block(character),
    }
    # Single-pass field substitution. We deliberately do NOT use str.format()
    # here: the template also carries {{char}}/{{user}} macros, and .format()
    # would collapse those to {char}/{user}, which substitute() (double-brace
    # only) would then never resolve. A single re.sub over the exact field
    # tokens leaves the macros intact for substitute() to handle later, and
    # never re-scans the inserted values.
    default = re.sub(
        r"\{(description|personality|scenario|persona_block|pov_line|profile_block)\}",
        lambda m: fields[m.group(1)],
        _DEFAULT_SYSTEM,
    )
    override = (character.get("system_prompt") or "").strip()
    if override:
        return override.replace("{{original}}", default)
    return default


# ---------------------------------------------------------------------------
# example dialogue (mes_example) parsing
# ---------------------------------------------------------------------------

def parse_examples(mes_example: str, char_name: str, user_name: str) -> list[dict]:
    """Parse <START>-separated example blocks into user/assistant message dicts."""
    if not mes_example or not mes_example.strip():
        return []
    text = substitute(mes_example, char_name, user_name)
    blocks = re.split(r"<START>", text, flags=re.IGNORECASE)
    msgs: list[dict] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Split on "{{user}}:"/"{{char}}:" style labels if present.
        lines = block.split("\n")
        current_role = None
        buf: list[str] = []

        def flush():
            if current_role and buf:
                msgs.append({"role": current_role, "content": "\n".join(buf).strip()})

        for line in lines:
            m = re.match(rf"^\s*({re.escape(user_name)}|{re.escape(char_name)})\s*:\s*(.*)$", line)
            if m:
                flush()
                buf = []
                current_role = "user" if m.group(1) == user_name else "assistant"
                if m.group(2):
                    buf.append(m.group(2))
            else:
                buf.append(line)
        flush()
        if current_role is None and block:
            # Unlabeled block — treat as a single assistant example.
            msgs.append({"role": "assistant", "content": block})
    return msgs


# ---------------------------------------------------------------------------
# lorebook / world info scanning
# ---------------------------------------------------------------------------

def _key_matches(text_lower: str, key: str, whole_words: bool) -> bool:
    key = (key or "").strip().lower()
    if not key:
        return False
    # Whole-word matching only makes sense for ASCII word-ish keys. For CJK
    # (no spaces between words) or keys bounded by punctuation, \b never fires,
    # so fall back to substring matching.
    if whole_words and key.isascii() and key[0].isalnum() and key[-1].isalnum():
        return re.search(r"\b" + re.escape(key) + r"\b", text_lower) is not None
    return key in text_lower


def _entry_fires(entry: dict, text_lower: str) -> bool:
    if not entry.get("enabled", True):
        return False
    if entry.get("constant"):
        return True
    whole = entry.get("matchWholeWords", True)
    keys = entry.get("keys", []) or []
    primary = any(_key_matches(text_lower, k, whole) for k in keys)
    if not primary:
        return False
    if entry.get("selective") and (entry.get("keysecondary") or []):
        sec = entry["keysecondary"]
        logic = (entry.get("selectiveLogic") or "AND_ANY").upper()
        sec_any = any(_key_matches(text_lower, k, whole) for k in sec)
        sec_all = all(_key_matches(text_lower, k, whole) for k in sec)
        if logic == "AND_ANY":
            return sec_any
        if logic == "AND_ALL":
            return sec_all
        if logic == "NOT_ANY":
            return not sec_any
        if logic == "NOT_ALL":
            return not sec_all
    return True


def scan_lorebook(lorebook: Optional[dict], recent_msgs: list[dict], budget_tokens: int) -> dict:
    """Return {'before': str, 'at_depth': [(depth, content), ...], 'fired': [names]}.

    Scans the last `scanDepth` messages for keys; fires matching + constant
    entries, sorts by descending `order`, and includes within `budget_tokens`,
    dropping lowest-order first. Logs what overflowed in 'dropped'.
    """
    result = {"before": "", "at_depth": [], "fired": [], "dropped": []}
    if not lorebook:
        return result
    entries = lorebook.get("entries", []) or []
    if not entries:
        return result

    sd = lorebook.get("scanDepth")
    scan_depth = 4 if sd is None else int(sd)
    scanned = recent_msgs[-scan_depth:] if scan_depth > 0 else recent_msgs
    text_lower = "\n".join(m.get("content", "") for m in scanned).lower()

    fired = [e for e in entries if _entry_fires(e, text_lower)]
    # Higher order = more influence = kept first when budget is tight.
    fired.sort(key=lambda e: e.get("order", 100), reverse=True)

    before_parts: list[str] = []
    depth_parts: list[tuple[int, str]] = []
    used = 0
    for e in fired:
        content = (e.get("content") or "").strip()
        if not content:
            continue
        cost = est_tokens(content)
        if used + cost > budget_tokens:
            result["dropped"].append(e.get("name") or content[:30])
            continue
        used += cost
        result["fired"].append(e.get("name") or content[:30])
        pos = (e.get("position") or "before").lower()
        if pos in ("after", "at_depth", "after_char"):
            ed = e.get("depth")
            depth_parts.append((4 if ed is None else int(ed), content))
        else:
            before_parts.append(content)

    result["before"] = "\n\n".join(before_parts)
    result["at_depth"] = depth_parts
    return result


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def build_group_character(active: dict, members: list[dict]) -> dict:
    """Return the active character augmented with a group-scene instruction so
    it only ever speaks for itself."""
    names = [m.get("name", "?") for m in members]
    others = [n for n in names if n != active.get("name")]
    roster = (
        f"\n\n[GROUP SCENE] Characters present: {', '.join(names)}. "
        f"You are {active.get('name')}. Write ONLY {active.get('name')}'s dialogue and actions; "
        f"never speak or act for {', '.join(others) or 'the others'} or {{{{user}}}}. "
        f"Do not prefix your reply with a name."
    )
    g = dict(active)
    g["system_prompt"] = (active.get("system_prompt") or "{{original}}") + roster
    return g


def label_group_history(messages: list[dict]) -> list[dict]:
    """Prefix each character's past message with their name so the model can
    tell the speakers apart in a group scene."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("name"):
            out.append({**m, "content": f"{m['name']}: {m.get('content', '')}"})
        else:
            out.append(m)
    return out


def summaries_block(summaries: list[dict]) -> str:
    if not summaries:
        return ""
    lines = []
    for s in summaries:
        t = (s.get("text") or "").strip()
        if t:
            lines.append(t)
    if not lines:
        return ""
    return "[Story so far]\n" + "\n".join(lines)


SUMMARIZE_PROMPT = (
    "You are a story archivist. Summarize the roleplay exchange below into 3-5 "
    "concise, past-tense factual bullet points: what happened, decisions made, "
    "promises or threats, information revealed, and any change in location, time, "
    "mood, or relationships. Record only what is stated. No commentary, no preamble."
)


# ---------------------------------------------------------------------------
# depth splicing
# ---------------------------------------------------------------------------

def _splice_depth(history: list[dict], injections: list[tuple[int, dict]]) -> list[dict]:
    """Insert (depth, msg) items into history. depth 0 = after the last message,
    depth N = before the most recent N messages. Deeper items placed first so
    shallower ones end up closer to the end."""
    if not injections:
        return history
    n = len(history)
    # Bucket by insert position = n - depth (clamped).
    buckets: dict[int, list[dict]] = {}
    for depth, msg in sorted(injections, key=lambda x: x[0], reverse=True):
        pos = max(0, min(n, n - depth))
        buckets.setdefault(pos, []).append(msg)
    out: list[dict] = []
    for i in range(n + 1):
        if i in buckets:
            out.extend(buckets[i])
        if i < n:
            out.append(history[i])
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def assemble_prompt(story: dict, character: dict, persona: Optional[dict],
                    lorebook: Optional[dict], settings: dict, rag_block: str = "") -> dict:
    """Build the messages list for Ollama. Returns {'messages', 'debug'}."""
    char_name = character.get("name", "the character")
    user_name = (persona or {}).get("name") or settings.get("user_name") or "You"

    num_ctx = int(settings.get("num_ctx") or 16384)
    num_predict = int(settings.get("num_predict") or 512)
    reserve = 256  # headroom for template/role tokens we can't see
    budget = max(512, num_ctx - num_predict - reserve)

    # --- fixed (stable-ish) head ---
    system_block = substitute(build_system_block(character, persona), char_name, user_name)
    if settings.get("inner_monologue"):
        system_block += (
            f"\n\n[INNER MONOLOGUE] Reveal {char_name}'s private thoughts in double "
            f"parentheses, e.g. ((I shouldn't trust this stranger.)) — brief, first-person, "
            f"known only to {char_name}, and kept separate from spoken lines and *actions*."
        )
    scene = substitute((story.get("scene") or "").strip(), char_name, user_name)

    history_all = [
        {"role": m["role"], "content": strip_think(substitute(m.get("content", ""), char_name, user_name))}
        for m in story.get("messages", [])
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]

    lore_budget = int((lorebook or {}).get("tokenBudget") or (num_ctx * 0.25))
    lore = scan_lorebook(lorebook, history_all, lore_budget)

    summaries = summaries_block(story.get("summaries", []))
    examples = parse_examples(character.get("example_dialogue", ""), char_name, user_name)
    pin_examples = bool(settings.get("pin_examples", False))

    # --- depth-injected, built now but spliced after budgeting ---
    depth_injections: list[tuple[int, dict]] = []
    for depth, content in lore.get("at_depth", []):
        depth_injections.append((depth, {"role": "system", "content": content}))

    an = story.get("authors_note") or {}
    an_text = substitute((an.get("text") or "").strip(), char_name, user_name)
    if an_text:
        an_d = an.get("depth")
        depth_injections.append((4 if an_d is None else int(an_d),
                                 {"role": an.get("role", "system"), "content": f"[Author's note: {an_text}]"}))

    mood = (story.get("mood") or "").strip()
    if mood:
        depth_injections.append((3, {"role": "system", "content": f"[Right now, {char_name} is feeling {mood}.]"}))

    phi = substitute((character.get("post_history_instructions") or "").strip(), char_name, user_name)
    if phi:
        depth_injections.append((0, {"role": "system", "content": phi}))

    reminder = (f"[Stay in character as {char_name}. Write only {char_name} and the world, "
                f"never {user_name}. End on an opening and stop.]")
    depth_injections.append((0, {"role": "system", "content": reminder}))

    # --- token accounting for fixed pieces ---
    head: list[dict] = [{"role": "system", "content": system_block}]
    if scene:
        head.append({"role": "system", "content": "[Current scene]\n" + scene})
    if lore.get("before"):
        head.append({"role": "system", "content": lore["before"]})
    if summaries:
        head.append({"role": "system", "content": summaries})
    if rag_block:
        head.append({"role": "system", "content": rag_block})

    fixed_tokens = _msgs_tokens(head) + _msgs_tokens([m for _, m in depth_injections])
    example_tokens = _msgs_tokens(examples)

    dropped = {"examples": False, "history_trimmed": 0, "lore_dropped": lore.get("dropped", [])}

    # Decide whether examples fit.
    include_examples = bool(examples)
    avail_for_history = budget - fixed_tokens
    if include_examples and (pin_examples or example_tokens <= avail_for_history * 0.4):
        avail_for_history -= example_tokens
    else:
        if examples:
            dropped["examples"] = True
        include_examples = False

    # Fill history newest-first within the remaining budget.
    kept_rev: list[dict] = []
    used = 0
    for m in reversed(history_all):
        c = _msgs_tokens([m])
        if used + c > avail_for_history and kept_rev:
            dropped["history_trimmed"] = len(history_all) - len(kept_rev)
            break
        used += c
        kept_rev.append(m)
    history = list(reversed(kept_rev))

    # --- assemble ---
    messages: list[dict] = list(head)
    if include_examples:
        messages.extend(examples)
    messages.extend(_splice_depth(history, depth_injections))

    debug = {
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "budget_tokens": budget,
        "estimated_prompt_tokens": _msgs_tokens(messages),
        "fixed_tokens": fixed_tokens,
        "example_tokens": example_tokens if include_examples else 0,
        "history_messages_kept": len(history),
        "history_messages_total": len(history_all),
        "lore_fired": lore.get("fired", []),
        "dropped": dropped,
        "sections": [m["role"] for m in messages],
    }
    return {"messages": messages, "debug": debug}
