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
    return max(1, len(text) // 4)


def _msgs_tokens(msgs: list[dict]) -> int:
    # +4 per message approximates role/format overhead.
    return sum(est_tokens(m.get("content", "")) + 4 for m in msgs)


# ---------------------------------------------------------------------------
# macros
# ---------------------------------------------------------------------------

def substitute(text: str, char_name: str, user_name: str) -> str:
    if not text:
        return text or ""
    return (text
            .replace("{{char}}", char_name or "the character")
            .replace("{{user}}", user_name or "You")
            .replace("<BOT>", char_name or "the character")
            .replace("<USER>", user_name or "You"))


# ---------------------------------------------------------------------------
# reasoning-model <think> stripping
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove closed reasoning blocks so they never reach the RP log or history."""
    if not text:
        return text or ""
    cleaned = _THINK_RE.sub("", text)
    # An unclosed opener (stream cut mid-thought) — drop from it to the end.
    low = cleaned.lower()
    if "<think>" in low and "</think>" not in low:
        cleaned = cleaned[: low.index("<think>")]
    return cleaned.strip()


# ---------------------------------------------------------------------------
# default system-prompt template (used when a character sets no system_prompt)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM = """You are {{char}}, a real person within this collaborative fiction, never an AI, assistant, or language model. Never break character, add disclaimers or content warnings, or mention being an AI. If unsure, stay in character and keep the scene going.

[CHARACTER]
{{char}} — {description}
Personality: {personality}

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

    override = (character.get("system_prompt") or "").strip()
    if override:
        default = _DEFAULT_SYSTEM.format(
            description=desc or "(no description)",
            personality=personality or "(unspecified)",
            scenario=scenario or "(open scene)",
            persona_block=persona_block,
            pov_line=_pov_line(character.get("pov", "")),
        )
        block = override.replace("{{original}}", default)
    else:
        block = _DEFAULT_SYSTEM.format(
            description=desc or "(no description)",
            personality=personality or "(unspecified)",
            scenario=scenario or "(open scene)",
            persona_block=persona_block,
            pov_line=_pov_line(character.get("pov", "")),
        )
    return block


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
    if whole_words:
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

    scan_depth = int(lorebook.get("scanDepth", 4) or 4)
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
            depth_parts.append((int(e.get("depth", 4) or 4), content))
        else:
            before_parts.append(content)

    result["before"] = "\n\n".join(before_parts)
    result["at_depth"] = depth_parts
    return result


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

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
                    lorebook: Optional[dict], settings: dict) -> dict:
    """Build the messages list for Ollama. Returns {'messages', 'debug'}."""
    char_name = character.get("name", "the character")
    user_name = (persona or {}).get("name") or settings.get("user_name") or "You"

    num_ctx = int(settings.get("num_ctx") or 16384)
    num_predict = int(settings.get("num_predict") or 512)
    reserve = 256  # headroom for template/role tokens we can't see
    budget = max(512, num_ctx - num_predict - reserve)

    # --- fixed (stable-ish) head ---
    system_block = substitute(build_system_block(character, persona), char_name, user_name)

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
        depth_injections.append((int(an.get("depth", 4) or 4),
                                 {"role": an.get("role", "system"), "content": f"[Author's note: {an_text}]"}))

    phi = substitute((character.get("post_history_instructions") or "").strip(), char_name, user_name)
    if phi:
        depth_injections.append((0, {"role": "system", "content": phi}))

    reminder = (f"[Stay in character as {char_name}. Write only {char_name} and the world, "
                f"never {user_name}. End on an opening and stop.]")
    depth_injections.append((0, {"role": "system", "content": reminder}))

    # --- token accounting for fixed pieces ---
    head: list[dict] = [{"role": "system", "content": system_block}]
    if lore.get("before"):
        head.append({"role": "system", "content": lore["before"]})
    if summaries:
        head.append({"role": "system", "content": summaries})

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
