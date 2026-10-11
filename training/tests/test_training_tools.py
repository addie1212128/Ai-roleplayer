"""Dependency-light tests for the training toolkit's pure logic.

Covers rp_common (templates, message cleaning, ShareGPT mapping, slop, dedup,
system-prompt build) and import_my_stories.convert_story (SFT + DPO extraction,
group-scene prefixing, filtering). No torch/unsloth/datasets — runs in CI.

Run:  python training/tests/test_training_tools.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent))   # training/ on path

import rp_common as rp                         # noqa: E402
import import_my_stories as ims                # noqa: E402

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ✓ {msg}")
    else:
        print(f"  ✗ {msg}")
        _failures.append(msg)


# --- rp_common ---------------------------------------------------------------

def test_clean_messages():
    print("clean_messages")
    msgs = rp.clean_messages([
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "hi", "reasoning": "DROP ME"},  # extra key stripped
        {"role": "user", "content": "  spaced  "},
        {"role": "assistant", "content": ""},        # empty dropped
        {"role": "narrator", "content": "nope"},      # bad role dropped
        "not-a-dict",
    ])
    check(len(msgs) == 3, "keeps 3 valid turns")
    check(all(set(m.keys()) == {"role", "content"} for m in msgs), "only role/content keys")
    check(msgs[1]["content"] == "hi", "assistant 'reasoning' key stripped, content kept")
    check(msgs[2]["content"] == "spaced", "content trimmed")


def test_messages_valid():
    print("messages_valid")
    check(rp.messages_valid([{"role": "assistant", "content": "x"}]), "true with an assistant turn")
    check(not rp.messages_valid([{"role": "user", "content": "x"}]), "false with no assistant turn")
    check(not rp.messages_valid([]), "false when empty")


def test_sharegpt():
    print("sharegpt_to_messages")
    conv = [
        {"from": "system", "value": "world"},
        {"from": "human", "value": "hi {{char}}"},
        {"from": "gpt", "value": "hello {{user}}"},
        {"from": "bot", "value": "again"},
    ]
    msgs = rp.sharegpt_to_messages(conv, char="Aria", user="Alex")
    check([m["role"] for m in msgs] == ["system", "user", "assistant", "assistant"],
          "from->role mapping (human/gpt/bot)")
    check(msgs[1]["content"] == "hi Aria", "{{char}} substituted")
    check(msgs[2]["content"] == "hello Alex", "{{user}} substituted")


def test_slop():
    print("slop")
    sloppy = [{"role": "assistant", "content": "As an AI, I cannot do that."}]
    clean = [{"role": "assistant", "content": "The dragon roared."}]
    check(rp.is_sloppy(sloppy), "flags assistant slop")
    check(not rp.is_sloppy(clean), "passes clean prose")
    check(rp.slop_hits([{"role": "user", "content": "as an ai"}]) == 0,
          "ignores slop in user turns")


def test_dedup():
    print("dedup")
    a = {"messages": [{"role": "assistant", "content": "the quick brown fox jumps over"}]}
    a2 = {"messages": [{"role": "assistant", "content": "the quick brown fox jumps over"}]}
    b = {"messages": [{"role": "assistant", "content": "a completely different sentence here"}]}
    check(len(rp.dedup_exact([a, a2, b])) == 2, "exact dedup drops the identical copy")
    near = {"messages": [{"role": "assistant", "content": "the quick brown fox jumps over now"}]}
    check(len(rp.dedup_near([a, near, b], threshold=0.6)) == 2, "near dedup drops a close variant")


def test_system_prompt():
    print("build_character_system_prompt")
    sp = rp.build_character_system_prompt(
        {"name": "Aria", "personality": "warm", "scenario": "a tavern", "traits": "kind"},
        {"name": "Alex", "description": "a traveler"})
    check("You are Aria." in sp, "names the character")
    check("warm" in sp and "tavern" in sp, "folds personality + scenario")
    check("Traits: kind" in sp, "includes structured profile field")
    check("Alex" in sp, "appends persona")
    override = rp.build_character_system_prompt(
        {"name": "Aria", "system_prompt": "CUSTOM OVERRIDE", "personality": "warm"})
    check("CUSTOM OVERRIDE" in override and "warm" not in override,
          "explicit system_prompt override wins")


# --- import_my_stories -------------------------------------------------------

CHARS = {"c1": {"id": "c1", "name": "Aria", "personality": "warm", "scenario": "a tavern"}}
PERSONAS = {"p1": {"id": "p1", "name": "Alex", "description": "a traveler"}}


def test_assistant_content_uses_swipe():
    print("_assistant_content uses the selected swipe")
    m = {"role": "assistant", "content": "STALE", "swipes": ["A", "B", "C"], "swipe_id": 1}
    check(ims._assistant_content(m, False) == "B", "reads swipes[swipe_id], not stale content")
    check(ims._assistant_content({"role": "assistant", "content": "only"}, False) == "only",
          "falls back to content when there are no swipes")
    mg = {"role": "assistant", "content": "x", "swipes": ["hi"], "swipe_id": 0, "name": "Vale"}
    check(ims._assistant_content(mg, True) == "Vale: hi", "group name-prefix applied to the swipe")


def test_convert_single():
    print("convert_story (single character)")
    story = {"id": "s1", "title": "T", "character_id": "c1", "persona_id": "p1", "messages": [
        {"role": "assistant", "content": "Welcome!", "swipes": ["Welcome!"], "swipe_id": 0},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there.",
         "swipes": ["Hello there.", "Hey!", "Yo."], "swipe_id": 0},
    ]}
    sft, dpo = ims.convert_story(story, CHARS, PERSONAS)
    check(len(sft) == 1, "one SFT record")
    m = sft[0]["messages"]
    check(m[0]["role"] == "system" and "Aria" in m[0]["content"], "system prompt first, from the card")
    check(m[-1]["content"] == "Hello there.", "uses the selected swipe as content")
    check(len(dpo) == 2, "two DPO pairs from the 3-swipe turn (chosen vs each alternate)")
    check(all(p["chosen"][0]["content"] == "Hello there." for p in dpo), "chosen = selected swipe")
    check({p["rejected"][0]["content"] for p in dpo} == {"Hey!", "Yo."}, "rejected = the alternates")
    pr = dpo[0]["prompt"]
    check(pr[-1]["role"] == "user", "DPO prompt ends on a user turn (template-valid)")
    check(not any("Welcome!" in t["content"] for t in pr),
          "leading assistant greeting dropped from the DPO prompt")


def test_normalize_prompt():
    print("_normalize_prompt (DPO prompts are template-valid)")
    sys_u = {"role": "system", "content": "s"}
    g = {"role": "assistant", "content": "greeting"}
    u = {"role": "user", "content": "u1"}
    a = {"role": "assistant", "content": "a1"}
    # leading greeting dropped, trailing assistant trimmed
    out = ims._normalize_prompt([sys_u, g, u, a])
    check(out == [sys_u, u], "drops leading greeting + trailing assistant")
    # consecutive assistants merged, then trailing trimmed -> ends on user
    out2 = ims._normalize_prompt([sys_u, u, a, {"role": "assistant", "content": "a2"}])
    check(out2 == [sys_u, u], "merges consecutive assistants then trims to the user")
    # alternating prompt ending on user is preserved
    keep = [u, a, {"role": "user", "content": "u2"}]
    check(ims._normalize_prompt(keep) == keep, "keeps a valid alternating prompt")
    # no user turn -> None
    check(ims._normalize_prompt([sys_u, g]) is None, "returns None when there is no user turn")


def test_convert_group():
    print("convert_story (group scene)")
    story = {"id": "s2", "character_id": "c1", "messages": [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "Hi, I'm Aria.", "name": "Aria",
         "swipes": ["Hi, I'm Aria."], "swipe_id": 0},
        {"role": "assistant", "content": "And I'm Vale.", "name": "Vale",
         "swipes": ["And I'm Vale.", "Vale here."], "swipe_id": 0},
    ]}
    sft, dpo = ims.convert_story(story, CHARS, PERSONAS)
    text = rp.assistant_text(sft[0]["messages"])
    check("Aria: Hi, I'm Aria." in text and "Vale: And I'm Vale." in text,
          "group assistant turns are name-prefixed")
    check(len(dpo) == 1 and dpo[0]["chosen"][0]["content"] == "Vale: And I'm Vale.",
          "group DPO chosen is name-prefixed")
    check(dpo[0]["rejected"][0]["content"] == "Vale: Vale here.", "group DPO rejected is name-prefixed")


def test_convert_filters():
    print("convert_story (filters)")
    story = {"id": "s3", "character_id": "c1", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "one"},
    ]}
    sft, _ = ims.convert_story(story, CHARS, PERSONAS, min_turns=3)
    check(sft == [], "min_turns filters out short stories")
    nochar = {"id": "s4", "character_name": "Mystery", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "boo"}]}
    sft2, _ = ims.convert_story(nochar, {}, {})
    check(len(sft2) == 1 and "Mystery" in sft2[0]["messages"][0]["content"],
          "falls back to character_name when the card is missing")


def main() -> int:
    for t in (test_clean_messages, test_messages_valid, test_sharegpt, test_slop,
              test_dedup, test_system_prompt, test_normalize_prompt,
              test_assistant_content_uses_swipe, test_convert_single,
              test_convert_group, test_convert_filters):
        t()
    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): " + "; ".join(_failures))
        return 1
    print("All training-toolkit tests passed. ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
