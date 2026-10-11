"""Turn YOUR AI Roleplayer stories into training data.

This is the highest-value, most personal slice of the dataset: a few hundred
exchanges in your own taste teach the model more than any public set.

It reads the app's local `data/` directory (stories + characters + personas)
and emits two files:

  * SFT  (--out-sft) : {"messages": [...]} — full conversations, with the
    character + persona rebuilt into the system prompt so the model learns to
    follow a card.
  * DPO  (--out-dpo) : {"prompt","chosen","rejected"} — FREE preference pairs
    mined from your swipes: the swipe you kept is `chosen`, each alternate you
    swiped past is `rejected`. Feed this to training/train_dpo.py later.

Pure stdlib — no heavy deps, so it runs in any Python 3.9+.

Usage:
    python training/import_my_stories.py                 # uses ../data, writes ./training/data/
    python training/import_my_stories.py --data-dir /path/to/data --min-turns 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `import rp_common` from any cwd
import rp_common as rp  # noqa: E402


def _load_dir(d: Path) -> dict[str, dict]:
    """Load every *.json in a store dir into {id: doc} (id falls back to stem)."""
    out: dict[str, dict] = {}
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(doc, dict):
            out[str(doc.get("id") or f.stem)] = doc
    return out


def _is_group(messages: list[dict]) -> bool:
    """A story is a group scene if >1 distinct assistant speaker name appears."""
    names = {str(m.get("name")).strip() for m in messages
             if m.get("role") == "assistant" and m.get("name")}
    return len(names) > 1


def _assistant_content(m: dict, group: bool) -> str:
    """Selected text for an assistant turn, name-prefixed in group scenes."""
    text = str(m.get("content", "")).strip()
    if group and m.get("name"):
        return f"{str(m['name']).strip()}: {text}"
    return text


def _swipe_texts(m: dict, group: bool) -> tuple[str, list[str]]:
    """(chosen, [rejected...]) from an assistant turn's swipes; name-prefixed in groups."""
    swipes = m.get("swipes")
    if not isinstance(swipes, list) or len(swipes) < 2:
        return "", []
    sid = m.get("swipe_id", 0)
    if not isinstance(sid, int) or not (0 <= sid < len(swipes)):
        sid = 0
    prefix = f"{str(m['name']).strip()}: " if (group and m.get("name")) else ""
    chosen = prefix + str(swipes[sid]).strip()
    rejected = []
    seen = {chosen}
    for j, sw in enumerate(swipes):
        if j == sid:
            continue
        txt = prefix + str(sw).strip()
        if txt and txt not in seen:   # skip blanks and dups of chosen/each other
            rejected.append(txt)
            seen.add(txt)
    return chosen, rejected


def convert_story(story: dict, characters: dict, personas: dict,
                  min_turns: int = 1) -> tuple[list[dict], list[dict]]:
    """Return (sft_records, dpo_records) for one story."""
    raw = story.get("messages")
    if not isinstance(raw, list) or not raw:
        return [], []

    char = characters.get(str(story.get("character_id")))
    if not char:
        char = {"name": story.get("character_name") or "Character"}
    persona = personas.get(str(story.get("persona_id"))) if story.get("persona_id") else None
    system = rp.build_character_system_prompt(char, persona)
    group = _is_group(raw)
    char_name = (char.get("name") or "Character")

    convo: list[dict] = [{"role": "system", "content": system}] if system else []
    dpo: list[dict] = []
    assistant_turns = 0

    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            # DPO pair(s) from this turn's swipes, using the conversation SO FAR
            # as the prompt. Needs at least one user turn in the prompt.
            chosen, rejected = _swipe_texts(m, group)
            if chosen and rejected and any(t["role"] == "user" for t in convo):
                prompt_msgs = [dict(t) for t in convo]
                for rej in rejected:
                    dpo.append({
                        "prompt": prompt_msgs,
                        "chosen": [{"role": "assistant", "content": chosen}],
                        "rejected": [{"role": "assistant", "content": rej}],
                        "source": "my_stories_swipes",
                        "story_id": story.get("id"),
                    })
            content = _assistant_content(m, group)
            if content:
                convo.append({"role": "assistant", "content": content})
                assistant_turns += 1
        elif role in ("user", "system"):
            content = str(m.get("content", "")).strip()
            if content:
                convo.append({"role": role, "content": content})

    convo = rp.clean_messages(convo)
    sft: list[dict] = []
    if rp.messages_valid(convo) and assistant_turns >= min_turns:
        sft.append({
            "messages": convo,
            "source": "my_stories",
            "capability": "roleplay",
            "story_id": story.get("id"),
            "title": story.get("title", "Untitled"),
            "character": char_name,
        })
    return sft, dpo


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Convert AI Roleplayer stories into SFT + DPO data.")
    ap.add_argument("--data-dir", default=str(repo_root / "data"),
                    help="App data directory (contains stories/, characters/, personas/).")
    ap.add_argument("--out-sft", default=str(repo_root / "training" / "data" / "my_stories_sft.jsonl"))
    ap.add_argument("--out-dpo", default=str(repo_root / "training" / "data" / "my_stories_dpo.jsonl"))
    ap.add_argument("--min-turns", type=int, default=1,
                    help="Skip stories with fewer than this many assistant turns.")
    args = ap.parse_args()

    data = Path(args.data_dir)
    stories = _load_dir(data / "stories")
    characters = _load_dir(data / "characters")
    personas = _load_dir(data / "personas")

    if not stories:
        print(f"No stories found under {data / 'stories'}.")
        print("Play a few stories in the app first, or point --data-dir at your data folder.")
        return 1

    all_sft: list[dict] = []
    all_dpo: list[dict] = []
    for story in stories.values():
        sft, dpo = convert_story(story, characters, personas, args.min_turns)
        all_sft.extend(sft)
        all_dpo.extend(dpo)

    # SFT is small and personal — exact-dedup only (keep near-dup variety).
    all_sft = rp.dedup_exact(all_sft)

    n_sft = rp.write_jsonl(args.out_sft, all_sft)
    n_dpo = rp.write_jsonl(args.out_dpo, all_dpo)

    print(f"Read {len(stories)} stories, {len(characters)} characters, {len(personas)} personas.")
    print(f"  SFT conversations : {n_sft:>5}  -> {args.out_sft}")
    print(f"  DPO swipe pairs   : {n_dpo:>5}  -> {args.out_dpo}")
    if n_dpo == 0:
        print("  (No DPO pairs yet — those come from swiping to alternate replies in the app.)")
    print("\nNext: fold my_stories_sft.jsonl into the blend via "
          "`python training/prepare_data.py --my-stories " + args.out_sft + "`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
