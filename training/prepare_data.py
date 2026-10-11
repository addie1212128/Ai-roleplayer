"""Build the blended SFT training set for a roleplay + teaching model.

Downloads and formats public datasets into the canonical {"messages": [...]}
record, keeps ROLEPLAY DOMINANT, folds in your own stories, scrubs slop,
deduplicates, and writes one train.jsonl (+ optional holdout) that
train_sft.py consumes directly.

Default blend (verified Oct-2026 guidance — roleplay must dominate; cap any
single source; audit by tokens, not just rows):
    roleplay  ~77% : PIPPA (deduped, length-filtered) + LimaRP + CoSER + YOURS
    general   ~15% : databricks-dolly-15k (open/general QA, brainstorm, summary)
    tutor     ~8%  : breitburg/rotut (Socratic) + garage-bAInd/Open-Platypus

All public datasets here are gated Not-For-All-Audiences and/or need a license
accept — run `huggingface-cli login` first. See training/README.md for sources,
licenses, and the "why" behind each choice.

Usage:
    pip install -r training/requirements-data.txt
    huggingface-cli login
    python training/import_my_stories.py          # (optional but recommended) first
    python training/prepare_data.py --my-stories training/data/my_stories_sft.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rp_common as rp  # noqa: E402

# ~~~ per-source converters (pure; operate on one HF row) ~~~~~~~~~~~~~~~~~~~~~~~


def pippa_to_messages(row: dict) -> list[dict]:
    name = (row.get("bot_name") or "the character").strip()
    system = f"You are {name}."
    if row.get("bot_description"):
        system += " " + str(row["bot_description"]).strip()
    if row.get("bot_definitions"):
        # Character.AI-formatted example dialogue; left as-is here (clean further
        # for production — see the PIPPA paper, Appendix A).
        system += "\n\n" + str(row["bot_definitions"]).strip()
    msgs = [{"role": "system", "content": system}]
    greeting = str(row.get("bot_greeting") or "").strip()
    convo = row.get("conversation") or []
    first_is_greeting = bool(convo) and (convo[0].get("is_human") is False) \
        and str(convo[0].get("message", "")).strip() == greeting
    if greeting and not first_is_greeting:
        msgs.append({"role": "assistant", "content": greeting})
    for turn in convo:
        role = "user" if turn.get("is_human") else "assistant"
        msgs.append({"role": role, "content": str(turn.get("message", "")).strip()})
    return msgs


def avg_bot_chars(row: dict) -> float:
    bot = [str(t.get("message", "")) for t in (row.get("conversation") or [])
           if not t.get("is_human")]
    return (sum(len(x) for x in bot) / len(bot)) if bot else 0.0


def dolly_to_messages(row: dict, char: str) -> list[dict]:
    user = str(row.get("instruction", "")).strip()
    ctx = str(row.get("context", "")).strip()
    if ctx:
        user = f"{user}\n\n{ctx}"
    return [
        {"role": "system", "content": rp.tutor_system_prompt(char)},
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(row.get("response", "")).strip()},
    ]


def platypus_to_messages(row: dict, char: str) -> list[dict]:
    user = str(row.get("instruction", "")).strip()
    inp = str(row.get("input", "")).strip()   # Open-Platypus's optional context col is 'input'
    if inp:
        user = f"{user}\n\n{inp}"
    return [
        {"role": "system", "content": rp.tutor_system_prompt(char)},
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(row.get("output", "")).strip()},
    ]


def rotut_to_messages(row: dict, char: str) -> list[dict]:
    # rotut assistant turns carry a non-standard 'reasoning' key -> clean_messages
    # strips it. Prepend the in-character tutor system prompt if none present.
    msgs = rp.clean_messages(row.get("messages") or [])
    if not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": rp.tutor_system_prompt(char)}] + msgs
    return msgs


# Open-Platypus sub-sources that are non-commercial; dropped for a clean mix.
_PLATYPUS_DROP = {"scienceqa", "reclor"}


# ~~~ loading / subsampling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _records_from(rows, convert, source: str, capability: str) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        msgs = rp.clean_messages(convert(row))
        if rp.messages_valid(msgs):
            out.append({"messages": msgs, "source": source, "capability": capability})
    return out


def _subsample(ds, cap: int | None, seed: int):
    if cap is None:
        return ds
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(cap, len(ds))))


def build(args) -> int:
    # Heavy deps imported here so the module compiles / unit-tests without them.
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Missing deps. Run:  pip install -r training/requirements-data.txt")
        return 1

    char = args.char_name
    seed = args.seed
    rp_records: list[dict] = []
    learn_records: list[dict] = []

    # --- roleplay core ---
    if not args.no_pippa:
        print("• PIPPA (pippa_deduped.jsonl via json builder — bypasses the removed loading script)")
        path = hf_hub_download(repo_id="PygmalionAI/PIPPA",
                               filename="pippa_deduped.jsonl", repo_type="dataset")
        ds = load_dataset("json", data_files=path, split="train")
        before = len(ds)
        ds = ds.filter(lambda r: avg_bot_chars(r) >= args.min_bot_chars)  # fix short-reply skew
        ds = _subsample(ds, args.max_pippa, seed)
        print(f"    {before} convos -> {len(ds)} after length-filter + cap")
        rp_records += _records_from(ds, pippa_to_messages, "pippa", "roleplay")

    if not args.no_limarp:
        print("• LimaRP (agentlans/lemonilia-LimaRP ShareGPT mirror)")
        ds = _subsample(load_dataset("agentlans/lemonilia-LimaRP", split="train"),
                        args.max_limarp, seed)
        rp_records += _records_from(
            ds, lambda r: rp.sharegpt_to_messages(
                r.get("conversations") or r.get("conversation"), char, "User"),
            "limarp", "roleplay")

    if not args.no_coser:
        print("• CoSER (Neph0s/CoSER novel-style; multi-character rows w/o an assistant turn are skipped)")
        ds = _subsample(load_dataset("Neph0s/CoSER",
                                     data_files="train/sft_conversations_sharegpt.json",
                                     split="train"), args.max_coser, seed)
        rp_records += _records_from(
            ds, lambda r: rp.sharegpt_to_messages(
                r.get("conversations") or r.get("conversation"), char, "User"),
            "coser", "roleplay")

    if args.my_stories:
        p = Path(args.my_stories)
        if p.exists():
            mine = [{"messages": rec.get("messages", []), "source": "my_stories",
                     "capability": "roleplay"}
                    for rec in rp.read_jsonl(p) if rp.messages_valid(rec.get("messages", []))]
            rp_records += mine
            print(f"• Your stories: {len(mine)} conversations from {p}")
        else:
            print(f"• Your stories: {p} not found — run import_my_stories.py first (skipping)")

    # --- learning slice (kept a clear minority) ---
    if not args.no_dolly:
        print("• Dolly-15k (open/general QA, brainstorm, summary — wrapped in-character)")
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        ds = ds.filter(lambda r: r.get("category") in
                       {"open_qa", "general_qa", "brainstorming", "summarization"})
        ds = _subsample(ds, args.max_dolly, seed)
        learn_records += _records_from(ds, lambda r: dolly_to_messages(r, char),
                                       "dolly15k", "instruction_general")

    if not args.no_rotut:
        print("• rotut (Socratic tutor)")
        ds = _subsample(load_dataset("breitburg/rotut", split="train"),
                        args.max_rotut, seed)
        learn_records += _records_from(ds, lambda r: rotut_to_messages(r, char),
                                       "rotut", "tutor_socratic")

    if not args.no_platypus:
        print("• Open-Platypus (STEM/logic; non-commercial sub-sources dropped)")
        ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
        ds = ds.filter(lambda r: str(r.get("data_source", "")).lower() not in _PLATYPUS_DROP)
        ds = _subsample(ds, args.max_platypus, seed)
        learn_records += _records_from(ds, lambda r: platypus_to_messages(r, char),
                                       "open-platypus", "stem_reasoning")

    records = rp_records + learn_records
    if not records:
        print("No records built — did you disable every source?")
        return 1

    # --- clean: slop filter, then exact + near dedup ---
    n0 = len(records)
    if not args.keep_slop:
        records = [r for r in records if not rp.is_sloppy(r["messages"], args.slop_max)]
    records = rp.dedup_exact(records)
    if not args.no_dedup:
        if len(records) <= args.near_dedup_cap:
            records = rp.dedup_near(records, threshold=args.near_threshold)
        else:
            records = _minhash_dedup(records, args.near_threshold)
    print(f"\nCleaned {n0} -> {len(records)} records (slop filter + dedup)")

    # --- shuffle deterministically, split, write ---
    _seeded_shuffle(records, seed)
    holdout_n = int(len(records) * args.holdout)
    holdout, train = records[:holdout_n], records[holdout_n:]

    n_train = rp.write_jsonl(args.out, train)
    print(f"\nWrote {n_train} train records -> {args.out}")
    if holdout_n:
        n_hold = rp.write_jsonl(args.holdout_out, holdout)
        print(f"Wrote {n_hold} holdout records -> {args.holdout_out}")

    _report_balance(train)
    return 0


def _minhash_dedup(records: list[dict], threshold: float) -> list[dict]:
    """Scalable near-dedup via datasketch MinHash-LSH; falls back to pure if absent."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("    (datasketch not installed — using slower pure-Python near-dedup)")
        return rp.dedup_near(records, threshold=threshold)
    import re as _re
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: list[dict] = []
    for i, r in enumerate(records):
        toks = _re.findall(r"\w+", rp.assistant_text(r["messages"]).lower())
        mh = MinHash(num_perm=128)
        for t in set(toks):
            mh.update(t.encode("utf-8"))
        if lsh.query(mh):
            continue
        lsh.insert(f"r{i}", mh)
        kept.append(r)
    return kept


def _seeded_shuffle(records: list[dict], seed: int) -> None:
    import random
    random.Random(seed).shuffle(records)


def _report_balance(records: list[dict]) -> None:
    from collections import Counter
    rows = Counter(r.get("capability", "?") for r in records)
    # Rough token share (chars/4) — row ratios lie about token ratios because
    # multi-turn tutoring is token-heavy. train_sft.py can do an exact count.
    approx = Counter()
    for r in records:
        chars = sum(len(m["content"]) for m in r["messages"])
        approx[r.get("capability", "?")] += chars / 4
    total = sum(approx.values()) or 1
    print("\nBalance:")
    print(f"  rows by capability : {dict(rows)}")
    print("  approx token share : "
          + str({k: round(v / total, 3) for k, v in approx.items()}))
    rp_share = sum(v for k, v in approx.items() if k == "roleplay") / total
    if rp_share < 0.70:
        print(f"  ⚠ roleplay token share {rp_share:.0%} < 70% — raise RP caps or cut learning caps.")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    d = repo_root / "training" / "data"
    ap = argparse.ArgumentParser(description="Blend roleplay + teaching SFT data.")
    ap.add_argument("--out", default=str(d / "train.jsonl"))
    ap.add_argument("--holdout-out", default=str(d / "holdout.jsonl"))
    ap.add_argument("--holdout", type=float, default=0.02, help="Fraction held out for eval.")
    ap.add_argument("--my-stories", default=str(d / "my_stories_sft.jsonl"),
                    help="Your imported stories SFT file (from import_my_stories.py).")
    ap.add_argument("--char-name", default="the character",
                    help="Name used in the tutor system prompt and {{char}} substitution.")
    # per-source caps (defaults encode the ~77/15/8 roleplay-dominant ratio)
    ap.add_argument("--max-pippa", type=int, default=3000)
    ap.add_argument("--max-limarp", type=int, default=2000)
    ap.add_argument("--max-coser", type=int, default=1500)
    ap.add_argument("--max-dolly", type=int, default=1200)
    ap.add_argument("--max-rotut", type=int, default=None)      # tiny (~299): keep all
    ap.add_argument("--max-platypus", type=int, default=400)
    ap.add_argument("--min-bot-chars", type=int, default=120,
                    help="Drop PIPPA convos whose bot turns average fewer chars (short-reply fix).")
    # source toggles
    for s in ("pippa", "limarp", "coser", "dolly", "rotut", "platypus"):
        ap.add_argument(f"--no-{s}", action="store_true", help=f"Exclude {s}.")
    # cleaning
    ap.add_argument("--keep-slop", action="store_true", help="Do not drop slop-phrase rows.")
    ap.add_argument("--slop-max", type=int, default=0, help="Allowed slop hits per row.")
    ap.add_argument("--no-dedup", action="store_true", help="Skip near-duplicate removal.")
    ap.add_argument("--near-threshold", type=float, default=0.9)
    ap.add_argument("--near-dedup-cap", type=int, default=20000,
                    help="Above this many rows, use MinHash-LSH instead of pure near-dedup.")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
