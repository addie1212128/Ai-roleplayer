"""Standalone self-test for the AI Roleplayer backend.

Run from the project root:  python tests/selftest.py
Exits non-zero on the first failure. Uses a temp data dir and a fake in-process
Ollama server, so it needs no real Ollama and never touches your real data/.
"""
import base64
import json
import os
import struct
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  — {extra}" if extra and not cond else ""))
    if not cond:
        FAILED.append(name)


# --- redirect storage to a temp dir BEFORE importing the app ------------------
TMP = Path(tempfile.mkdtemp(prefix="rp_test_"))
import app.storage as storage  # noqa: E402
storage.use_data_dir(TMP / "data")

from app import cards, context, rag     # noqa: E402
from app.samplers import preset_for, build_options, pick_sampler_fields, DEFAULTS  # noqa: E402

# ============================================================ pure logic tests

def test_samplers():
    s = preset_for("mistral-nemo:12b")
    check("preset: nemo min_p=0.08", s["min_p"] == 0.08)
    check("preset: cydonia repeat_penalty=1.0", preset_for("hf.co/x/Cydonia-24B")["repeat_penalty"] == 1.0)
    check("preset: unknown falls back to defaults", preset_for("something-random")["temperature"] == DEFAULTS["temperature"])
    opts = build_options({"num_ctx": 16384, "temperature": 1.0, "keep_alive": "30m", "model": "x", "bogus": 1})
    check("build_options keeps num_ctx", opts.get("num_ctx") == 16384)
    check("build_options drops non-ollama keys", "keep_alive" not in opts and "model" not in opts and "bogus" not in opts)
    clamped = build_options({"num_ctx": 10_000_000, "num_predict": 99999, "top_k": 99999, "temperature": 50})
    check("clamp: num_ctx capped", clamped["num_ctx"] == 131072)
    check("clamp: num_predict capped", clamped["num_predict"] == 8192)
    check("clamp: temperature capped", clamped["temperature"] == 3.0)
    picked = pick_sampler_fields({"temperature": 0.7, "ban_phrases": "x", "bogus": 9, "model": "m", "seed": 3, "min_p": None})
    check("pick_sampler_fields keeps knobs", picked.get("temperature") == 0.7 and picked.get("ban_phrases") == "x")
    check("pick_sampler_fields drops junk/model/seed/None",
          "bogus" not in picked and "model" not in picked and "seed" not in picked and "min_p" not in picked)


def test_strip_think():
    check("strip closed think", context.strip_think("<think>plan</think>Hello") == "Hello")
    check("strip unclosed think", context.strip_think("Hi <think>oops no close") == "Hi")
    check("no think untouched", context.strip_think("Just text") == "Just text")


def test_substitute():
    out = context.substitute("Hi {{user}}, I am {{char}}.", "Aria", "Sam")
    check("macro substitution", out == "Hi Sam, I am Aria.")
    # single pass: a name containing a macro token must not be re-substituted
    out2 = context.substitute("{{char}} and {{user}}", "{{user}}", "Bob")
    check("substitute single-pass (no re-substitution)", out2 == "{{user}} and Bob", out2)


def test_paper_features():
    character = {"name": "Rex", "description": "A pilot.", "personality": "Bold.", "scenario": "A hangar.",
                 "career": "Astronaut", "age": "34", "traits": "brave, reckless",
                 "speech_style": "clipped military jargon"}
    story = {"messages": [{"role": "user", "content": "hi"}],
             "scene": "Midnight on the launchpad, a storm rolling in.", "mood": "Tense"}
    built = context.assemble_prompt(story, character, None, None, {"num_ctx": 8192, "inner_monologue": True})
    allc = "\n".join(m["content"] for m in built["messages"])
    check("paper: profile fields in prompt", "Career: Astronaut" in allc and "clipped military jargon" in allc)
    check("paper: scene block injected", "Midnight on the launchpad" in allc)
    check("paper: mood injected", "feeling Tense" in allc, allc[-200:])
    check("paper: inner-monologue instruction present", "INNER MONOLOGUE" in allc and "((" in allc)
    built2 = context.assemble_prompt(story, character, None, None, {"num_ctx": 8192})
    allc2 = "\n".join(m["content"] for m in built2["messages"])
    check("paper: inner monologue off by default", "INNER MONOLOGUE" not in allc2)
    # empty profile -> no stray labels
    plain = context.assemble_prompt({"messages": []}, {"name": "Q", "description": "d"}, None, None, {"num_ctx": 8192})
    check("paper: no profile labels when empty", "Career:" not in plain["messages"][0]["content"])


def test_ban_phrases():
    # parser: newline-split, case-insensitive dedupe, blanks dropped, and a
    # phrase may contain commas (so the classic "not X, but Y" survives intact).
    bans = context.parse_ban_phrases("shiver down the spine\n\nNOT X, but Y\nshiver down the spine\n")
    check("ban: parse dedupes + keeps comma phrases",
          bans == ["shiver down the spine", "NOT X, but Y"], str(bans))
    check("ban: empty/None -> []", context.parse_ban_phrases("") == [] and context.parse_ban_phrases(None) == [])
    check("ban: count is bounded", len(context.parse_ban_phrases("\n".join(f"p{i}" for i in range(200)))) == 60)
    # injection reaches the assembled prompt with the phrases quoted
    story = {"messages": [{"role": "user", "content": "hi"}]}
    built = context.assemble_prompt(story, {"name": "Q", "description": "d"}, None, None,
                                    {"num_ctx": 8192, "ban_phrases": "barely above a whisper\neyes sparkling"})
    contents = [m["content"] for m in built["messages"]]
    joined = "\n".join(contents)
    check("ban: injected into prompt", "Banned words and phrases" in joined and "barely above a whisper" in joined)
    # the final in-character reminder must stay last (ban line goes before it)
    ban_idx = max(i for i, c in enumerate(contents) if "Banned words" in c)
    rem_idx = max(i for i, c in enumerate(contents) if "Stay in character" in c)
    check("ban: reminder stays after the ban line", rem_idx > ban_idx, f"ban={ban_idx} rem={rem_idx}")
    plain = context.assemble_prompt(story, {"name": "Q", "description": "d"}, None, None, {"num_ctx": 8192})
    check("ban: absent when unset", not any("Banned words" in m["content"] for m in plain["messages"]))


def test_storage_id_safety():
    saved = storage.save_persona({"id": "../../evil", "name": "n"})
    check("storage: unsafe id regenerated on save", storage._safe_id(saved["id"]))
    check("storage: get rejects traversal id", storage.get_persona("../../evil") is None)
    check("storage: delete rejects traversal id", storage.delete_persona("../../evil") is False)
    check("storage: no file escaped the store", not (storage.PERSONA_DIR.parent.parent / "evil.json").exists())


def test_examples():
    ex = "<START>\nSam: Hello\nAria: *waves* Hi there\n<START>\nSam: Bye\nAria: Farewell"
    msgs = context.parse_examples(ex, "Aria", "Sam")
    roles = [m["role"] for m in msgs]
    check("examples parsed into 4 msgs", len(msgs) == 4, f"got {len(msgs)}")
    check("examples role order", roles == ["user", "assistant", "user", "assistant"], str(roles))
    # examples are injected as ONE fenced system block, never as fake chat turns
    built = context.assemble_prompt({"messages": [{"role": "user", "content": "hi"}]},
                                    {"name": "Aria", "description": "d", "example_dialogue": ex}, None, None, {"num_ctx": 8192})
    check("examples: injected as one system block",
          any(m["role"] == "system" and "Example dialogue" in m["content"] for m in built["messages"]))
    check("examples: not injected as fake assistant turns",
          sum(1 for m in built["messages"] if m["role"] == "assistant") == 0)


def test_lorebook():
    lb = {"scanDepth": 4, "entries": [
        {"keys": ["castle"], "content": "The castle is haunted.", "order": 100, "position": "before"},
        {"keys": ["dragon"], "content": "Dragons hoard gold.", "order": 50, "position": "after", "depth": 4},
        {"keys": ["unused"], "content": "nope", "order": 10},
        {"keys": [], "constant": True, "content": "Always here.", "order": 5},
    ]}
    recent = [{"role": "user", "content": "I enter the CASTLE looking for a dragon."}]
    res = context.scan_lorebook(lb, recent, 1000)
    check("lore fires castle (before)", "The castle is haunted." in res["before"])
    check("lore fires constant", "Always here." in res["before"])
    check("lore fires dragon at depth", any("Dragons" in c for _, c in res["at_depth"]))
    check("lore skips unmatched", not any("nope" in c for _, c in res["at_depth"]) and "nope" not in res["before"])
    # whole-word matching: "cat" must not match "category"
    lb2 = {"entries": [{"keys": ["cat"], "content": "meow", "matchWholeWords": True}]}
    res2 = context.scan_lorebook(lb2, [{"role": "user", "content": "I like category theory"}], 1000)
    check("lore whole-word guard", res2["fired"] == [])


def test_assemble_order_and_budget():
    character = {"name": "Aria", "description": "A tavern keeper.", "personality": "Warm.",
                 "scenario": "A rainy night.", "greeting": "Hello."}
    persona = {"name": "Sam", "description": "A tired traveler."}
    story = {
        "messages": [
            {"role": "assistant", "content": "Welcome in."},
            {"role": "user", "content": "I sit by the fire."},
        ],
        "summaries": [{"range": [0, 2], "text": "- Sam arrived at the tavern."}],
        "authors_note": {"text": "Keep it cozy.", "depth": 1, "role": "system"},
    }
    built = context.assemble_prompt(story, character, persona, None, {"num_ctx": 8192, "num_predict": 300})
    msgs = built["messages"]
    check("assemble: first msg is system", msgs[0]["role"] == "system")
    check("assemble: card in system block", "tavern keeper" in msgs[0]["content"])
    check("assemble: persona in system block", "tired traveler" in msgs[0]["content"])
    check("assemble: story-so-far present", any("Story so far" in m["content"] for m in msgs))
    check("assemble: author's note injected", any("Author's note" in m["content"] for m in msgs))
    check("assemble: reminder at end region", any("Stay in character" in m["content"] for m in msgs))
    check("assemble: history preserved", any(m["content"] == "I sit by the fire." for m in msgs))
    check("assemble: debug token estimate present", built["debug"]["estimated_prompt_tokens"] > 0)

    # Budget trimming: a huge history must be trimmed to fit a tiny context.
    big = {"messages": [{"role": "user" if i % 2 == 0 else "assistant",
                         "content": "word " * 200} for i in range(50)]}
    built2 = context.assemble_prompt(big, character, None, None, {"num_ctx": 2048, "num_predict": 256})
    check("assemble: trims history under budget",
          built2["debug"]["history_messages_kept"] < built2["debug"]["history_messages_total"],
          str(built2["debug"]))


def test_splice_depth():
    hist = [{"role": "user", "content": f"m{i}"} for i in range(4)]  # m0..m3
    out = context._splice_depth(hist, [(0, {"role": "system", "content": "END"})])
    check("splice depth 0 -> very end", out[-1]["content"] == "END")
    out2 = context._splice_depth(hist, [(1, {"role": "system", "content": "D1"})])
    check("splice depth 1 -> before last", out2[-2]["content"] == "D1" and out2[-1]["content"] == "m3")


def _make_png_with_card(card_obj, keyword=b"chara"):
    b64 = base64.b64encode(json.dumps(card_obj).encode()).decode()

    def chunk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)

    text = keyword + b"\x00" + b64.encode()
    return cards.PNG_SIG + chunk(b"IHDR", b"\x00" * 13) + chunk(b"tEXt", text) + chunk(b"IEND", b"")


def test_cards():
    v2 = {"spec": "chara_card_v2", "spec_version": "2.0", "data": {
        "name": "Vale", "description": "A captain.", "personality": "Dry.",
        "scenario": "Docked.", "first_mes": "Hey.", "mes_example": "<START>\nYou: hi\nVale: hello",
        "character_book": {"entries": [{"keys": ["ship"], "content": "The Rust Sparrow is old.",
                                        "insertion_order": 100, "enabled": True}]}}}
    parsed = cards.character_from_card(v2)
    char = parsed["character"]
    check("card: name mapped", char["name"] == "Vale")
    check("card: first_mes -> greeting", char["greeting"] == "Hey.")
    check("card: mes_example -> example_dialogue", "hello" in char["example_dialogue"])
    lb = cards.lorebook_from_character_book(parsed["character_book"], "Vale lore")
    check("card: character_book -> lorebook", lb["entries"][0]["keys"] == ["ship"])

    # untrusted character_book with wrong field types must normalize, not crash
    book = {"scan_depth": "bad", "entries": [{"keys": "castle", "content": 123, "depth": "oops"}]}
    lb2 = cards.lorebook_from_character_book(book, "B")
    check("card: string key -> single-element list", lb2["entries"][0]["keys"] == ["castle"])
    check("card: non-string content coerced", lb2["entries"][0]["content"] == "123")
    check("card: bad depth -> 4", lb2["entries"][0]["depth"] == 4)
    check("card: bad scanDepth -> 4", lb2["scanDepth"] == 4)

    # PNG round trip
    png = _make_png_with_card(v2)
    got = cards.extract_png_card(png)
    check("card: extract from PNG tEXt", got and got["data"]["name"] == "Vale")
    parsed_png = cards.parse_card_bytes(png)
    check("card: parse_card_bytes(PNG)", parsed_png["character"]["name"] == "Vale")

    # flat V1 json
    v1 = json.dumps({"name": "Old", "description": "d", "first_mes": "hi"}).encode()
    check("card: parse flat V1", cards.parse_card_bytes(v1)["character"]["name"] == "Old")

    # export
    exported = cards.card_from_character({"name": "X", "greeting": "yo", "description": "d"})
    check("card: export envelope", exported["spec"] == "chara_card_v2" and exported["data"]["first_mes"] == "yo")


def test_macro_resolution():
    # Regression: default system prompt must resolve {{char}}/{{user}}, not leak {char}/{user}.
    character = {"name": "Zara", "description": "A rogue.", "personality": "Sly.", "scenario": "A heist."}
    persona = {"name": "Max", "description": "A nervous guard."}
    built = context.assemble_prompt({"messages": [{"role": "user", "content": "hi"}]},
                                    character, persona, None, {"num_ctx": 8192})
    sysmsg = built["messages"][0]["content"]
    check("macro: char name resolved", "You are Zara" in sysmsg, sysmsg[:40])
    check("macro: no {char} leak", "{char}" not in sysmsg and "{{char}}" not in sysmsg)
    check("macro: no {user} leak", "{user}" not in sysmsg and "{{user}}" not in sysmsg)
    check("macro: persona name present", "Max" in sysmsg)
    reminder = [m for m in built["messages"] if "Stay in character" in m["content"]][0]["content"]
    check("macro: reminder fully resolved", "Zara" in reminder and "Max" in reminder and "{" not in reminder)
    # override with {{original}} still works and resolves
    c2 = {**character, "system_prompt": "ONLY: {{original}}"}
    b2 = context.assemble_prompt({"messages": []}, c2, persona, None, {"num_ctx": 8192})
    check("macro: override {{original}} expands", "ONLY:" in b2["messages"][0]["content"] and "You are Zara" in b2["messages"][0]["content"])


def test_depth_zero_and_est():
    story = {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}],
             "authors_note": {"text": "Be tense.", "depth": 0, "role": "system"}}
    built = context.assemble_prompt(story, {"name": "C", "description": "d"}, None, None, {"num_ctx": 8192})
    contents = [m["content"] for m in built["messages"]]
    b_idx = max(i for i, c in enumerate(contents) if c == "b")
    an_idx = [i for i, c in enumerate(contents) if "Author's note" in c][0]
    check("AN depth=0 lands after last history msg", an_idx > b_idx, f"an={an_idx} b={b_idx}")
    check("est_tokens ascii ~ len/4", context.est_tokens("abcdefgh") == 2)
    check("est_tokens cjk heavier", context.est_tokens("日本語テスト") == 6)
    check("strip unclosed <thinking>", context.strip_think("Hi <thinking>pondering") == "Hi")
    check("strip closed <thinking>", context.strip_think("<thinking>x</thinking>Yo") == "Yo")
    lb = {"entries": [{"keys": ["日本"], "content": "Japan lore", "matchWholeWords": True}]}
    res = context.scan_lorebook(lb, [{"role": "user", "content": "私は日本に行く"}], 1000)
    check("lore CJK key matches (substring fallback)", "Japan lore" in res["before"])


def test_cards_hardening():
    import struct as _st, zlib as _zl, time as _t
    # non-object JSON embedded in a PNG must raise ValueError, not crash with 500
    png = _make_png_with_card([1, 2, 3])
    try:
        cards.parse_card_bytes(png); ok = False
    except ValueError:
        ok = True
    check("cards: non-object PNG JSON -> ValueError", ok)
    # zip bomb: a zTXt chunk inflating past the cap must be defused (no hang/OOM)
    comp = _zl.compress(b"\x00" * 5_000_000)

    def chunk(typ, data):
        return _st.pack(">I", len(data)) + typ + data + _st.pack(">I", _zl.crc32(typ + data) & 0xffffffff)

    ztxt = b"chara\x00\x00" + comp  # keyword\0 + method byte + compressed
    bomb = cards.PNG_SIG + chunk(b"IHDR", b"\x00" * 13) + chunk(b"zTXt", ztxt) + chunk(b"IEND", b"")
    t0 = _t.time()
    got = cards.extract_png_card(bomb)
    check("cards: zip-bomb defused (no card, fast)", got is None and (_t.time() - t0) < 2.0)


def test_rag():
    import asyncio
    vocab = ["dragon", "castle", "ship", "forest", "king", "market"]

    def fake_vec(t):
        tl = t.lower()
        return rag._normalize([1.0 if w in tl else 0.0 for w in vocab] + [0.05])

    async def fake_embed(texts, model, url):
        return [fake_vec(t) for t in texts]

    check("rag: cosine identical ~1", abs(rag.cosine(fake_vec("dragon"), fake_vec("dragon")) - 1.0) < 1e-6)
    check("rag: cosine unrelated ~0", abs(rag.cosine(fake_vec("dragon"), fake_vec("market"))) < 0.5)
    check("rag: chunking splits long text", len(rag.chunk_messages([{"role": "user", "content": "w " * 1000}])) >= 2)

    orig = rag.embed_texts
    rag.embed_texts = fake_embed
    try:
        msgs = [{"role": "assistant", "content": "A fearsome DRAGON guards the castle hoard."}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"we visit the market, day {i}"})
            msgs.append({"role": "assistant", "content": f"the day is calm, {i}"})
        story = storage.save_story({"messages": msgs, "summaries": []})
        asyncio.run(rag.update_store(story, "fake", "http://x"))
        store = storage.get_vec(story["id"])
        check("rag: store persisted vectors", bool(store and store.get("chunks")))
        block = asyncio.run(rag.retrieve(story, "tell me about the dragon and castle", "fake", "http://x", k=2))
        check("rag: retrieves relevant moment", "DRAGON" in block, block[:80])
        # pruning must persist even when nothing new is embedded, or deleted
        # content would keep being retrieved from the stale sidecar.
        short = {"id": story["id"], "messages": msgs[:3], "summaries": []}
        asyncio.run(rag.update_store(short, "fake", "http://x"))
        check("rag: prune persisted with no new embeds", len(storage.get_vec(story["id"])["chunks"]) == 0)
    finally:
        rag.embed_texts = orig


def test_storage_atomic():
    storage.seed_presets()
    c = storage.save_character({"name": "Tmp", "description": "x", "custom_field": 42})
    got = storage.get_character(c["id"])
    check("storage: round-trips extra fields", got.get("custom_field") == 42)
    check("storage: atomic (no .tmp left)", not any(storage.CHAR_DIR.glob("*.tmp")))


# ============================================================ integration test

_last_payload = {}
_fake_ctl = {"empty": False}


def _start_fake_ollama(port):
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn

    fake = FastAPI()

    @fake.get("/api/tags")
    async def tags():
        return {"models": [{"name": "mistral-nemo:12b"}]}

    @fake.post("/api/embed")
    async def embed(request: Request):
        body = await request.json()
        inp = body.get("input") or []
        vocab = ["dragon", "castle", "ship", "forest", "king", "market"]

        def vec(t):
            tl = t.lower()
            return [1.0 if w in tl else 0.0 for w in vocab] + [0.05]
        return {"embeddings": [vec(t) for t in inp]}

    @fake.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        if not body.get("stream"):  # non-streaming paths — don't record them
            sysmsg = (body.get("messages") or [{}])[0].get("content", "")
            if "design characters" in sysmsg:   # character-generator path -> JSON
                return JSONResponse({"message": {"content": (
                    '{"name":"Ember","description":"A fire mage.","personality":"Bold.",'
                    '"scenario":"A ruined tower.","greeting":"*flames flicker* Hello."}')},
                    "done": True})
            return JSONResponse({"message": {"content": "- A thing happened."}, "done": True})
        _last_payload.clear()
        _last_payload.update(body)  # record only the streaming RP request

        async def gen():
            if _fake_ctl.get("empty"):   # simulate a failed/empty generation
                yield json.dumps({"message": {"content": ""}, "done": True,
                                  "prompt_eval_count": 5, "eval_count": 0}) + "\n"
                return
            for tok in ["*She ", "smiles.* ", "\"Hello", " there.\""]:
                yield json.dumps({"message": {"content": tok}, "done": False}) + "\n"
            yield json.dumps({"message": {"content": ""}, "done": True,
                              "prompt_eval_count": 123, "eval_count": 45}) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    config = uvicorn.Config(fake, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    # wait for readiness
    import httpx
    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/tags", timeout=0.5).status_code == 200:
                return server
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("fake ollama did not start")


def test_integration():
    port = 11577
    _start_fake_ollama(port)

    import app.main as main
    main.OLLAMA_URL = f"http://127.0.0.1:{port}"
    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    # models
    r = client.get("/api/models")
    check("api: models ok", r.json()["ok"] and "mistral-nemo:12b" in r.json()["models"])

    # token auth: when API_TOKEN is set, /api requires it; the page still loads
    main.API_TOKEN = "secret123"
    check("auth: /api without token -> 401", client.get("/api/models").status_code == 401)
    check("auth: /api with header token -> ok", client.get("/api/models", headers={"X-RP-Token": "secret123"}).json()["ok"])
    check("auth: /api with query token -> ok", client.get("/api/models?token=secret123").json()["ok"])
    check("auth: page loads without token", client.get("/").status_code == 200)
    main.API_TOKEN = ""

    # character + story
    char = client.post("/api/characters", json={"name": "Aria", "description": "Tavern keeper.",
                                                 "personality": "Warm.", "scenario": "Rainy night.",
                                                 "greeting": "Welcome in."}).json()
    story = client.post("/api/stories", json={"title": "Test", "character_id": char["id"],
                                              "character_name": "Aria", "model": "mistral-nemo:12b",
                                              "messages": [{"role": "assistant", "content": "Welcome in."}]}).json()
    check("api: story seeds greeting swipe", story["messages"][0].get("swipes") == ["Welcome in."])

    # chat send (streaming)
    r = client.post("/api/chat", json={"story_id": story["id"], "action": "send",
                                        "user_message": "I sit by the fire."})
    body = r.text
    tokens = [json.loads(l[6:]) for l in body.splitlines() if l.startswith("data: ")]
    text = "".join(t.get("token", "") for t in tokens)
    done = [t for t in tokens if t.get("done")]
    check("api: streamed tokens", "Hello there." in text, text)
    check("api: P0 num_ctx sent to ollama", _last_payload.get("options", {}).get("num_ctx") == DEFAULTS["num_ctx"],
          str(_last_payload.get("options")))
    check("api: min_p sent to ollama", _last_payload.get("options", {}).get("min_p") == 0.08)
    check("api: real token counts returned", done and done[0].get("prompt_tokens") == 123)

    # server persisted both turns
    story2 = client.get(f"/api/stories/{story['id']}").json()
    roles = [m["role"] for m in story2["messages"]]
    check("api: server persisted user+assistant", roles == ["assistant", "user", "assistant"], str(roles))
    check("api: assistant stored clean", story2["messages"][-1]["content"] == "*She smiles.* \"Hello there.\"")

    # swipe -> adds an alternate
    client.post("/api/chat", json={"story_id": story["id"], "action": "swipe"})
    story3 = client.get(f"/api/stories/{story['id']}").json()
    check("api: swipe adds alternate", len(story3["messages"][-1]["swipes"]) == 2,
          str(len(story3["messages"][-1]["swipes"])))

    # regenerate -> replaces last assistant (count stays at 3)
    client.post("/api/chat", json={"story_id": story["id"], "action": "regenerate"})
    story4 = client.get(f"/api/stories/{story['id']}").json()
    check("api: regenerate keeps length", len(story4["messages"]) == 3, str(len(story4["messages"])))

    # regenerate FAILURE must NOT lose the previous reply (data-loss fix)
    prior = client.get(f"/api/stories/{story['id']}").json()["messages"][-1]["content"]
    _fake_ctl["empty"] = True
    client.post("/api/chat", json={"story_id": story["id"], "action": "regenerate"})
    after = client.get(f"/api/stories/{story['id']}").json()["messages"]
    check("api: failed regenerate preserves prior reply", len(after) == 3 and after[-1]["content"] == prior, f"len={len(after)}")
    # send FAILURE must NOT persist an orphan user turn (deferred-commit fix)
    client.post("/api/chat", json={"story_id": story["id"], "action": "send", "user_message": "doomed line"})
    after2 = client.get(f"/api/stories/{story['id']}").json()["messages"]
    check("api: failed send leaves no orphan user turn", len(after2) == 3, f"len={len(after2)}")
    _fake_ctl["empty"] = False
    r_big = client.post("/api/chat", json={"story_id": story["id"], "action": "send", "user_message": "x" * 100001})
    check("api: oversized message rejected", "too long" in r_big.text)

    # export with a non-ASCII name must not 500 (Latin-1 header fix)
    jp = client.post("/api/characters", json={"name": "ゼロ丸", "description": "d", "greeting": "hi"}).json()
    ex = client.get(f"/api/characters/{jp['id']}/export")
    check("api: export non-ASCII name ok", ex.status_code == 200 and ex.json()["data"]["name"] == "ゼロ丸", str(ex.status_code))

    # persona can be set and then CLEARED via explicit null (exclude_unset fix)
    pz = client.post("/api/personas", json={"name": "Zed"}).json()
    client.put(f"/api/stories/{story['id']}/meta", json={"persona_id": pz["id"]})
    check("api: persona set", client.get(f"/api/stories/{story['id']}").json().get("persona_id") == pz["id"])
    client.put(f"/api/stories/{story['id']}/meta", json={"persona_id": None})
    check("api: persona cleared via null", client.get(f"/api/stories/{story['id']}").json().get("persona_id") is None)

    # edit message
    r = client.put(f"/api/stories/{story['id']}/message/1", json={"content": "I warm my hands."})
    check("api: edit message", client.get(f"/api/stories/{story['id']}").json()["messages"][1]["content"] == "I warm my hands.")

    # debug prompt
    dbg = client.get(f"/api/debug/prompt?story_id={story['id']}").json()
    check("api: debug prompt assembles", dbg["messages"][0]["role"] == "system" and dbg["debug"]["num_ctx"] == DEFAULTS["num_ctx"])

    # import a PNG card
    png = _make_png_with_card({"spec": "chara_card_v2", "data": {"name": "Imported", "first_mes": "hi", "description": "d"}})
    imp = client.post("/api/characters/import", json={"filename": "x.png",
                                                      "data_b64": base64.b64encode(png).decode()}).json()
    check("api: import PNG card", imp["name"] == "Imported" and imp["greeting"] == "hi")
    check("api: import sets avatar from PNG", imp.get("avatar_image") == f"/avatars/{imp['id']}.png")

    # avatar upload + serve
    av = client.post(f"/api/characters/{char['id']}/avatar",
                     json={"data_b64": base64.b64encode(png).decode(), "ext": "png"}).json()
    check("api: avatar set", av.get("avatar_image") == f"/avatars/{char['id']}.png")
    check("api: avatar served", client.get(av["avatar_image"]).status_code == 200)

    # story export -> re-import (fresh id, same messages)
    exp = client.get(f"/api/stories/{story['id']}/export")
    check("api: story export ok", exp.status_code == 200 and "messages" in exp.json())
    imp_story = client.post("/api/stories/import",
                            json={"filename": "s.json", "data_b64": base64.b64encode(exp.content).decode()}).json()
    check("api: story import new id + messages kept",
          imp_story["id"] != story["id"] and len(imp_story["messages"]) == len(exp.json()["messages"]))
    bad = base64.b64encode(json.dumps({"title": "x", "messages": [{"role": "user", "content": "ok"}, "nope"]}).encode()).decode()
    check("api: import rejects malformed messages",
          client.post("/api/stories/import", json={"filename": "b.json", "data_b64": bad}).status_code == 400)

    # full backup zip
    bk = client.get("/api/backup")
    check("api: backup zip", bk.status_code == 200 and bk.content[:2] == b"PK")

    # RAG end-to-end (via the fake /api/embed): a distinctive older moment is recalled
    big = [{"role": "assistant", "content": "A fearsome DRAGON guards the castle hoard."}]
    for i in range(20):
        big.append({"role": "user", "content": f"we visit the market, day {i}"})
        big.append({"role": "assistant", "content": f"the day is calm, {i}"})
    rstory = client.post("/api/stories", json={"title": "RAG", "character_id": char["id"],
                                               "character_name": "Aria", "model": "mistral-nemo:12b", "messages": big}).json()
    client.put(f"/api/stories/{rstory['id']}/meta", json={"settings": {"rag_enabled": True, "embed_model": "fake-embed"}})
    client.post("/api/chat", json={"story_id": rstory["id"], "action": "send", "user_message": "tell me about the dragon"})
    sent = " ".join(m.get("content", "") for m in _last_payload.get("messages", []))
    check("api: RAG injects a relevant earlier moment",
          "Relevant earlier moments" in sent and "DRAGON" in sent, sent[:120])

    # group scene: attribution + roster + routing
    bob = client.post("/api/characters", json={"name": "Bob", "description": "A bard.", "greeting": "Hi, I'm Bob."}).json()
    gstory = client.post("/api/stories", json={
        "title": "Group", "member_ids": [char["id"], bob["id"]], "character_name": "Group",
        "model": "mistral-nemo:12b",
        "messages": [{"role": "assistant", "content": "Hi, I'm Bob.", "char_id": bob["id"], "name": "Bob", "swipes": ["Hi, I'm Bob."], "swipe_id": 0}],
    }).json()
    check("api: group story keeps members", gstory.get("member_ids") == [char["id"], bob["id"]])
    client.post("/api/chat", json={"story_id": gstory["id"], "action": "send", "user_message": "Hey Bob, what's your tale?"})
    last = client.get(f"/api/stories/{gstory['id']}").json()["messages"][-1]
    check("api: group mention routes to named member", last.get("char_id") == bob["id"] and last.get("name") == "Bob", str(last.get("name")))
    sysmsg = next((m["content"] for m in _last_payload.get("messages", []) if m["role"] == "system"), "")
    check("api: group roster in system prompt", "[GROUP SCENE]" in sysmsg and "Bob" in sysmsg)
    client.post("/api/chat", json={"story_id": gstory["id"], "action": "group_speak", "char_id": char["id"]})
    check("api: group_speak routes to chosen member",
          client.get(f"/api/stories/{gstory['id']}").json()["messages"][-1].get("char_id") == char["id"])
    before = client.get(f"/api/stories/{gstory['id']}").json()["messages"][-1]
    client.post("/api/chat", json={"story_id": gstory["id"], "action": "swipe"})
    after = client.get(f"/api/stories/{gstory['id']}").json()["messages"][-1]
    check("api: group swipe keeps attribution + adds alternate",
          after.get("char_id") == before.get("char_id") and len(after["swipes"]) == len(before["swipes"]) + 1)

    # persona
    p = client.post("/api/personas", json={"name": "Sam", "description": "A traveler."}).json()
    check("api: persona created", p["id"] and p["name"] == "Sam")

    # delete-from truncation
    client.delete(f"/api/stories/{story['id']}/message/1")
    check("api: truncate", len(client.get(f"/api/stories/{story['id']}").json()["messages"]) == 1)

    # --- new features -------------------------------------------------------
    istory = client.post("/api/stories", json={
        "title": "Imp", "character_id": char["id"], "character_name": "Aria",
        "model": "mistral-nemo:12b",
        "messages": [{"role": "assistant", "content": "Welcome in."},
                     {"role": "user", "content": "I nod."}]}).json()

    # impersonate: returns a drafted user line and persists NOTHING
    before = len(client.get(f"/api/stories/{istory['id']}").json()["messages"])
    r = client.post("/api/chat", json={"story_id": istory["id"], "action": "impersonate"})
    itoks = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    idone = [t for t in itoks if t.get("done")]
    check("api: impersonate returns a draft",
          bool(idone) and idone[0].get("impersonate") and "Hello there." in idone[0].get("text", ""),
          r.text[:160])
    after = len(client.get(f"/api/stories/{istory['id']}").json()["messages"])
    check("api: impersonate persists nothing", after == before, f"{before}->{after}")

    # guided regenerate: a one-off director's note steers THIS turn — it reaches
    # the model as a final system line but is never written to the story.
    client.post("/api/chat", json={"story_id": istory["id"], "action": "regenerate",
                                   "note": "make it colder and shorter"})
    note_sys = "\n".join(m.get("content", "") for m in _last_payload.get("messages", []) if m["role"] == "system")
    check("api: director's note reaches the model",
          "Director's note" in note_sys and "colder and shorter" in note_sys, note_sys[-160:])
    saved = client.get(f"/api/stories/{istory['id']}").json()
    check("api: director's note not persisted",
          not any("Director's note" in (m.get("content") or "") for m in saved["messages"]))

    # anti-slop ban list flows story.settings -> effective_settings -> prompt
    client.put(f"/api/stories/{istory['id']}/meta", json={"settings": {"ban_phrases": "mischievous glint"}})
    client.post("/api/chat", json={"story_id": istory["id"], "action": "regenerate"})
    ban_sys = "\n".join(m.get("content", "") for m in _last_payload.get("messages", []) if m["role"] == "system")
    check("api: ban_phrases flow via story settings",
          "Banned words and phrases" in ban_sys and "mischievous glint" in ban_sys, ban_sys[-160:])

    # per-character default sampler settings: the character's default applies,
    # and an explicit story setting still overrides it.
    dc = client.post("/api/characters", json={"name": "Dee", "description": "d", "greeting": "hi",
                                              "default_settings": {"temperature": 0.4, "bogus": 1}}).json()
    check("api: character stores default_settings", dc.get("default_settings", {}).get("temperature") == 0.4)
    dstory = client.post("/api/stories", json={"title": "D", "character_id": dc["id"], "character_name": "Dee",
                                               "model": "mistral-nemo:12b",
                                               "messages": [{"role": "assistant", "content": "hi"},
                                                            {"role": "user", "content": "yo"}]}).json()
    client.post("/api/chat", json={"story_id": dstory["id"], "action": "regenerate"})
    check("api: character default temperature applied",
          _last_payload.get("options", {}).get("temperature") == 0.4, str(_last_payload.get("options")))
    client.put(f"/api/stories/{dstory['id']}/meta", json={"settings": {"temperature": 1.3}})
    client.post("/api/chat", json={"story_id": dstory["id"], "action": "regenerate"})
    check("api: story setting overrides character default",
          _last_payload.get("options", {}).get("temperature") == 1.3, str(_last_payload.get("options")))

    # sampler presets (named generation-setting bundles): CRUD + field filtering
    sp = client.post("/api/sampler_presets",
                     json={"name": "Spicy", "settings": {"temperature": 1.4, "bogus": 9, "model": "x"}}).json()
    check("api: sampler preset created + filtered",
          bool(sp["id"]) and sp["settings"].get("temperature") == 1.4
          and "bogus" not in sp["settings"] and "model" not in sp["settings"], str(sp.get("settings")))
    check("api: sampler preset listed", any(p["id"] == sp["id"] for p in client.get("/api/sampler_presets").json()))
    up = client.put(f"/api/sampler_presets/{sp['id']}",
                    json={"name": "Spicy2", "settings": {"temperature": 1.5}}).json()
    check("api: sampler preset updated", up["name"] == "Spicy2" and up["settings"]["temperature"] == 1.5)
    check("api: sampler preset deleted", client.delete(f"/api/sampler_presets/{sp['id']}").json().get("ok"))
    check("api: sampler preset gone",
          not any(p["id"] == sp["id"] for p in client.get("/api/sampler_presets").json()))

    # character generator: a concept -> drafted card fields
    gj = client.post("/api/generate_character",
                     json={"concept": "a fire mage", "model": "mistral-nemo:12b"}).json()
    check("api: generate_character returns fields",
          gj.get("ok") and gj["character"]["name"] == "Ember" and bool(gj["character"]["greeting"]))
    check("api: generate_character requires a model",
          client.post("/api/generate_character", json={"concept": "x"}).status_code == 400)

    # markdown transcript export
    tr = client.get(f"/api/stories/{istory['id']}/transcript")
    check("api: transcript is markdown",
          tr.status_code == 200 and tr.headers["content-type"].startswith("text/markdown"))
    check("api: transcript has turns",
          "# Imp" in tr.text and "**You:**" in tr.text and "**Aria:**" in tr.text, tr.text[:120])

    # Content-Disposition is sanitized: a control char (CR/LF) in the title must
    # not land in the header (crash / header-injection vector).
    cstory = client.post("/api/stories", json={
        "title": "Bad\nName", "character_id": char["id"], "character_name": "Aria",
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}).json()
    ctr = client.get(f"/api/stories/{cstory['id']}/transcript")
    cd = ctr.headers.get("content-disposition", "")
    check("api: export header strips control chars",
          ctr.status_code == 200 and "\n" not in cd and "\r" not in cd and "BadName.md" in cd, repr(cd[:90]))


def main():
    print("Running AI Roleplayer backend self-test...\n")
    for fn in [test_samplers, test_strip_think, test_substitute, test_examples,
               test_lorebook, test_assemble_order_and_budget, test_splice_depth,
               test_macro_resolution, test_depth_zero_and_est, test_cards_hardening,
               test_cards, test_rag, test_paper_features, test_ban_phrases,
               test_storage_atomic, test_storage_id_safety, test_integration]:
        try:
            fn()
        except Exception as e:
            import traceback
            check(fn.__name__ + " (crashed)", False, repr(e))
            traceback.print_exc()
    print()
    sys.stdout.flush()
    if FAILED:
        print(f"❌ {len(FAILED)} check(s) failed: {FAILED}")
        sys.stdout.flush()
        os._exit(1)
    print("✅ All checks passed.")
    sys.stdout.flush()
    # The fake Ollama runs in daemon uvicorn threads; Python/uvicorn interpreter
    # teardown can segfault at process exit AFTER every check has passed (exit 139
    # in CI). The pass/fail result is already decided, so exit hard with the real
    # status and skip the flaky native teardown.
    os._exit(0)


if __name__ == "__main__":
    main()
