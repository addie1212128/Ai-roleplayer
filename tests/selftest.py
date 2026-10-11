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
storage.DATA = TMP / "data"
storage.CHAR_DIR = storage.DATA / "characters"
storage.STORY_DIR = storage.DATA / "stories"
storage.LORE_DIR = storage.DATA / "lorebooks"
storage.PERSONA_DIR = storage.DATA / "personas"
storage._DIRS = (storage.CHAR_DIR, storage.STORY_DIR, storage.LORE_DIR, storage.PERSONA_DIR)

from app import cards, context          # noqa: E402
from app.samplers import preset_for, build_options, DEFAULTS  # noqa: E402

# ============================================================ pure logic tests

def test_samplers():
    s = preset_for("mistral-nemo:12b")
    check("preset: nemo min_p=0.08", s["min_p"] == 0.08)
    check("preset: cydonia repeat_penalty=1.0", preset_for("hf.co/x/Cydonia-24B")["repeat_penalty"] == 1.0)
    check("preset: unknown falls back to defaults", preset_for("something-random")["temperature"] == DEFAULTS["temperature"])
    opts = build_options({"num_ctx": 16384, "temperature": 1.0, "keep_alive": "30m", "model": "x", "bogus": 1})
    check("build_options keeps num_ctx", opts.get("num_ctx") == 16384)
    check("build_options drops non-ollama keys", "keep_alive" not in opts and "model" not in opts and "bogus" not in opts)


def test_strip_think():
    check("strip closed think", context.strip_think("<think>plan</think>Hello") == "Hello")
    check("strip unclosed think", context.strip_think("Hi <think>oops no close") == "Hi")
    check("no think untouched", context.strip_think("Just text") == "Just text")


def test_substitute():
    out = context.substitute("Hi {{user}}, I am {{char}}.", "Aria", "Sam")
    check("macro substitution", out == "Hi Sam, I am Aria.")


def test_examples():
    ex = "<START>\nSam: Hello\nAria: *waves* Hi there\n<START>\nSam: Bye\nAria: Farewell"
    msgs = context.parse_examples(ex, "Aria", "Sam")
    roles = [m["role"] for m in msgs]
    check("examples parsed into 4 msgs", len(msgs) == 4, f"got {len(msgs)}")
    check("examples role order", roles == ["user", "assistant", "user", "assistant"], str(roles))


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

    @fake.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        _last_payload.clear()
        _last_payload.update(body)
        if not body.get("stream"):  # summarizer path
            return JSONResponse({"message": {"content": "- A thing happened."}, "done": True})

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

    # persona
    p = client.post("/api/personas", json={"name": "Sam", "description": "A traveler."}).json()
    check("api: persona created", p["id"] and p["name"] == "Sam")

    # delete-from truncation
    client.delete(f"/api/stories/{story['id']}/message/1")
    check("api: truncate", len(client.get(f"/api/stories/{story['id']}").json()["messages"]) == 1)


def main():
    print("Running AI Roleplayer backend self-test...\n")
    for fn in [test_samplers, test_strip_think, test_substitute, test_examples,
               test_lorebook, test_assemble_order_and_budget, test_splice_depth,
               test_macro_resolution, test_depth_zero_and_est, test_cards_hardening,
               test_cards, test_storage_atomic, test_integration]:
        try:
            fn()
        except Exception as e:
            import traceback
            check(fn.__name__ + " (crashed)", False, repr(e))
            traceback.print_exc()
    print()
    if FAILED:
        print(f"❌ {len(FAILED)} check(s) failed: {FAILED}")
        sys.exit(1)
    print("✅ All checks passed.")


if __name__ == "__main__":
    main()
