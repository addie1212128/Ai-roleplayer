"""Headless-browser smoke test of the full UI against a fake Ollama.

Run:  python tests/browser_test.py
Drives the real page in Chromium: start a story, send a message, see the reply
stream in, generate a swipe, open settings. Fails on any JS console error.
"""
import glob
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _detect_chrome():
    """$CHROME_PATH wins; else a pre-installed Playwright Chromium if present;
    else None, which makes Playwright use its own managed browser."""
    if os.environ.get("CHROME_PATH"):
        return os.environ["CHROME_PATH"]
    hits = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    return hits[-1] if hits else None


CHROME = _detect_chrome()
FAILED = []


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {extra}" if extra and not cond else ""))
    if not cond:
        FAILED.append(name)


# redirect storage to a temp dir before importing the app
TMP = Path(tempfile.mkdtemp(prefix="rp_browser_"))
import app.storage as storage  # noqa: E402
storage.DATA = TMP / "data"
storage.CHAR_DIR = storage.DATA / "characters"
storage.STORY_DIR = storage.DATA / "stories"
storage.LORE_DIR = storage.DATA / "lorebooks"
storage.PERSONA_DIR = storage.DATA / "personas"
storage._DIRS = (storage.CHAR_DIR, storage.STORY_DIR, storage.LORE_DIR, storage.PERSONA_DIR)


def start_fake_ollama(port):
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
        if not body.get("stream"):
            return JSONResponse({"message": {"content": "- summary."}, "done": True})

        async def gen():
            for tok in ["*She looks up.* ", "\"Hello", " there,", " traveler.\""]:
                yield json.dumps({"message": {"content": tok}}) + "\n"
            yield json.dumps({"message": {"content": ""}, "done": True,
                              "prompt_eval_count": 210, "eval_count": 30}) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    cfg = uvicorn.Config(fake, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    return srv


def start_app(port, ollama_port):
    import uvicorn
    import app.main as main
    main.OLLAMA_URL = f"http://127.0.0.1:{ollama_port}"
    cfg = uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    return srv


def wait_up(url):
    import httpx
    for _ in range(80):
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"server {url} not up")


def main():
    oport, aport = 11601, 8611
    start_fake_ollama(oport)
    wait_up(f"http://127.0.0.1:{oport}/api/tags")
    start_app(aport, oport)
    wait_up(f"http://127.0.0.1:{aport}/")
    base = f"http://127.0.0.1:{aport}"

    from playwright.sync_api import sync_playwright
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME or None, args=["--no-sandbox"])
        page = browser.new_page()
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base)

        # characters seeded from presets should appear
        page.wait_for_selector("#character-list .list-item", timeout=5000)
        n = page.locator("#character-list .list-item").count()
        check("UI: characters listed", n >= 1, f"count={n}")

        # start a story with the first character
        page.locator("#character-list .list-item").first.click()
        page.wait_for_selector("#composer:not(.hidden)", timeout=5000)
        check("UI: composer shows after starting story", True)
        check("UI: greeting rendered", page.locator(".msg.ai .bubble").count() >= 1)

        # send a message
        page.fill("#input", "I step inside out of the rain.")
        page.keyboard.press("Enter")
        page.wait_for_function("document.querySelector('#messages').innerText.includes('Hello there, traveler.')", timeout=8000)
        check("UI: user message shown", page.locator(".msg.user").count() >= 1)
        check("UI: assistant reply streamed & rendered", "Hello there, traveler." in page.inner_text("#messages"))
        check("UI: action rendered as italics", page.locator(".bubble em").count() >= 1)

        # context meter updated from real token counts
        page.wait_for_function("!document.querySelector('#ctx-meter').classList.contains('hidden')", timeout=3000)
        check("UI: ctx meter shows tokens", "210" in page.inner_text("#ctx-meter"), page.inner_text("#ctx-meter"))

        # swipe: click the '›' on the last assistant message to generate an alternate
        last = page.locator(".msg.ai").last
        last.locator(".swipe-nav button").nth(1).click()
        page.wait_for_function("document.querySelector('.msg.ai:last-child .swipe-nav .count') && document.querySelector('.msg.ai:last-child .swipe-nav .count').innerText.trim()==='2/2'", timeout=8000)
        check("UI: swipe generated a 2nd version", "2/2" in page.inner_text(".msg.ai:last-child .swipe-nav .count"))

        # open settings drawer
        page.click("#settings-btn")
        page.wait_for_selector("#settings-modal:not(.hidden)", timeout=3000)
        check("UI: settings drawer opens", page.locator("#s-ctx").count() == 1)
        check("UI: settings num_ctx populated", int(page.input_value("#s-ctx")) >= 4096)
        page.click("#settings-cancel")

        # character editor opens with advanced fields
        page.click("#edit-active-char")
        page.wait_for_selector("#char-modal:not(.hidden)", timeout=3000)
        check("UI: char editor has example field", page.locator("#c-example").count() == 1)
        check("UI: permanent-token badge computed", "tokens" in page.inner_text("#perm-tokens"))
        page.click("#char-cancel")

        page.screenshot(path=str(TMP / "ui.png"), full_page=True)
        browser.close()

    check("UI: no JS console errors", not errors, str(errors))
    print()
    if FAILED:
        print(f"❌ {len(FAILED)} UI check(s) failed: {FAILED}")
        sys.exit(1)
    print("✅ All UI checks passed.")


if __name__ == "__main__":
    main()
