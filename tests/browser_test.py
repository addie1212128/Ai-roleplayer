"""Headless-browser smoke test of the full UI against a fake Ollama.

Run:  python tests/browser_test.py
Drives the real page in Chromium: start a story, send a message, see the reply
stream in, generate a swipe, open settings. Fails on any JS console error.
"""
import base64
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
storage.use_data_dir(TMP / "data")


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
            sysmsg = (body.get("messages") or [{}])[0].get("content", "")
            if "design characters" in sysmsg:
                return JSONResponse({"message": {"content": (
                    '{"name":"Ember","description":"A fire mage.","personality":"Bold.",'
                    '"scenario":"A ruined tower.","greeting":"*flames flicker* Hello."}')},
                    "done": True})
            if "memory archivist" in sysmsg:
                return JSONResponse({"message": {"content": '["A fact worth remembering."]'}, "done": True})
            if "track the emotional arc" in sysmsg:
                return JSONResponse({"message": {"content":
                    '{"affinity_delta": 5, "status": "warming up", "mood": "Calm", "reason": "pleasant chat"}'},
                    "done": True})
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

        # keyboard shortcut: ArrowLeft/Right step through versions of the last reply
        page.click("#active-name")  # move focus off any control, not into an input
        page.keyboard.press("ArrowLeft")
        page.wait_for_function("document.querySelector('.msg.ai:last-child .swipe-nav .count') && document.querySelector('.msg.ai:last-child .swipe-nav .count').innerText.trim()==='1/2'", timeout=4000)
        check("UI: ArrowLeft steps to previous version", "1/2" in page.inner_text(".msg.ai:last-child .swipe-nav .count"))
        page.keyboard.press("ArrowRight")
        page.wait_for_function("document.querySelector('.msg.ai:last-child .swipe-nav .count') && document.querySelector('.msg.ai:last-child .swipe-nav .count').innerText.trim()==='2/2'", timeout=4000)
        check("UI: ArrowRight steps to next version", "2/2" in page.inner_text(".msg.ai:last-child .swipe-nav .count"))

        # copy button, guided-regenerate note box, and the shortcuts toast
        check("UI: copy button on messages", page.locator(".msg .msg-controls button[title='Copy text']").count() >= 1)
        gbtn = page.locator(".msg.ai").last.locator("button[title^='Guided regenerate']")
        check("UI: guided-regenerate button on last reply", gbtn.count() == 1)
        gbtn.click()
        page.wait_for_selector(".guide-row", timeout=3000)
        check("UI: guide note box opens", page.locator(".guide-row .guide-input").count() == 1)
        page.keyboard.press("Escape")
        check("UI: Escape closes the guide box", page.locator(".guide-row").count() == 0)
        page.click("#shortcuts-btn")
        page.wait_for_function("!document.querySelector('#toast').classList.contains('hidden')", timeout=3000)
        check("UI: shortcuts toast shows", "Shortcuts" in page.inner_text("#toast"))

        # chat branching: save the current timeline, see it listed, restore it
        page.on("dialog", lambda d: d.accept())  # auto-accept restore/delete confirms
        check("UI: branch button present", page.locator("#branch-btn").count() == 1)
        page.click("#branch-btn")
        page.wait_for_selector("#branch-modal:not(.hidden)", timeout=3000)
        page.fill("#branch-name", "Path A")
        page.click("#branch-save")
        page.wait_for_function("document.querySelectorAll('#branch-list .list-item').length >= 1", timeout=4000)
        check("UI: branch saved + listed", page.locator("#branch-list .list-item").count() >= 1)
        page.locator("#branch-list .list-item .act[title='Restore this timeline']").first.click()
        page.wait_for_function("document.querySelector('#branch-modal').classList.contains('hidden')", timeout=4000)
        check("UI: branch restore closes modal", page.locator("#branch-modal.hidden").count() == 1)

        # impersonate: draft the user's next line into the composer
        page.fill("#input", "")
        page.click("#impersonate-btn")
        page.wait_for_function("document.querySelector('#input').value.includes('traveler.')", timeout=8000)
        check("UI: impersonate drafts into composer", "traveler." in page.input_value("#input"))
        page.wait_for_function("!document.querySelector('#input').disabled", timeout=8000)  # draft finished, composer re-enabled
        page.fill("#input", "")

        # open settings drawer
        page.click("#settings-btn")
        page.wait_for_selector("#settings-modal:not(.hidden)", timeout=3000)
        check("UI: settings drawer opens", page.locator("#s-ctx").count() == 1)
        check("UI: settings num_ctx populated", int(page.input_value("#s-ctx")) >= 4096)
        check("UI: inner-monologue + scene controls", page.locator("#s-inner").count() == 1 and page.locator("#s-scene").count() == 1)
        check("UI: banned-phrases field present", page.locator("#s-ban").count() == 1)
        # sampler presets: save the current sliders as a named, reusable preset
        check("UI: sampler-preset controls present",
              page.locator("#s-preset-list").count() == 1 and page.locator("#s-preset-save").count() == 1)
        page.fill("#s-preset-name", "My Feel")
        page.click("#s-preset-save")
        page.wait_for_function("[...document.querySelectorAll('#s-preset-list option')].some(o => o.textContent === 'My Feel')", timeout=4000)
        check("UI: saved sampler preset appears in list",
              page.evaluate("[...document.querySelectorAll('#s-preset-list option')].some(o => o.textContent === 'My Feel')"))
        # Escape closes the top modal (keyboard-shortcut handler)
        page.keyboard.press("Escape")
        page.wait_for_function("document.querySelector('#settings-modal').classList.contains('hidden')", timeout=3000)
        check("UI: Escape closes a modal", page.locator("#settings-modal.hidden").count() == 1)

        # ---- tags, scene background, pin-to-top ----
        png_1x1 = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
        bg_path = TMP / "bg.png"; bg_path.write_bytes(png_1x1)

        page.click("#settings-btn")
        page.wait_for_selector("#settings-modal:not(.hidden)", timeout=3000)
        check("UI: tags field present", page.locator("#s-tags").count() == 1)
        check("UI: auto-memory toggle present", page.locator("#s-automem").count() == 1)
        check("UI: auto-relationship toggle present", page.locator("#s-autorel").count() == 1)
        check("UI: scene-background controls present",
              page.locator("#s-bg-preview").count() == 1 and page.locator("#s-bg-upload").count() == 1
              and page.locator("#s-bg-clear").count() == 1)

        # upload a scene background -> it paints behind the chat, with a preview
        page.set_input_files("#s-bg-file", str(bg_path))
        page.wait_for_function("document.querySelector('#messages').classList.contains('has-bg')", timeout=6000)
        check("UI: scene background applied to chat", page.locator("#messages.has-bg").count() == 1)
        check("UI: background preview shows image", page.locator("#s-bg-preview img").count() == 1)
        page.click("#s-bg-clear")
        page.wait_for_function("!document.querySelector('#messages').classList.contains('has-bg')", timeout=6000)
        check("UI: scene background cleared", page.locator("#messages.has-bg").count() == 0)

        # tag the story, save, and see the sidebar tag bar + chips appear
        page.fill("#s-tags", "fantasy, test-tag")
        page.click("#settings-save")
        page.wait_for_function("document.querySelector('#settings-modal').classList.contains('hidden')", timeout=3000)
        page.wait_for_function("!document.querySelector('#story-tags').classList.contains('hidden')", timeout=4000)
        labels = page.evaluate("[...document.querySelectorAll('#story-tags .tag-chip')].map(c => c.textContent)")
        check("UI: tag bar shows the new tags", "fantasy" in labels and "test-tag" in labels, str(labels))

        # pin the active story to the top (star toggles 'Pin to top' -> 'Unpin')
        spin = page.locator("#story-list .list-item .act[title='Pin to top']").first
        check("UI: story pin button present", spin.count() == 1)
        spin.click()
        page.wait_for_function("!!document.querySelector(\"#story-list .list-item .act[title='Unpin']\")", timeout=4000)
        check("UI: story pins to top", page.locator("#story-list .list-item .act[title='Unpin']").count() >= 1)
        check("UI: character pin button present",
              page.locator("#character-list .list-item .act[title='Pin to top']").count() >= 1)

        # ---- memories: open drawer, add a fact, pin it ----
        check("UI: memory button present", page.locator("#memory-btn").count() == 1)
        page.click("#memory-btn")
        page.wait_for_selector("#memory-modal:not(.hidden)", timeout=3000)
        check("UI: memory drawer opens", page.locator("#mem-story-list").count() == 1)
        check("UI: character memory section present", page.locator("#mem-char-section").count() == 1)
        page.fill("#mem-add-input", "Kael distrusts the royal guard.")
        page.click("#mem-add-btn")
        page.wait_for_function(
            "[...document.querySelectorAll('#mem-story-list .mem-text')].some(e => e.textContent.includes('royal guard'))",
            timeout=4000)
        check("UI: memory added + listed",
              page.evaluate("[...document.querySelectorAll('#mem-story-list .mem-text')].some(e => e.textContent.includes('royal guard'))"))
        page.locator("#mem-story-list .mem-row .act.pin").first.click()
        page.wait_for_function("!!document.querySelector('#mem-story-list .mem-row.pinned')", timeout=4000)
        check("UI: memory can be pinned", page.locator("#mem-story-list .mem-row.pinned").count() >= 1)
        page.keyboard.press("Escape")
        page.wait_for_function("document.querySelector('#memory-modal').classList.contains('hidden')", timeout=3000)
        check("UI: Escape closes memory drawer", page.locator("#memory-modal.hidden").count() == 1)

        # ---- relationship badge + editor ----
        check("UI: relationship badge visible", page.locator("#rel-badge:not(.hidden)").count() == 1)
        page.click("#rel-badge")
        page.wait_for_selector("#rel-modal:not(.hidden)", timeout=3000)
        check("UI: relationship modal opens", page.locator("#rel-aff").count() == 1)
        page.fill("#rel-status", "test allies")
        page.evaluate("const s=document.querySelector('#rel-aff'); s.value=60; s.dispatchEvent(new Event('input'));")
        check("UI: band label updates with the slider",
              ("attached" in page.inner_text("#rel-band")) or ("warm" in page.inner_text("#rel-band")),
              page.inner_text("#rel-band"))
        page.click("#rel-save")
        page.wait_for_function("document.querySelector('#rel-modal').classList.contains('hidden')", timeout=3000)
        page.wait_for_function(
            "document.querySelector('#rel-badge') && document.querySelector('#rel-badge').textContent.includes('test allies')",
            timeout=4000)
        check("UI: badge reflects the saved relationship", "test allies" in page.inner_text("#rel-badge"))

        # character editor opens with advanced fields
        page.click("#edit-active-char")
        page.wait_for_selector("#char-modal:not(.hidden)", timeout=3000)
        check("UI: char editor has example field", page.locator("#c-example").count() == 1)
        check("UI: char editor has profile fields", page.locator("#c-career").count() == 1)
        check("UI: char default-settings fields",
              page.locator("#c-def-temp").count() == 1 and page.locator("#c-def-pred").count() == 1)
        check("UI: permanent-token badge computed", "tokens" in page.inner_text("#perm-tokens"))
        # character generator fills the editor fields from a concept
        page.fill("#c-concept", "a fire mage")
        page.click("#c-generate")
        page.wait_for_function("document.querySelector('#c-name').value === 'Ember'", timeout=8000)
        check("UI: character generator fills fields",
              page.input_value("#c-name") == "Ember" and page.input_value("#c-greeting") != "")
        page.click("#char-cancel")
        check("UI: mood selector present", page.locator("#mood-select").count() == 1)
        check("UI: inner-thought rendering", page.evaluate("formatContent('((I am afraid.))').includes('thought')"))

        # new Batch-A controls
        check("UI: backup + import-story buttons present",
              page.locator("#backup-btn").count() == 1 and page.locator("#import-story-btn").count() == 1)
        if page.locator("#tts-btn").is_visible():
            before = page.inner_text("#tts-btn")
            page.click("#tts-btn")
            check("UI: TTS toggle changes label", page.inner_text("#tts-btn") != before)

        # theme toggle, story search, and markdown transcript export
        page.click("#theme-btn")
        check("UI: theme toggle applies a theme",
              page.evaluate("document.documentElement.getAttribute('data-theme')") in ("dark", "light"))
        check("UI: story search present", page.locator("#story-search").count() == 1)
        check("UI: transcript export button present",
              page.locator("#story-list [title='Export transcript (.md)']").count() >= 1)

        # group scene creation
        page.click("#new-group-btn")
        page.wait_for_selector("#group-modal:not(.hidden)", timeout=3000)
        boxes = page.locator("#group-pick input[type=checkbox]")
        check("UI: group picker lists characters", boxes.count() >= 2, str(boxes.count()))
        boxes.nth(0).check(); boxes.nth(1).check()
        page.click("#group-create")
        page.wait_for_selector("#group-bar:not(.hidden)", timeout=3000)
        check("UI: group bar shows member chips", page.locator("#group-bar .group-chip").count() >= 2)
        check("UI: group speaker labels render", page.locator(".msg .speaker").count() >= 1)

        page.screenshot(path=str(TMP / "ui.png"), full_page=True)
        browser.close()

    check("UI: no JS console errors", not errors, str(errors))
    print()
    if FAILED:
        print(f"❌ {len(FAILED)} UI check(s) failed: {FAILED}")
        sys.stdout.flush()
        os._exit(1)
    print("✅ All UI checks passed.")
    sys.stdout.flush()
    # Playwright/Chromium + the daemon uvicorn threads can segfault during Python
    # interpreter teardown in CI *after* every check has passed (exit 139). The
    # test result is already decided here, so exit hard with the real status and
    # skip the flaky native teardown — a teardown crash must not fail a green run.
    os._exit(0)


if __name__ == "__main__":
    main()
