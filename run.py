"""Launch the AI Roleplayer web app locally.

Run with:  python run.py
Then open: http://localhost:8000

By default it binds to 127.0.0.1 (this machine only). If you set HOST to a
non-loopback address (e.g. HOST=0.0.0.0 to reach it from your phone), the app
would otherwise be reachable with NO authentication — so a one-time access
token is generated and required. The token is printed below and included in
the URL this script opens.
"""
import os
import secrets
import webbrowser
from threading import Timer

import uvicorn

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
_LOOPBACK = {"127.0.0.1", "localhost", "::1", ""}

# Non-loopback bind with no token set -> mint one so the API isn't wide open.
if HOST not in _LOOPBACK and not os.environ.get("RP_TOKEN"):
    os.environ["RP_TOKEN"] = secrets.token_urlsafe(16)


def _token_query():
    t = os.environ.get("RP_TOKEN")
    return f"?token={t}" if t else ""


def _open_browser():
    try:
        webbrowser.open(f"http://localhost:{PORT}/{_token_query()}")
    except Exception:
        pass


if __name__ == "__main__":
    print("\n🎭  AI Roleplayer")
    token = os.environ.get("RP_TOKEN")
    if token:
        print("    ⚠️  Bound beyond loopback — an access token is required.")
        print(f"    Open this exact URL (keep the token private):")
        print(f"      http://localhost:{PORT}/?token={token}")
    else:
        print(f"    Open your browser at:  http://localhost:{PORT}")
    print("    Press Ctrl+C to stop.\n")
    Timer(1.2, _open_browser).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
