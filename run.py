"""Launch the AI Roleplayer web app locally.

Run with:  python run.py
Then open: http://localhost:8000
"""
import os
import webbrowser
from threading import Timer

import uvicorn

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


def _open_browser():
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    print("\n🎭  AI Roleplayer")
    print(f"    Open your browser at:  http://localhost:{PORT}")
    print("    Press Ctrl+C to stop.\n")
    # Open the browser a moment after the server starts.
    Timer(1.2, _open_browser).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
