from __future__ import annotations

import argparse
import webbrowser

import uvicorn

from app import settings
from app.db import init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Start LoRA-Studio local web app.")
    parser.add_argument("--host", default=settings.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=settings.DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    init_db()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
