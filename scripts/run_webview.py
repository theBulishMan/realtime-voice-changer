from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.desktop import launch_webview_shell

    parser = argparse.ArgumentParser(description="Launch Realtime Voice Changer in desktop webview mode.")
    parser.add_argument("--url", default="http://127.0.0.1:8787", help="Backend URL to open in webview.")
    parser.add_argument("--width", type=int, default=1360, help="Window width.")
    parser.add_argument("--height", type=int, default=860, help="Window height.")
    args = parser.parse_args()
    launch_webview_shell(base_url=args.url, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
