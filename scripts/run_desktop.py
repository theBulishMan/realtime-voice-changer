from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    from app.desktop import launch_desktop_app
    launch_desktop_app()


if __name__ == "__main__":
    main()
