from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build_args(
    *,
    mode: str,
    name: str,
    dist_dir: Path,
    work_dir: Path,
    hook_dir: Path,
) -> list[str]:
    entry = ROOT / "scripts" / "run_webview.py"
    frontend_dir = ROOT / "app" / "frontend"
    add_data = f"{frontend_dir}{';'}app/frontend"

    args = [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--paths",
        str(ROOT),
        "--name",
        name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(work_dir),
        "--additional-hooks-dir",
        str(hook_dir),
        "--add-data",
        add_data,
        "--copy-metadata",
        "webrtcvad-wheels",
        "--hidden-import",
        "app.backend.main",
        "--hidden-import",
        "app.desktop.webview_shell",
        "--hidden-import",
        "_webrtcvad",
        "--collect-submodules",
        "uvicorn",
        "--collect-submodules",
        "webview",
    ]
    if mode == "onefile":
        args.append("--onefile")
    else:
        args.append("--onedir")
    args.append(str(entry))
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Windows desktop package for webview shell.")
    parser.add_argument("--mode", choices=["onedir", "onefile"], default="onedir")
    parser.add_argument("--name", default="RealtimeVoiceChanger")
    parser.add_argument("--dist-dir", default=str(ROOT / "dist"))
    parser.add_argument("--work-dir", default=str(ROOT / "build" / "pyinstaller"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    hook_dir = (ROOT / "scripts" / "pyinstaller_hooks").resolve()
    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    hook_dir.mkdir(parents=True, exist_ok=True)

    pyinstaller_args = _build_args(
        mode=args.mode,
        name=args.name,
        dist_dir=dist_dir,
        work_dir=work_dir,
        hook_dir=hook_dir,
    )
    if args.dry_run:
        print("pyinstaller " + " ".join(pyinstaller_args))
        return

    try:
        import PyInstaller.__main__ as pyinstaller_main
    except Exception as exc:
        raise RuntimeError(
            "PyInstaller is required. Install with: pip install -e \".[packaging]\""
        ) from exc

    pyinstaller_main.run(pyinstaller_args)
    if args.mode == "onedir":
        target = dist_dir / args.name / f"{args.name}.exe"
    else:
        target = dist_dir / f"{args.name}.exe"
    print(f"Build completed: {target}")


if __name__ == "__main__":
    main()
