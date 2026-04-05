#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


def resolve_comfyui_python(comfyui_dir: Path, comfyui_python: str | None) -> Path:
    candidates = []
    if comfyui_python:
        candidates.append(Path(comfyui_python))
    candidates.extend(
        [
            comfyui_dir / ".venv/bin/python",
            comfyui_dir / "venv/bin/python",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    system_python = shutil.which("python")
    if system_python:
        return Path(system_python)

    raise SystemExit("No usable Python executable found for ComfyUI")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--args", default="")
    parser.add_argument("--python", dest="comfyui_python")
    args = parser.parse_args()

    comfyui_dir = Path(args.comfyui_dir)
    comfyui_python = resolve_comfyui_python(comfyui_dir, args.comfyui_python)

    cmd = [
        str(comfyui_python),
        str(comfyui_dir / "main.py"),
        "--listen",
        "0.0.0.0",
        "--port",
        str(args.port),
    ]
    if args.args.strip():
        cmd.extend(shlex.split(args.args))

    print(f"Using python: {comfyui_python}")
    print("$", shlex.join(cmd))
    subprocess.run(cmd, cwd=comfyui_dir, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
