#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-dir", required=True)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()

    comfyui_dir = Path(args.comfyui_dir)
    model_root = Path(args.model_root)

    if not shutil.which("hf"):
        raise SystemExit("hf command not found")
    if not comfyui_dir.exists():
        raise SystemExit(f"ComfyUI directory not found: {comfyui_dir}")
    if not (comfyui_dir / "main.py").exists():
        raise SystemExit(f"main.py not found under {comfyui_dir}")

    model_root.mkdir(parents=True, exist_ok=True)

    python_version = subprocess.run(
        ["python", "-c", "import sys; print(sys.version.split()[0])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hf_process = subprocess.run(
        ["hf", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    hf_version = hf_process.stdout.strip() or hf_process.stderr.strip()

    print(f"python {python_version}")
    print(hf_version)
    print(f"MODEL_ROOT ready: {model_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
