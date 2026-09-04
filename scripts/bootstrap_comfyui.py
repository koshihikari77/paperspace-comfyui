#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("$", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def resolve_comfyui_python(comfyui_dir: Path, configured: str | None) -> Path:
    candidates = [Path(configured)] if configured else []
    candidates.extend([comfyui_dir / ".venv/bin/python", comfyui_dir / "venv/bin/python"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    system_python = shutil.which("python")
    if system_python:
        return Path(system_python)
    raise SystemExit("No usable Python executable found for ComfyUI")


def is_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def ensure_sageattention(comfyui_python: Path, requirements: Path, env: dict[str, str]) -> None:
    check = subprocess.run(
        [str(comfyui_python), "-c", "import sageattention"],
        env=env,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        print("SageAttention: ready", flush=True)
        return
    run([str(comfyui_python), "-m", "pip", "install", "-r", str(requirements)], env)
    run([str(comfyui_python), "-c", "import sageattention; print('SageAttention: ready')"], env)


def write_active_workflow(source: Path, destination: Path, preset: str | None) -> Path:
    workflow = json.loads(source.read_text(encoding="utf-8"))
    if preset:
        workflow["112:118"]["inputs"]["lora"] = f"Nsfw/{preset}-H.safetensors"
        workflow["112:119"]["inputs"]["lora"] = f"Nsfw/{preset}-L.safetensors"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def launch_comfyui(
    comfyui_dir: Path,
    comfyui_python: Path,
    port: int,
    extra_args: str,
    env: dict[str, str],
) -> str:
    if is_ready(port):
        return "already-running"

    log_file = comfyui_dir / "user/logs/comfyui.log"
    pid_file = Path(f"/tmp/paperspace-comfyui-{port}.pid")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(comfyui_python),
        str(comfyui_dir / "main.py"),
        "--listen",
        "0.0.0.0",
        "--port",
        str(port),
        *shlex.split(extra_args),
    ]
    print("$", shlex.join(cmd), flush=True)
    with log_file.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=comfyui_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if is_ready(port):
            return "started"
        if process.poll() is not None:
            raise SystemExit(f"ComfyUI exited with status {process.returncode}; see {log_file}")
        time.sleep(2)
    raise SystemExit(f"ComfyUI did not become ready within 120 seconds; see {log_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare models and start Paperspace ComfyUI.")
    parser.add_argument("--repo-root", default="/notebooks")
    parser.add_argument("--comfyui-dir", default="/storage/ComfyUI")
    parser.add_argument("--model-root", default="/app/models")
    parser.add_argument("--hf-home", default="/storage/.cache/huggingface")
    parser.add_argument("--hf-repo-config", default="/notebooks/hf-repo.yaml")
    parser.add_argument("--download-image-loras", action="store_true")
    parser.add_argument("--download-eye-loras", action="store_true")
    parser.add_argument("--download-wan22-models", action="store_true")
    parser.add_argument("--wan22-lora-preset", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--comfyui-args", default="--preview-method auto")
    parser.add_argument("--comfyui-python")
    parser.add_argument("--paperspace-fqdn", default=os.environ.get("PAPERSPACE_FQDN", ""))
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    scripts = repo_root / "scripts"
    comfyui_dir = Path(args.comfyui_dir)
    model_root = Path(args.model_root)
    env = os.environ.copy()
    env["HF_HOME"] = args.hf_home

    run(
        [
            sys.executable,
            str(scripts / "check_runtime.py"),
            "--comfyui-dir",
            str(comfyui_dir),
            "--model-root",
            str(model_root),
        ],
        env,
    )

    downloads_enabled = (
        args.download_image_loras
        or args.download_eye_loras
        or args.download_wan22_models
        or bool(args.wan22_lora_preset)
    )
    if downloads_enabled:
        run([sys.executable, str(scripts / "check_hf_auth.py")], env)

    if args.download_image_loras:
        cmd = [
            sys.executable,
            str(scripts / "sync_hf_repo.py"),
            "--config",
            args.hf_repo_config,
            "--model-root",
            str(model_root),
            "--mode",
            "image",
        ]
        if args.force:
            cmd.append("--force")
        run(cmd, env)

    download_cmd = [
        sys.executable,
        str(scripts / "download_easywan22.py"),
        "--model-root",
        str(model_root),
        "--max-workers",
        str(args.max_workers),
    ]
    selected = False
    if args.download_eye_loras:
        download_cmd.extend(["--group", "eye-loras"])
        selected = True
    if args.download_wan22_models:
        download_cmd.extend(["--group", "floyo-wan22-core"])
        selected = True
    for preset in args.wan22_lora_preset:
        download_cmd.extend(["--wan22-lora-preset", preset])
        selected = True
    if args.force:
        download_cmd.append("--force")
    if selected:
        run(download_cmd, env)

    run(
        [
            sys.executable,
            str(scripts / "write_extra_model_paths.py"),
            "--comfyui-dir",
            str(comfyui_dir),
            "--model-root",
            str(model_root),
        ],
        env,
    )

    comfyui_python = resolve_comfyui_python(comfyui_dir, args.comfyui_python)
    if args.download_wan22_models:
        ensure_sageattention(comfyui_python, repo_root / "comfyui-requirements.txt", env)

    source_workflow = repo_root / "workflows/floyo_wanvideowrapper_i2v.json"
    active_workflow = write_active_workflow(
        source_workflow,
        Path("/app/workflows/floyo_wanvideowrapper_i2v_active.json"),
        args.wan22_lora_preset[0] if args.wan22_lora_preset else None,
    )
    status = launch_comfyui(comfyui_dir, comfyui_python, args.port, args.comfyui_args, env)

    local_url = f"http://127.0.0.1:{args.port}"
    public_url = f"https://tensorboard-{args.paperspace_fqdn}" if args.paperspace_fqdn else ""
    print(f"COMFYUI_STATUS=ready ({status})", flush=True)
    print(f"COMFYUI_LOCAL_URL={local_url}", flush=True)
    print(f"COMFYUI_URL={public_url or 'PAPERSPACE_FQDN is not set'}", flush=True)
    print(f"COMFYUI_WORKFLOW={active_workflow}", flush=True)
    print(f"COMFYUI_LOG={comfyui_dir / 'user/logs/comfyui.log'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
