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
from datetime import datetime, timezone
from pathlib import Path


def print_status(name: str, state: str, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    print(f"{name}={state}{suffix}", flush=True)


def run_step(
    name: str,
    cmd: list[str],
    env: dict[str, str],
    log_file: Path,
) -> None:
    print_status(name, "running")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {name}\n")
        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print_status(name, "failed", f"see {log_file}")
        raise SystemExit(f"{name} failed; see {log_file}")
    print_status(name, "complete")


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


def ensure_sageattention(
    comfyui_python: Path,
    requirements: Path,
    env: dict[str, str],
    log_file: Path,
) -> None:
    check = subprocess.run(
        [str(comfyui_python), "-c", "import sageattention"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode == 0:
        print_status("SAGEATTENTION", "complete", "already installed")
        return
    run_step(
        "SAGEATTENTION",
        [str(comfyui_python), "-m", "pip", "install", "-r", str(requirements)],
        env,
        log_file,
    )


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
    startup_timeout: int,
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

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if is_ready(port):
            return "started"
        if process.poll() is not None:
            raise SystemExit(f"ComfyUI exited with status {process.returncode}; see {log_file}")
        time.sleep(2)
    if process.poll() is None:
        return "starting"
    raise SystemExit(f"ComfyUI exited with status {process.returncode}; see {log_file}")


def downloader_base(args: argparse.Namespace, scripts: Path, model_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(scripts / "download_easywan22.py"),
        "--model-root",
        str(model_root),
        "--max-workers",
        str(args.max_workers),
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def prepare(args: argparse.Namespace, env: dict[str, str]) -> Path:
    repo_root = Path(args.repo_root)
    scripts = repo_root / "scripts"
    comfyui_dir = Path(args.comfyui_dir)
    model_root = Path(args.model_root)
    setup_log = Path(args.setup_log)

    run_step(
        "RUNTIME_CHECK",
        [
            sys.executable,
            str(scripts / "check_runtime.py"),
            "--comfyui-dir",
            str(comfyui_dir),
            "--model-root",
            str(model_root),
        ],
        env,
        setup_log,
    )

    downloads_enabled = (
        args.download_image_loras
        or args.download_eye_loras
        or args.download_wan22_models
        or args.download_wan22_nsfw_loras
        or bool(args.wan22_lora_preset)
    )
    if downloads_enabled:
        run_step("HF_AUTH", [sys.executable, str(scripts / "check_hf_auth.py")], env, setup_log)
    else:
        print_status("HF_AUTH", "skipped", "no downloads selected")

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
        run_step("DOWNLOAD_IMAGE_LORAS", cmd, env, setup_log)
    else:
        print_status("DOWNLOAD_IMAGE_LORAS", "skipped", "disabled")

    if args.download_eye_loras:
        run_step(
            "DOWNLOAD_EYE_LORAS",
            [*downloader_base(args, scripts, model_root), "--group", "eye-loras"],
            env,
            setup_log,
        )
    else:
        print_status("DOWNLOAD_EYE_LORAS", "skipped", "disabled")

    if args.download_wan22_models:
        run_step(
            "DOWNLOAD_WAN22_MODELS",
            [*downloader_base(args, scripts, model_root), "--group", "floyo-wan22-core"],
            env,
            setup_log,
        )
    else:
        print_status("DOWNLOAD_WAN22_MODELS", "skipped", "disabled")

    if args.download_wan22_nsfw_loras:
        run_step(
            "DOWNLOAD_WAN22_NSFW_LORAS",
            [*downloader_base(args, scripts, model_root), "--group", "wan22-nsfw-loras"],
            env,
            setup_log,
        )
    else:
        print_status("DOWNLOAD_WAN22_NSFW_LORAS", "skipped", "disabled")

    if args.wan22_lora_preset:
        if args.download_wan22_nsfw_loras:
            print_status("WAN22_LORA_PRESETS", "complete", "workflow selection; full bundle enabled")
        else:
            cmd = downloader_base(args, scripts, model_root)
            for preset in args.wan22_lora_preset:
                cmd.extend(["--wan22-lora-preset", preset])
            run_step(
                "WAN22_LORA_PRESETS",
                cmd,
                env,
                setup_log,
            )
    else:
        print_status("WAN22_LORA_PRESETS", "skipped", "empty")

    run_step(
        "EXTERNAL_MODEL_PATHS",
        [
            sys.executable,
            str(scripts / "write_extra_model_paths.py"),
            "--comfyui-dir",
            str(comfyui_dir),
            "--model-root",
            str(model_root),
        ],
        env,
        setup_log,
    )

    comfyui_python = resolve_comfyui_python(comfyui_dir, args.comfyui_python)
    if args.download_wan22_models:
        ensure_sageattention(
            comfyui_python,
            repo_root / "comfyui-requirements.txt",
            env,
            setup_log,
        )
    else:
        print_status("SAGEATTENTION", "skipped", "Wan 2.2 disabled")

    active_workflow = write_active_workflow(
        repo_root / "workflows/floyo_wanvideowrapper_i2v.json",
        Path("/app/workflows/floyo_wanvideowrapper_i2v_active.json"),
        args.wan22_lora_preset[0] if args.wan22_lora_preset else None,
    )
    print_status("ACTIVE_WORKFLOW", "complete", str(active_workflow))
    print_status("PREPARE_STATUS", "complete", f"details: {setup_log}")
    return active_workflow


def start(args: argparse.Namespace, env: dict[str, str]) -> None:
    comfyui_dir = Path(args.comfyui_dir)
    comfyui_python = resolve_comfyui_python(comfyui_dir, args.comfyui_python)
    print_status("COMFYUI_PROCESS", "running")
    status = launch_comfyui(
        comfyui_dir,
        comfyui_python,
        args.port,
        args.comfyui_args,
        env,
        args.startup_timeout,
    )

    public_url = f"https://tensorboard-{args.paperspace_fqdn}" if args.paperspace_fqdn else ""
    print_status("COMFYUI_PROCESS", status)
    print(f"COMFYUI_STATUS={'starting' if status == 'starting' else 'ready'}", flush=True)
    print(f"COMFYUI_LOCAL_URL=http://127.0.0.1:{args.port}", flush=True)
    print(f"COMFYUI_URL={public_url or 'PAPERSPACE_FQDN is not set'}", flush=True)
    print("COMFYUI_WORKFLOW=/app/workflows/floyo_wanvideowrapper_i2v_active.json", flush=True)
    print(f"COMFYUI_LOG={comfyui_dir / 'user/logs/comfyui.log'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or start Paperspace ComfyUI.")
    parser.add_argument("--phase", choices=["prepare", "start", "all"], default="all")
    parser.add_argument("--repo-root", default="/notebooks")
    parser.add_argument("--comfyui-dir", default="/storage/ComfyUI")
    parser.add_argument("--model-root", default="/app/models")
    parser.add_argument("--hf-home", default="/storage/.cache/huggingface")
    parser.add_argument("--hf-repo-config", default="/notebooks/hf-repo.yaml")
    parser.add_argument("--setup-log", default="/storage/ComfyUI/user/logs/bootstrap.log")
    parser.add_argument("--download-image-loras", action="store_true")
    parser.add_argument("--download-eye-loras", action="store_true")
    parser.add_argument("--download-wan22-models", action="store_true")
    parser.add_argument("--download-wan22-nsfw-loras", action="store_true")
    parser.add_argument("--wan22-lora-preset", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--comfyui-args", default="--preview-method auto")
    parser.add_argument("--comfyui-python")
    parser.add_argument("--paperspace-fqdn", default=os.environ.get("PAPERSPACE_FQDN", ""))
    args = parser.parse_args()

    env = os.environ.copy()
    env["HF_HOME"] = args.hf_home
    if args.phase in {"prepare", "all"}:
        prepare(args, env)
    if args.phase in {"start", "all"}:
        start(args, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
