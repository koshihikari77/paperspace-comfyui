#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import yaml
from huggingface_hub import HfApi

SUPPORTED_DIRS = {
    "checkpoints",
    "loras",
    "vae",
    "clip",
    "clip_vision",
    "controlnet",
    "embeddings",
    "upscale_models",
    "text_encoders",
    "ipadapter",
    "diffusion_models",
    "unet",
}


def normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = value.strip()
    return token or None


def load_repo_config(config_path: Path) -> tuple[str, str]:
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Config file not found: {config_path}\n"
            "Create it in the cloned repo, or update HF_REPO_CONFIG in the notebook."
        ) from exc

    if config.get("version") != 1:
        raise SystemExit("Config version must be 1")

    repo = config.get("repo")
    revision = config.get("revision", "main")
    if not repo:
        raise SystemExit("Config must contain repo")
    return repo, revision


def should_include(top_level_dir: str, mode: str) -> bool:
    if top_level_dir not in SUPPORTED_DIRS:
        return False
    if mode == "image":
        return top_level_dir == "loras"
    if mode == "video":
        return top_level_dir != "loras"
    raise SystemExit(f"Unsupported mode: {mode}")


def list_repo_files(repo: str, revision: str, token: str | None, mode: str) -> list[tuple[str, str]]:
    api = HfApi(token=token or None)
    selected: list[tuple[str, str]] = []

    for entry in api.list_repo_tree(repo_id=repo, repo_type="model", revision=revision, recursive=True):
        path = getattr(entry, "path", "")
        if not path or path.endswith("/"):
            continue

        parts = PurePosixPath(path).parts
        if len(parts) < 2:
            continue

        top_level_dir = parts[0]
        if should_include(top_level_dir, mode):
            selected.append((top_level_dir, path))

    return selected


def download_file(repo: str, revision: str, token: str | None, source_path: str, target_path: Path, force: bool) -> None:
    if target_path.exists() and not force:
        print(f"skip: {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        repo,
        source_path,
        "--repo-type",
        "model",
        "--revision",
        revision,
    ]
    if token:
        cmd.extend(["--token", token])
    if force:
        cmd.append("--force-download")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    downloaded_path = Path(result.stdout.strip().splitlines()[-1])
    if not downloaded_path.exists():
        raise SystemExit(f"hf download did not return a valid path: {downloaded_path}")

    shutil.copy2(downloaded_path, target_path)
    print(f"saved: {target_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    token = normalize_token(os.environ.get("HF_TOKEN"))

    repo, revision = load_repo_config(Path(args.config))
    files = list_repo_files(repo, revision, token, args.mode)
    if not files:
        raise SystemExit(f"No matching files found for mode={args.mode} in repo={repo}")

    print(f"Repo: {repo}@{revision}")
    print(f"Mode: {args.mode}")
    print(f"Found {len(files)} files")
    for top_level_dir, source_path in files:
        target_path = Path(args.model_root) / top_level_dir / PurePosixPath(source_path).name
        download_file(repo, revision, token, source_path, target_path, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
