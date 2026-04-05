#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
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


def include_patterns(mode: str) -> list[str]:
    dirs = sorted(dir_name for dir_name in SUPPORTED_DIRS if should_include(dir_name, mode))
    patterns: list[str] = []
    for dir_name in dirs:
        patterns.extend([f"{dir_name}/*", f"{dir_name}/**"])
    return patterns


def list_repo_files(repo: str, revision: str, token: str | None, mode: str) -> list[str]:
    api = HfApi(token=token or None)
    selected: list[str] = []

    for path in api.list_repo_files(repo_id=repo, repo_type="model", revision=revision):
        parts = PurePosixPath(path).parts
        if len(parts) < 2:
            continue

        top_level_dir = parts[0]
        if should_include(top_level_dir, mode):
            selected.append(path)

    return selected


def sync_repo_subset(repo: str, revision: str, token: str | None, model_root: Path, mode: str, force: bool) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        repo,
        "--repo-type",
        "model",
        "--revision",
        revision,
        "--local-dir",
        str(model_root),
    ]
    if token:
        cmd.extend(["--token", token])
    if force:
        cmd.append("--force-download")
    for pattern in include_patterns(mode):
        cmd.extend(["--include", pattern])

    subprocess.run(cmd, check=True)


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
    print(f"Syncing into: {args.model_root}")
    sync_repo_subset(repo, revision, token, Path(args.model_root), args.mode, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
