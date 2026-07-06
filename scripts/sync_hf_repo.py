#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import shutil
import tempfile
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


def sync_hf_subdir(
    repo: str,
    revision: str,
    token: str | None,
    repo_subdir: str,
    destination_dir: Path,
    force: bool,
) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sync-hf-subdir-") as temp_dir:
        temp_root = Path(temp_dir)
        cmd = [
            "hf",
            "download",
            repo,
            "--repo-type",
            "model",
            "--revision",
            revision,
            "--local-dir",
            str(temp_root),
            "--include",
            f"{repo_subdir}/*",
            "--include",
            f"{repo_subdir}/**",
        ]
        if token:
            cmd.extend(["--token", token])
        if force:
            cmd.append("--force-download")

        subprocess.run(cmd, check=True)

        source_dir = temp_root / repo_subdir
        if not source_dir.exists():
            raise SystemExit(f"Downloaded directory not found: {source_dir}")

        copied = 0
        for item in source_dir.iterdir():
            destination = destination_dir / item.name
            if destination.exists():
                if force:
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                else:
                    continue
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=force)
            else:
                shutil.copy2(item, destination)
            copied += 1
    return copied


def sync_hf_files(
    repo: str,
    revision: str,
    token: str | None,
    repo_paths: list[str],
    destination_dir: Path,
    force: bool,
) -> tuple[int, int]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sync-hf-files-") as temp_dir:
        temp_root = Path(temp_dir)
        cmd = [
            "hf",
            "download",
            repo,
            "--repo-type",
            "model",
            "--revision",
            revision,
            "--local-dir",
            str(temp_root),
        ]
        for repo_path in repo_paths:
            cmd.extend(["--include", repo_path])
        if token:
            cmd.extend(["--token", token])
        if force:
            cmd.append("--force-download")

        subprocess.run(cmd, check=True)

        copied = 0
        skipped = 0
        for repo_path in repo_paths:
            source = temp_root / repo_path
            if not source.exists():
                raise SystemExit(f"Downloaded file not found: {source}")
            destination = destination_dir / Path(repo_path).name
            if destination.exists():
                if force:
                    destination.unlink()
                else:
                    skipped += 1
                    continue
            shutil.copy2(source, destination)
            copied += 1
    return copied, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-eye-loras", action="store_true")
    parser.add_argument("--eye-lora-repo", default="JujoHotaru/lora")
    parser.add_argument("--eye-lora-revision", default="main")
    parser.add_argument("--eye-lora-subdir", default="sdxl/eyecollexl")
    parser.add_argument("--eye-lora-target", default="loras/eyes")
    parser.add_argument("--download-smoothmix", action="store_true")
    parser.add_argument("--smoothmix-repo", default="Babaladen/SmoothMIX")
    parser.add_argument("--smoothmix-revision", default="main")
    parser.add_argument("--smoothmix-target", default="diffusion_models")
    parser.add_argument("--download-wan22-fp8-scaled", action="store_true")
    parser.add_argument("--wan22-fp8-scaled-repo", default="Comfy-Org/Wan_2.2_ComfyUI_Repackaged")
    parser.add_argument("--wan22-fp8-scaled-revision", default="main")
    parser.add_argument("--wan22-fp8-scaled-target", default="diffusion_models")
    parser.add_argument("--download-wan22-lightx2v-4steps", action="store_true")
    parser.add_argument("--wan22-lightx2v-4steps-repo", default="Comfy-Org/Wan_2.2_ComfyUI_Repackaged")
    parser.add_argument("--wan22-lightx2v-4steps-revision", default="main")
    parser.add_argument("--wan22-lightx2v-4steps-target", default="loras/Fast")
    parser.add_argument("--download-wan22-lightx2v-260412", action="store_true")
    parser.add_argument("--wan22-lightx2v-260412-repo", default="Kijai/WanVideo_comfy")
    parser.add_argument("--wan22-lightx2v-260412-revision", default="main")
    parser.add_argument("--wan22-lightx2v-260412-target", default="loras/Fast")
    parser.add_argument("--download-wan22-clipvision", action="store_true")
    parser.add_argument("--wan22-clipvision-repo", default="ricecake/wan21NSFWClipVisionH_v10")
    parser.add_argument("--wan22-clipvision-revision", default="main")
    parser.add_argument("--wan22-clipvision-target", default="clip_vision")
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

    if args.download_eye_loras and args.mode == "image":
        eye_target = Path(args.model_root) / args.eye_lora_target
        copied = sync_hf_subdir(
            repo=args.eye_lora_repo,
            revision=args.eye_lora_revision,
            token=token,
            repo_subdir=args.eye_lora_subdir,
            destination_dir=eye_target,
            force=args.force,
        )
        print(f"Synced {copied} eye lora items into: {eye_target}")

    if args.download_smoothmix and args.mode == "video":
        smoothmix_target = Path(args.model_root) / args.smoothmix_target
        copied, skipped = sync_hf_files(
            repo=args.smoothmix_repo,
            revision=args.smoothmix_revision,
            token=token,
            repo_paths=[
                "smoothMixWan2214BI2V_i2vV20High.safetensors",
                "smoothMixWan2214BI2V_i2vV20Low.safetensors",
            ],
            destination_dir=smoothmix_target,
            force=args.force,
        )
        print(f"Synced {copied} SmoothMIX files into: {smoothmix_target} (skipped {skipped})")

    if args.download_wan22_fp8_scaled and args.mode == "video":
        fp8_scaled_target = Path(args.model_root) / args.wan22_fp8_scaled_target
        copied, skipped = sync_hf_files(
            repo=args.wan22_fp8_scaled_repo,
            revision=args.wan22_fp8_scaled_revision,
            token=token,
            repo_paths=[
                "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            ],
            destination_dir=fp8_scaled_target,
            force=args.force,
        )
        print(f"Synced {copied} Wan2.2 fp8_scaled 14B files into: {fp8_scaled_target} (skipped {skipped})")

    if args.download_wan22_lightx2v_4steps and args.mode == "video":
        lightx2v_target = Path(args.model_root) / args.wan22_lightx2v_4steps_target
        copied, skipped = sync_hf_files(
            repo=args.wan22_lightx2v_4steps_repo,
            revision=args.wan22_lightx2v_4steps_revision,
            token=token,
            repo_paths=[
                "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            ],
            destination_dir=lightx2v_target,
            force=args.force,
        )
        print(f"Synced {copied} Wan2.2 lightx2v 4steps LoRA files into: {lightx2v_target} (skipped {skipped})")

    if args.download_wan22_lightx2v_260412 and args.mode == "video":
        lightx2v_260412_target = Path(args.model_root) / args.wan22_lightx2v_260412_target
        copied, skipped = sync_hf_files(
            repo=args.wan22_lightx2v_260412_repo,
            revision=args.wan22_lightx2v_260412_revision,
            token=token,
            repo_paths=[
                "LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_260412_rank_256_fp16.safetensors",
                "LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_LOW_lightx2v_4step_lora_260412_rank_256_fp16.safetensors",
            ],
            destination_dir=lightx2v_260412_target,
            force=args.force,
        )
        print(f"Synced {copied} Wan2.2 lightx2v 260412 LoRA files into: {lightx2v_260412_target} (skipped {skipped})")

    if args.download_wan22_clipvision and args.mode == "video":
        clipvision_target = Path(args.model_root) / args.wan22_clipvision_target
        copied, skipped = sync_hf_files(
            repo=args.wan22_clipvision_repo,
            revision=args.wan22_clipvision_revision,
            token=token,
            repo_paths=["wan21NSFWClipVisionH_v10.safetensors"],
            destination_dir=clipvision_target,
            force=args.force,
        )
        print(f"Synced {copied} Wan2.2 clip vision files into: {clipvision_target} (skipped {skipped})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
