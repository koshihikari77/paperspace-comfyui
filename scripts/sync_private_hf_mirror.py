#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from download_easywan22 import GROUP_ASSETS, PRIVATE_MIRROR_REPO, iter_assets

DEFAULT_PRUNE_PATHS = [
    "diffusion_models/smoothMixWan2214BI2V_i2vV20High.safetensors",
    "diffusion_models/smoothMixWan2214BI2V_i2vV20Low.safetensors",
]


def normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = value.strip()
    return token or None


def collect_mirrored_files(private_repo: str) -> set[str]:
    mirrored_files: set[str] = set()

    for _, asset in iter_assets(sorted(GROUP_ASSETS)):
        if asset.source == "hf_mirror_file" and asset.repo_id == private_repo:
            mirrored_files.add(asset.relative_path)
    return mirrored_files


def copy_mirrored_files(source_root: Path, private_root: Path, mirrored_files: set[str]) -> tuple[list[str], list[str]]:
    copied: list[str] = []
    missing: list[str] = []

    for relative_path in sorted(mirrored_files):
        src = source_root / relative_path
        dest = private_root / relative_path
        if not src.exists():
            missing.append(relative_path)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(relative_path)

    return copied, missing


def prune_paths(private_root: Path, relative_paths: list[str]) -> list[str]:
    removed: list[str] = []

    for relative_path in sorted(set(relative_paths)):
        target = private_root / relative_path
        if not target.exists():
            continue
        if target.is_dir():
            for file_path in sorted(path for path in target.rglob("*") if path.is_file()):
                removed.append(str(file_path.relative_to(private_root)).replace(os.sep, "/"))
            shutil.rmtree(target)
        else:
            target.unlink()
            removed.append(relative_path)

    return removed


def remove_empty_dirs(root: Path) -> None:
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            continue


def upload_private_root(
    repo_id: str,
    private_root: Path,
    token: str | None,
    delete_patterns: list[str],
) -> None:
    api = HfApi(token=token or None)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=private_root,
        path_in_repo="",
        ignore_patterns=[".cache/**", ".cache", "**/.DS_Store"],
        delete_patterns=sorted(set(delete_patterns)) or None,
        commit_message="Sync mirrored Civitai assets and prune public-downloadable files",
    )
    print(f"Uploaded commit: {commit.commit_url}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror Civitai-derived EasyWan22 assets into the private HF repo checkout and prune selected redundant files."
    )
    parser.add_argument("--repo", default=PRIVATE_MIRROR_REPO)
    parser.add_argument("--private-root", default="/hfprivate")
    parser.add_argument("--mirror-source-root", default="/storage/ComfyUI/models")
    parser.add_argument("--prune-path", action="append", default=[])
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = normalize_token(os.environ.get("HF_TOKEN"))
    private_root = Path(args.private_root)
    mirror_source_root = Path(args.mirror_source_root)

    mirrored_files = collect_mirrored_files(args.repo)
    prune_paths_list = args.prune_path or list(DEFAULT_PRUNE_PATHS)

    print(f"Private repo: {args.repo}")
    print(f"Private root: {private_root}")
    print(f"Mirror source root: {mirror_source_root}")
    print(f"Mirrored files to copy: {len(mirrored_files)}")
    print(f"Prune paths: {len(prune_paths_list)}")

    if args.dry_run:
        for relative_path in sorted(mirrored_files):
            print(f"[mirror] {relative_path}")
        for relative_path in sorted(prune_paths_list):
            print(f"[prune] {relative_path}")
        return 0

    copied, missing = copy_mirrored_files(mirror_source_root, private_root, mirrored_files)
    removed = prune_paths(private_root, prune_paths_list)
    remove_empty_dirs(private_root)

    print(f"Copied {len(copied)} mirrored files into {private_root}")
    print(f"Removed {len(removed)} redundant repo paths from {private_root}")
    if missing:
        print("Missing local mirror sources:")
        for relative_path in missing:
            print(f"  {relative_path}")

    if args.upload:
        upload_private_root(args.repo, private_root, token, prune_paths_list)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
