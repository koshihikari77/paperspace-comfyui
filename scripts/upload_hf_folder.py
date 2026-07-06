#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = value.strip()
    return token or None


def iter_files(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root)).replace(os.sep, "/")
        for path in root.rglob("*")
        if path.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a local folder into a Hugging Face model repo path."
    )
    parser.add_argument("--repo", required=True, help="Target Hugging Face repo id.")
    parser.add_argument("--local-dir", required=True, help="Local folder to upload.")
    parser.add_argument("--path-in-repo", required=True, help="Destination path inside the repo.")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default="main", help="Target branch or revision.")
    parser.add_argument("--message", default="Upload folder from local workspace", help="Commit message.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Env var containing the HF token.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete remote files under path-in-repo that are not present locally.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the upload plan without uploading.")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    if not local_dir.is_dir():
        raise SystemExit(f"Local directory not found: {local_dir}")

    files = iter_files(local_dir)
    if not files:
        raise SystemExit(f"No files found under: {local_dir}")

    print(f"Repo: {args.repo}")
    print(f"Revision: {args.revision}")
    print(f"Local dir: {local_dir}")
    print(f"Path in repo: {args.path_in_repo}")
    print(f"Files: {len(files)}")

    if args.dry_run:
        for relative_path in files:
            print(relative_path)
        return 0

    token = normalize_token(os.environ.get(args.token_env))
    api = HfApi(token=token or None)
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type=args.repo_type,
        folder_path=str(local_dir),
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.message,
        delete_patterns=[f"{args.path_in_repo}/**"] if args.delete else None,
        ignore_patterns=[".cache/**", ".cache", "**/.DS_Store"],
    )
    print(f"Uploaded commit: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
