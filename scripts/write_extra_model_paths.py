#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

MODEL_TYPES = [
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
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-dir", required=True)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()

    comfyui_dir = Path(args.comfyui_dir)
    model_root = Path(args.model_root)
    extra_model_paths_file = comfyui_dir / "extra_model_paths.yaml"
    backup_file = comfyui_dir / "extra_model_paths.yaml.bak.paperspace-comfyui"

    extra_model_paths = {"hf_models": {"base_path": str(model_root)}}
    for model_type in MODEL_TYPES:
        extra_model_paths["hf_models"][model_type] = f"{model_type}/"

    if extra_model_paths_file.exists():
        shutil.copy2(extra_model_paths_file, backup_file)
        print(f"Backed up existing file to {backup_file}")

    with extra_model_paths_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(extra_model_paths, fh, sort_keys=False, allow_unicode=False)

    print(f"Wrote {extra_model_paths_file}")
    print(extra_model_paths_file.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
