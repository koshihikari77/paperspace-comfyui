#!/usr/bin/env python3
"""Download pinned MiniMax H3 weights for the unified ComfyUI."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class Asset:
    path: str
    repo: str
    revision: str
    source: str
    size: int
    sha256: str
    url: str = ""
    location: str = "h3"


COMFY_REV = "7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1"
FUSED_REV = "8a8dffaa0cd99c6184833ae0a3b4e9b0089c17b3"
FAST_REV = "ec1e3aa374a91c57b0b94a1623b7e657c0498cf2"

X2_REV = "af8c92d267c6849fec5032c35a65d5766737b338"

ASSETS = {
    "anime-sharp": Asset(
        "upscale_models/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
        "Kim2091/2x-AnimeSharpV4", "1a9339b5c308ab3990f6233be2c1169a75772878",
        "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors", 31359158,
        "b641c9eb10b43f26538177aa8f0fef8b9fc2a153afd1431d0a062a84c49ce6d0",
        location="comfy-models",
    ),
    "rife49": Asset(
        "rife49.pth", "", "", "", 21345274,
        "e55fd00f3cc184e3c65961f4bb827a9da022e78eed36b055242c0ac30000d533",
        url="https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife49.pth",
        location="rife",
    ),
    "x2-vae": Asset(
        "vae/MiniMax-H3-X2-Detail-v1.safetensors",
        "speach1sdef178/MiniMax-H3-X2-Detail-VAE", X2_REV,
        "MiniMax-H3-X2-Detail-v1.safetensors",
        5246877148, "2296840f4acedcaa976688e7d7b97f7bf570b136e400385d3f46224011897aac",
    ),
    "fused": Asset(
        "diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors",
        "MATLOWAI/minimax-h3-fused-turbo-int8-convrot", FUSED_REV,
        "diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors",
        20980178976, "4262e4e9963c553fa00016bbe83961407a4fc0a888be95fd836c8d4f2304e48b",
    ),
    "qwen": Asset(
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "Comfy-Org/MiniMax-H3", COMFY_REV,
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        15687142551, "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
    ),
    "video-vae": Asset(
        "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "Comfy-Org/MiniMax-H3", COMFY_REV,
        "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        2811065184, "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e",
    ),
    "audio-vae": Asset(
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "Comfy-Org/MiniMax-H3", COMFY_REV,
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        605254808, "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    ),
    "official-fl2va": Asset(
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "Comfy-Org/MiniMax-H3", COMFY_REV,
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        20970379616, "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
    ),
    "official-ref2va": Asset(
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "Comfy-Org/MiniMax-H3", COMFY_REV,
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        20970379616, "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
    ),
    "fasth3": Asset(
        "diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
        "FastVideo/FastVideo-FastH3-Comfy", FAST_REV,
        "diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
        22128378696, "0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8",
    ),
}

GROUPS = {
    "postprocess": ("anime-sharp", "rife49"),
    "fused-x2": ("fused", "qwen", "video-vae", "x2-vae"),
    "x2": ("x2-vae",),
    "fused-core": ("fused", "qwen", "video-vae"),
    "audio": ("audio-vae",),
    "official-fl2va": ("official-fl2va", "qwen", "video-vae"),
    "official-ref2va": ("official-ref2va", "qwen", "video-vae"),
    "fasth3": ("fasth3", "qwen", "video-vae"),
}


def destination_for(asset: Asset, model_root: Path, comfyui_dir: Path) -> Path:
    if asset.location == "comfy-models":
        return comfyui_dir / "models" / asset.path
    if asset.location == "rife":
        import yaml
        candidates = [comfyui_dir / "custom_nodes" / name for name in
                      ("comfyui-frame-interpolation", "ComfyUI-Frame-Interpolation")]
        node_dir = next((path for path in candidates if path.exists()), candidates[0])
        config_file = node_dir / "config.yaml"
        checkpoint_root = "./ckpts"
        if config_file.exists():
            checkpoint_root = yaml.safe_load(config_file.read_text())["ckpts_path"]
        return (node_dir / checkpoint_root / "rife" / asset.path).resolve()
    return model_root / asset.path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download(asset: Asset, destination: Path, verify_existing: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != asset.size:
            raise RuntimeError(f"Existing file has wrong size; refusing to replace: {destination}")
        if verify_existing and digest(destination) != asset.sha256:
            raise RuntimeError(f"Existing file has wrong SHA-256; refusing to replace: {destination}")
        print(f"EXISTS {destination}" + (" (SHA-256 verified)" if verify_existing else " (size checked)"), flush=True)
        return

    partial = destination.with_name(destination.name + ".partial")
    url = asset.url or f"https://huggingface.co/{asset.repo}/resolve/{asset.revision}/{quote(asset.source, safe='/')}"
    headers = {}
    if not asset.url and os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    print(f"DOWNLOAD {destination} ({asset.size / 1e9:.2f} GB)", flush=True)
    for attempt in range(10):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > asset.size:
            raise RuntimeError(f"Partial file larger than expected: {partial}")
        if offset == asset.size:
            break
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=request_headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError(f"Server ignored Range for {partial}; partial retained")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError(f"Invalid Content-Range for {partial}; partial retained")
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
        except requests.RequestException as error:
            if attempt == 9:
                raise RuntimeError(f"Download failed after retries: {destination}") from error
            delay = min(2**attempt, 30)
            print(f"RETRY {destination.name}: {type(error).__name__}; waiting {delay}s", flush=True)
            time.sleep(delay)
    if not partial.exists() or partial.stat().st_size != asset.size or digest(partial) != asset.sha256:
        raise RuntimeError(f"Downloaded file failed size/SHA-256 check; partial retained: {partial}")
    partial.replace(destination)
    print(f"VERIFIED {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=GROUPS, help="Repeat for extra model sets; default: fused-core + postprocess")
    parser.add_argument("--model-root", type=Path, default=Path("/storage/h3-research/models"))
    parser.add_argument("--dry-run", action="store_true", help="Show selected files without downloading or hashing")
    parser.add_argument("--verify-existing", action="store_true", help="SHA-256 check already complete files too")
    parser.add_argument("--prepare-x2-int8", action="store_true", help="After download, convert X2 with the installed H3-X2-Stream node and ComfyUI Python (CUDA required for first conversion)")
    parser.add_argument("--comfyui-dir", type=Path, default=Path("/storage/ComfyUI"))
    parser.add_argument("--comfyui-python", type=Path, help="Default: COMFYUI_DIR/.venv/bin/python")
    args = parser.parse_args()
    args.model_root = args.model_root.resolve()
    args.comfyui_dir = args.comfyui_dir.resolve()
    if args.comfyui_python:
        args.comfyui_python = args.comfyui_python.resolve()
    groups = args.group or ["fused-core", "postprocess"]
    keys = list(dict.fromkeys(key for group in groups for key in GROUPS[group]))
    if "rife49" in keys and not args.dry_run:
        configs = [args.comfyui_dir / "custom_nodes" / name / "config.yaml" for name in
                   ("comfyui-frame-interpolation", "ComfyUI-Frame-Interpolation")]
        if not any(path.is_file() for path in configs):
            parser.error("Install ComfyUI-Frame-Interpolation with config.yaml first; no models were downloaded")
    conversion = None
    if args.prepare_x2_int8:
        if "x2-vae" not in keys:
            parser.error("--prepare-x2-int8 requires --group x2 or --group fused-x2")
        converter = args.comfyui_dir / "custom_nodes/ComfyUI-H3-X2-Stream/quantization.py"
        python = args.comfyui_python or args.comfyui_dir / ".venv/bin/python"
        output = args.model_root / "vae/h3_x2_stream/MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors"
        conversion = [str(python), str(converter), str(args.model_root / ASSETS["x2-vae"].path), str(output)]
        if not args.dry_run and (not python.is_file() or not converter.is_file()):
            parser.error("Install ComfyUI-H3-X2-Stream in the selected ComfyUI first; no models were downloaded")
    for key in keys:
        asset = ASSETS[key]
        destination = destination_for(asset, args.model_root, args.comfyui_dir)
        if args.dry_run:
            state = "present" if destination.is_file() and destination.stat().st_size == asset.size else "needed"
            print(f"{key:18} {state:7} {asset.size / 1e9:5.2f} GB  {destination}")
        else:
            download(asset, destination, args.verify_existing)
    if conversion:
        if args.dry_run:
            print("PREPARE X2 INT8 (additional ~2.83 GB):", " ".join(conversion))
        else:
            subprocess.run(conversion, check=True, cwd=args.comfyui_dir)
    print(f"TOTAL selected: {sum(ASSETS[key].size for key in keys) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
