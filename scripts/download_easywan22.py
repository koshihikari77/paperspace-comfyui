#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class Asset:
    name: str
    relative_path: str
    source: str
    description: str
    target_root: str = "model"
    repo_id: str | None = None
    repo_path: str | None = None
    url: str | None = None
    strip_prefix: str | None = None
    model_version_id: int | None = None
    archive_member: str | None = None


GROUP_DESCRIPTIONS = {
    "floyo-wan22-stable": "Minimal model set for the stable Floyo WanVideoWrapper I2V workflow.",
    "easywan22-default": "Full EasyWan22 Default.bat asset set, including preset LoRAs and detectors.",
    "easywan22-default-no-gguf": "EasyWan22 default asset set without GGUF video models; pair with fp8_scaled or SmoothMIX downloads.",
    "eye-loras": "JujoHotaru eyecollexl eye LoRAs used by the notebook image mode.",
    "workflow-core": "Assets required by EasyWan22/Workflow/00-I2v_ImageToVideo.json core generation path.",
    "workflow-gguf-models": "GGUF video models used by EasyWan22 core workflow.",
    "workflow-detailer": "Minimal detector asset used by the Detailer block in the workflow.",
    "workflow-automosaic": "Detector asset used by the workflow's AutoMosaic branch.",
    "ultralytics-hf": "Additional Hugging Face-hosted Ultralytics segmentation models used by EasyWan22.",
    "default-hf": "Hugging Face-hosted subset of EasyWan22/Download/Default.bat.",
    "fast-loras": "Fast / Lightning LoRAs referenced by EasyWan22's default download set.",
    "wan21fast": "Wan2.1 / Lightx2v LoRAs related to fast generation.",
    "nashikone-i2v": "Nashikone Wan2.2 I2V preset LoRA bundle from Hugging Face.",
    "nashikone-i2vwan21": "Nashikone Wan2.1 I2V preset LoRA bundle from Hugging Face.",
}

DOWNLOAD_ROOT = Path("/notebooks/EasyWan22/Download")
DEFAULT_BAT = DOWNLOAD_ROOT / "Default.bat"
PRIVATE_MIRROR_REPO = os.environ.get("EASYWAN22_PRIVATE_REPO", "korokoro77/paperspace_models_mirror")
EYE_LORA_REPO = "JujoHotaru/lora"
EYE_LORA_SUBDIR = "sdxl/eyecollexl"

CALL_RELATIVE_BAT_RE = re.compile(r"^call\s+%~dp0(?P<path>.+?\.bat)\s*$", re.IGNORECASE)
HF_CALL_RE = re.compile(
    r"^call\s+%HUGGING_FACE%\s+(?P<subdir>\S+)\s+(?P<filename>\S+)\s+"
    r"(?P<repo>\S+)(?:\s+(?P<repo_path>\S+))?\s*$",
    re.IGNORECASE,
)
HF_HUB_RE = re.compile(
    r"^call\s+%HUGGING_FACE_HUB%\s+(?P<tmpdir>\S+)\s+(?P<repo>\S+)\s+model\s+"
    r"(?P<pattern>\S+)\s*$",
    re.IGNORECASE,
)
CIVITAI_CALL_RE = re.compile(
    r"^call\s+%CIVITAI_MODEL_DOWNLOAD%\s+(?P<subdir>\S+)\s+(?P<filename>\S+)\s+"
    r"(?P<model_id>\d+)\s+(?P<version_id>\d+)\s*$",
    re.IGNORECASE,
)
CIVITAI_UNZIP_RE = re.compile(
    r"^call\s+%CIVITAI_MODEL_DOWNLOAD_UNZIP%\s+(?P<subdir>\S+)\s+(?P<filename>\S+)\s+"
    r"(?P<model_id>\d+)\s+(?P<version_id>\d+)\s*$",
    re.IGNORECASE,
)
JUNCTION_RE = re.compile(
    r"^call\s+%JUNCTION%\s+(?P<dest>\S+)\s+(?P<src>\S+)\s*$",
    re.IGNORECASE,
)


GROUP_ASSETS: dict[str, list[Asset]] = {
    "floyo-wan22-stable": [
        Asset(
            name="wan22_i2v_high_fp8_scaled",
            relative_path="diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            source="hf_file",
            repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            repo_path="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            description="High-noise Wan2.2 I2V model used by the stable Floyo workflow.",
        ),
        Asset(
            name="wan22_i2v_low_fp8_scaled",
            relative_path="diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            source="hf_file",
            repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            repo_path="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            description="Low-noise Wan2.2 I2V model used by the stable Floyo workflow.",
        ),
        Asset(
            name="umt5_wrapper_bf16",
            relative_path="text_encoders/umt5-xxl-enc-bf16.safetensors",
            source="hf_file",
            repo_id="Kijai/WanVideo_comfy",
            repo_path="umt5-xxl-enc-bf16.safetensors",
            description="BF16 T5 encoder used by LoadWanVideoT5TextEncoder.",
        ),
        Asset(
            name="wan_vae_wrapper_bf16",
            relative_path="vae/Wan2_1_VAE_bf16.safetensors",
            source="hf_file",
            repo_id="Kijai/WanVideo_comfy",
            repo_path="Wan2_1_VAE_bf16.safetensors",
            description="WanVideoWrapper VAE used by the stable Floyo workflow.",
        ),
        Asset(
            name="lightx2v_i2v_rank64_root",
            relative_path="loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            source="hf_file",
            repo_id="Kijai/WanVideo_comfy",
            repo_path="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            description="Distill LoRA at the root path expected by the stable Floyo workflow.",
        ),
        Asset(
            name="deepthroat_high",
            relative_path="loras/Nsfw/DeepthroatBlowjob_v10-H.safetensors",
            source="hf_file",
            repo_id=PRIVATE_MIRROR_REPO,
            repo_path="loras/Nsfw/DeepthroatBlowjob_v10-H.safetensors",
            description="Default high-noise motion LoRA used by the stable Floyo workflow.",
        ),
        Asset(
            name="deepthroat_low",
            relative_path="loras/Nsfw/DeepthroatBlowjob_v10-L.safetensors",
            source="hf_file",
            repo_id=PRIVATE_MIRROR_REPO,
            repo_path="loras/Nsfw/DeepthroatBlowjob_v10-L.safetensors",
            description="Default low-noise motion LoRA used by the stable Floyo workflow.",
        ),
        Asset(
            name="realesrgan_x2",
            relative_path="upscale_models/RealESRGAN_x2.pth",
            source="hf_file",
            repo_id="ai-forever/Real-ESRGAN",
            repo_path="RealESRGAN_x2.pth",
            description="2x upscaler used by the stable Floyo workflow.",
        ),
    ],
    "workflow-core": [
        Asset(
            name="wan_vae_bf16",
            relative_path="vae/Wan2_1_VAE_bf16.safetensors",
            source="hf_file",
            repo_id="Kijai/WanVideo_comfy",
            repo_path="Wan2_1_VAE_bf16.safetensors",
            description="WanVideoWrapper VAE expected by the EasyWan22 workflow.",
        ),
        Asset(
            name="umt5_wrapper",
            relative_path="text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors",
            source="hf_file",
            repo_id="Kijai/WanVideo_comfy",
            repo_path="umt5-xxl-enc-fp8_e4m3fn.safetensors",
            description="Wrapper-specific T5 text encoder used by LoadWanVideoT5TextEncoder.",
        ),
        Asset(
            name="qwen25_3b_wrapper",
            relative_path="text_encoders/Qwen2.5_3B_instruct_bf16.safetensors",
            source="hf_file",
            repo_id="Kijai/WanVideo_comfy",
            repo_path="Qwen/Qwen2.5_3B_instruct_bf16.safetensors",
            description="Qwen prompt rewrite model referenced by the workflow.",
        ),
        Asset(
            name="lightx2v_i2v_rank64",
            relative_path="loras/Wan21Fast/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            source="hf_file",
            repo_id="Kijai/WanVideo_comfy",
            repo_path="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            description="Wan2.1 fast LoRA referenced directly by the workflow sampler.",
        ),
        Asset(
            name="anime_sharp_upscaler",
            relative_path="upscale_models/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
            source="hf_file",
            repo_id="Kim2091/2x-AnimeSharpV4",
            repo_path="2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
            description="Upscaler model referenced by the workflow.",
        ),
    ],
    "workflow-gguf-models": [
        Asset(
            name="fastmix_high_q4",
            relative_path="diffusion_models/FastMix/Wan22-I2V-FastMix_v10-H-Q4_K_M.gguf",
            source="hf_file",
            repo_id="Zuntan/Wan22-FastMix",
            repo_path="Wan22-I2V-FastMix_v10-H-Q4_K_M.gguf",
            description="FastMix high-noise GGUF model.",
        ),
        Asset(
            name="fastmix_low_q4",
            relative_path="diffusion_models/FastMix/Wan22-I2V-FastMix_v10-L-Q4_K_M.gguf",
            source="hf_file",
            repo_id="Zuntan/Wan22-FastMix",
            repo_path="Wan22-I2V-FastMix_v10-L-Q4_K_M.gguf",
            description="FastMix low-noise GGUF model.",
        ),
        Asset(
            name="base_high_q4",
            relative_path="diffusion_models/Base/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
            source="hf_file",
            repo_id="QuantStack/Wan2.2-I2V-A14B-GGUF",
            repo_path="HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
            description="Base high-noise GGUF model.",
        ),
        Asset(
            name="base_low_q4",
            relative_path="diffusion_models/Base/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
            source="hf_file",
            repo_id="QuantStack/Wan2.2-I2V-A14B-GGUF",
            repo_path="LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
            description="Base low-noise GGUF model.",
        ),
    ],
    "workflow-detailer": [
        Asset(
            name="anzhc_face_seg",
            relative_path="ultralytics/segm/AnzhcFace-v3-640-seg.pt",
            source="hf_file",
            repo_id="Anzhc/Anzhcs_YOLOs",
            repo_path="Anzhc Face seg 640 v3 y11n.pt",
            description="Face segmentation model used by the workflow detailer.",
        ),
    ],
    "workflow-automosaic": [
        Asset(
            name="anime_nsfw_detection_all_v4",
            relative_path="ultralytics/segm/ntd11_anime_nsfw_segm_v4_all.pt",
            source="civitai_zip_member",
            model_version_id=1863248,
            archive_member="ntd11_anime_nsfw_segm_v4_all.pt",
            description="AutoMosaic detector used by the workflow's NSFW segmentation branch.",
        ),
    ],
    "ultralytics-hf": [
        Asset(
            name="anzhc_breasts_seg",
            relative_path="ultralytics/segm/AnzhcBreasts-v1-1024-seg.pt",
            source="hf_file",
            repo_id="Anzhc/Anzhcs_YOLOs",
            repo_path="Anzhc Breasts Seg v1 1024m.pt",
            description="Breasts segmentation model from EasyWan22 default downloads.",
        ),
        Asset(
            name="anzhc_eyes_seg",
            relative_path="ultralytics/segm/AnzhcEyes-seg.pt",
            source="hf_file",
            repo_id="Anzhc/Anzhcs_YOLOs",
            repo_path="Anzhc Eyes -seg-hd.pt",
            description="Eyes segmentation model from EasyWan22 default downloads.",
        ),
        Asset(
            name="anzhc_headhair_seg",
            relative_path="ultralytics/segm/AnzhcHeadHair-seg.pt",
            source="hf_file",
            repo_id="Anzhc/Anzhcs_YOLOs",
            repo_path="Anzhc HeadHair seg y8m.pt",
            description="Head/hair segmentation model from EasyWan22 default downloads.",
        ),
    ],
    "fast-loras": [
        Asset(
            name="fast_lora_high",
            relative_path="loras/Fast/Wan22-I2V-A14B-4steps-lora-rank64-Seko-V1-H.safetensors",
            source="hf_file",
            repo_id="lightx2v/Wan2.2-Lightning",
            repo_path="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
            description="Fast 4-step high-noise LoRA from EasyWan22 default downloads.",
        ),
        Asset(
            name="fast_lora_low",
            relative_path="loras/Fast/Wan22-I2V-A14B-4steps-lora-rank64-Seko-V1-L.safetensors",
            source="hf_file",
            repo_id="lightx2v/Wan2.2-Lightning",
            repo_path="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
            description="Fast 4-step low-noise LoRA from EasyWan22 default downloads.",
        ),
    ],
    "wan21fast": [
        Asset(
            name="taew2_1",
            relative_path="vae_approx/taew2_1.safetensors",
            source="hf_file",
            repo_id="Kijai/WanVideo_comfy",
            repo_path="taew2_1.safetensors",
            description="Approximate VAE used by EasyWan22 default downloads.",
        ),
        Asset(
            name="umt5_scaled_native",
            relative_path="text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            source="hf_file",
            repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            repo_path="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            description="Scaled native UMT5 model from EasyWan22 default downloads.",
        ),
        Asset(
            name="clip_vision_h",
            relative_path="clip_vision/clip_vision_h.safetensors",
            source="hf_file",
            repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            repo_path="split_files/clip_vision/clip_vision_h.safetensors",
            description="Clip vision model downloaded by EasyWan22 helper scripts.",
        ),
    ],
    "nashikone-i2v": [
        Asset(
            name="nashikone_i2v_bundle",
            relative_path="loras/Nashikone-I2v",
            source="hf_snapshot",
            repo_id="nashikone/iroiroLoRA",
            repo_path="Wan2.2_i2v_A14B/**",
            strip_prefix="Wan2.2_i2v_A14B",
            description="Wan2.2 I2V Nashikone preset bundle.",
        ),
    ],
    "nashikone-i2vwan21": [
        Asset(
            name="nashikone_i2vwan21_bundle",
            relative_path="loras/Nashikone-I2vWan21",
            source="hf_snapshot",
            repo_id="nashikone/iroiroLoRA",
            repo_path="Wan2.1_i2v_720p_14B_fp16/**",
            strip_prefix="Wan2.1_i2v_720p_14B_fp16",
            description="Wan2.1 I2V Nashikone preset bundle.",
        ),
    ],
}

GROUP_ASSETS["default-hf"] = (
    GROUP_ASSETS["workflow-core"]
    + GROUP_ASSETS["workflow-detailer"]
    + GROUP_ASSETS["ultralytics-hf"]
    + GROUP_ASSETS["fast-loras"]
    + GROUP_ASSETS["wan21fast"]
)


def normalize_windows_relpath(value: str) -> str:
    value = value.replace("\\", "/").strip()
    if value in {".", "./"}:
        return ""
    if value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def normalize_hf_repo_path(value: str) -> str:
    normalized = value.rstrip("?").replace("\\", "/")
    normalized = re.sub(r"%{2,}20", " ", normalized)
    return normalized


def relative_model_root_from_windows_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    for marker in ("ComfyUI/models/", "Model/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1].strip("/").rstrip("/")
    raise SystemExit(f"Unsupported model path in EasyWan22 batch file: {value}")


def resolve_default_bat_calls(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    resolved: list[Path] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("@rem"):
            continue
        match = CALL_RELATIVE_BAT_RE.match(stripped)
        if not match:
            continue
        target = (path.parent / match.group("path").replace("\\", "/")).resolve()
        if not target.exists():
            raise SystemExit(f"Referenced batch file not found: {target}")
        resolved.append(target)
    return resolved


def parse_easywan22_asset_batch(path: Path) -> list[Asset]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    model_base: str | None = None
    hub_repo: str | None = None
    hub_pattern: str | None = None
    assets: list[Asset] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.lower().startswith("@rem"):
            continue

        if line.lower().startswith("pushd "):
            if "ComfyUI\\models\\" in line or "Model\\" in line or "ComfyUI/models/" in line:
                model_base = relative_model_root_from_windows_path(line.split(None, 1)[1])
            continue

        match = HF_CALL_RE.match(line)
        if match:
            if model_base is None:
                raise SystemExit(f"Missing pushd model root in {path}")
            subdir = normalize_windows_relpath(match.group("subdir"))
            filename = match.group("filename").rstrip("?")
            relative_path = "/".join(
                part for part in (model_base, subdir, filename) if part
            )
            repo_path_arg = match.group("repo_path")
            if repo_path_arg:
                repo_path_arg = normalize_hf_repo_path(repo_path_arg)
                repo_path = (
                    f"{repo_path_arg.rstrip('/')}/{filename}"
                    if repo_path_arg.endswith(("/", "\\"))
                    else repo_path_arg
                )
            else:
                repo_path = filename
            assets.append(
                Asset(
                    name=path.stem,
                    relative_path=relative_path,
                    source="hf_file",
                    repo_id=match.group("repo"),
                    repo_path=repo_path,
                    description=f"Parsed from {path.relative_to(DOWNLOAD_ROOT)}",
                )
            )
            continue

        match = CIVITAI_CALL_RE.match(line)
        if match:
            if model_base is None:
                raise SystemExit(f"Missing pushd model root in {path}")
            subdir = normalize_windows_relpath(match.group("subdir"))
            filename = match.group("filename")
            relative_path = "/".join(
                part for part in (model_base, subdir, filename) if part
            )
            assets.append(
                Asset(
                    name=path.stem,
                    relative_path=relative_path,
                    source="civitai_file",
                    model_version_id=int(match.group("version_id")),
                    description=f"Parsed from {path.relative_to(DOWNLOAD_ROOT)}",
                )
            )
            continue

        match = CIVITAI_UNZIP_RE.match(line)
        if match:
            if model_base is None:
                raise SystemExit(f"Missing pushd model root in {path}")
            subdir = normalize_windows_relpath(match.group("subdir"))
            filename = match.group("filename")
            relative_path = "/".join(
                part for part in (model_base, subdir, filename) if part
            )
            assets.append(
                Asset(
                    name=path.stem,
                    relative_path=relative_path,
                    source="civitai_zip_member",
                    model_version_id=int(match.group("version_id")),
                    archive_member=filename,
                    description=f"Parsed from {path.relative_to(DOWNLOAD_ROOT)}",
                )
            )
            continue

        match = HF_HUB_RE.match(line)
        if match:
            hub_repo = match.group("repo")
            hub_pattern = match.group("pattern")
            continue

        match = JUNCTION_RE.match(line)
        if match and hub_repo and hub_pattern:
            dest = relative_model_root_from_windows_path(match.group("dest"))
            strip_prefix = match.group("src").replace("\\", "/").split("%~n0/", 1)[-1]
            assets.append(
                Asset(
                    name=path.stem,
                    relative_path=dest,
                    source="hf_snapshot",
                    repo_id=hub_repo,
                    repo_path=hub_pattern,
                    strip_prefix=strip_prefix.rstrip("/"),
                    description=f"Parsed from {path.relative_to(DOWNLOAD_ROOT)}",
                )
            )
            continue

    if not assets:
        raise SystemExit(f"Could not parse asset batch file: {path}")
    return assets


def build_easywan22_default_assets() -> list[Asset]:
    assets: list[Asset] = []
    for bat_path in resolve_default_bat_calls(DEFAULT_BAT):
        assets.extend(parse_easywan22_asset_batch(bat_path))
    return assets


GROUP_ASSETS["easywan22-default"] = (
    build_easywan22_default_assets() if DEFAULT_BAT.exists() else []
)
GROUP_ASSETS["easywan22-default-no-gguf"] = [
    asset for asset in GROUP_ASSETS["easywan22-default"]
    if not asset.relative_path.endswith(".gguf")
]


def build_eye_lora_assets() -> list[Asset]:
    return [
        Asset(
            name="eye_lora_bundle",
            relative_path="loras/eyes",
            source="hf_snapshot",
            repo_id=EYE_LORA_REPO,
            repo_path=f"{EYE_LORA_SUBDIR}/**",
            strip_prefix=EYE_LORA_SUBDIR,
            description="JujoHotaru eyecollexl eye LoRA bundle.",
        )
    ]


def remap_civitai_assets_to_private_mirror(repo_id: str) -> None:
    for group_name, assets in list(GROUP_ASSETS.items()):
        remapped: list[Asset] = []
        for asset in assets:
            if asset.source not in {"civitai_file", "civitai_zip_member"}:
                remapped.append(asset)
                continue
            remapped.append(
                Asset(
                    name=asset.name,
                    relative_path=asset.relative_path,
                    source="hf_mirror_file",
                    target_root="civitai",
                    repo_id=repo_id,
                    repo_path=asset.relative_path,
                    description=f"{asset.description} Mirrored from Civitai into the private HF repo.",
                )
            )
        GROUP_ASSETS[group_name] = remapped


GROUP_ASSETS["eye-loras"] = build_eye_lora_assets()
remap_civitai_assets_to_private_mirror(PRIVATE_MIRROR_REPO)


def normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = value.strip()
    return token or None


def available_groups() -> list[str]:
    return sorted(GROUP_ASSETS)


def resolve_groups(requested_groups: list[str]) -> list[str]:
    resolved: list[str] = []
    for group in requested_groups:
        if group == "all":
            for name in available_groups():
                if name not in resolved:
                    resolved.append(name)
            continue
        if group not in GROUP_ASSETS:
            choices = ", ".join(["all", *available_groups()])
            raise SystemExit(f"Unknown group: {group}\nChoices: {choices}")
        if group not in resolved:
            resolved.append(group)
    return resolved


def iter_assets(groups: list[str]) -> list[tuple[str, Asset]]:
    seen: set[str] = set()
    selected: list[tuple[str, Asset]] = []
    for group in groups:
        for asset in GROUP_ASSETS[group]:
            key = asset.relative_path
            if key in seen:
                continue
            seen.add(key)
            selected.append((group, asset))
    return selected


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dest: Path, force: bool) -> str:
    ensure_parent(dest)
    if dest.exists() and not force:
        return "skipped"
    shutil.copy2(src, dest)
    return "downloaded"


def asset_model_root(asset: Asset, model_root: Path, civitai_model_root: Path) -> Path:
    if asset.target_root == "civitai":
        return civitai_model_root
    return model_root


def stream_download(url: str, dest: Path, force: bool) -> str:
    ensure_parent(dest)
    if dest.exists() and not force:
        return "skipped"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp_dest = dest.with_name(f".{dest.name}.part")
    try:
        with urllib.request.urlopen(request) as response, tmp_dest.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        tmp_dest.replace(dest)
    finally:
        if tmp_dest.exists():
            tmp_dest.unlink()
    return "downloaded"


def download_civitai_zip_member(
    asset: Asset,
    destination: Path,
    civitai_token: str | None,
    force: bool,
) -> str:
    if asset.model_version_id is None or asset.archive_member is None:
        raise SystemExit(f"Invalid civitai asset definition: {asset.name}")
    if destination.exists() and not force:
        return "skipped"
    if civitai_token is None:
        raise SystemExit(
            "Civitai asset selected but CIVITAI_API_KEY/CIVITAI_TOKEN is not set: "
            f"{asset.relative_path}"
        )

    query = urlencode({"token": civitai_token})
    url = f"https://civitai.com/api/download/models/{asset.model_version_id}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with tempfile.TemporaryDirectory(prefix="easywan22-civitai-") as temp_dir:
        archive_path = Path(temp_dir) / "download.bin"
        with urllib.request.urlopen(request) as response, archive_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)

        if not zipfile.is_zipfile(archive_path):
            raise SystemExit(
                f"Civitai download for {asset.name} was not a zip archive. "
                "Update the downloader for the current file format."
            )

        with zipfile.ZipFile(archive_path) as zf:
            member_name = None
            for candidate in zf.namelist():
                normalized = candidate.rstrip("/")
                if normalized == asset.archive_member or normalized.endswith(f"/{asset.archive_member}"):
                    member_name = candidate
                    break
            if member_name is None:
                raise SystemExit(
                    f"Archive member not found for {asset.name}: {asset.archive_member}"
                )

            ensure_parent(destination)
            tmp_dest = destination.with_name(f".{destination.name}.part")
            with zf.open(member_name) as src, tmp_dest.open("wb") as dest_fh:
                shutil.copyfileobj(src, dest_fh)
            tmp_dest.replace(destination)
        return "downloaded"


def download_civitai_file(
    asset: Asset,
    destination: Path,
    civitai_token: str | None,
    force: bool,
) -> str:
    if asset.model_version_id is None:
        raise SystemExit(f"Invalid civitai asset definition: {asset.name}")
    if destination.exists() and not force:
        return "skipped"
    if civitai_token is None:
        raise SystemExit(
            "Civitai asset selected but CIVITAI_API_KEY/CIVITAI_TOKEN is not set: "
            f"{asset.relative_path}"
        )

    query = urlencode({"token": civitai_token})
    url = f"https://civitai.com/api/download/models/{asset.model_version_id}?{query}"
    return stream_download(url, destination, force)


def copy_snapshot_subdir(snapshot_root: Path, strip_prefix: str, dest_dir: Path, force: bool) -> str:
    src_dir = snapshot_root / strip_prefix
    if not src_dir.exists():
        raise SystemExit(f"Snapshot path not found: {src_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False
    for src in src_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(src_dir)
        dest = dest_dir / rel
        ensure_parent(dest)
        if dest.exists() and not force:
            continue
        shutil.copy2(src, dest)
        copied_any = True
    return "downloaded" if copied_any else "skipped"


def download_hf_file_batch(
    repo_id: str,
    assets: list[Asset],
    destination_root: Path,
    token: str | None,
    force: bool,
    max_workers: int,
    keep_going: bool,
) -> dict[str, str]:
    results: dict[str, str] = {}
    pending_assets: list[Asset] = []
    for asset in assets:
        destination = destination_root / asset.relative_path
        if destination.exists() and not force:
            results[asset.relative_path] = "skipped"
        else:
            pending_assets.append(asset)

    if not pending_assets:
        return results

    patterns = [asset.repo_path for asset in pending_assets if asset.repo_path is not None]
    if len(patterns) != len(pending_assets):
        raise SystemExit(f"Invalid hf_file asset definition in repo batch: {repo_id}")

    with tempfile.TemporaryDirectory(prefix="easywan22-repo-download-") as temp_dir:
        snapshot_root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=patterns,
                local_dir=temp_dir,
                token=token,
                max_workers=max_workers,
            )
        )

        for asset in pending_assets:
            src = snapshot_root / asset.repo_path
            if not src.exists():
                if keep_going:
                    results[asset.relative_path] = f"missing in repo: {asset.repo_path}"
                    continue
                raise SystemExit(f"Downloaded file not found: {src}")
            results[asset.relative_path] = copy_file(src, destination_root / asset.relative_path, force)
        return results


def download_asset(
    asset: Asset,
    model_root: Path,
    civitai_model_root: Path,
    hf_token: str | None,
    civitai_token: str | None,
    force: bool,
    max_workers: int,
) -> str:
    destination = asset_model_root(asset, model_root, civitai_model_root) / asset.relative_path

    if asset.source == "url":
        if asset.url is None:
            raise SystemExit(f"Invalid url asset definition: {asset.name}")
        return stream_download(asset.url, destination, force)

    if asset.source == "civitai_zip_member":
        return download_civitai_zip_member(asset, destination, civitai_token, force)

    if asset.source == "civitai_file":
        return download_civitai_file(asset, destination, civitai_token, force)

    if asset.source == "hf_snapshot":
        if asset.repo_id is None or asset.repo_path is None or asset.strip_prefix is None:
            raise SystemExit(f"Invalid hf_snapshot asset definition: {asset.name}")
        with tempfile.TemporaryDirectory(prefix="easywan22-download-") as temp_dir:
            snapshot_root = Path(
                snapshot_download(
                    repo_id=asset.repo_id,
                    repo_type="model",
                    allow_patterns=[asset.repo_path],
                    local_dir=temp_dir,
                    token=hf_token,
                    max_workers=max_workers,
                )
            )
            return copy_snapshot_subdir(snapshot_root, asset.strip_prefix, destination, force)

    raise SystemExit(f"Unsupported source type: {asset.source}")


def print_group_list() -> None:
    print("Available groups:")
    print("  all")
    for name in available_groups():
        print(f"  {name}: {GROUP_DESCRIPTIONS.get(name, '')}")


def print_asset_list(groups: list[str]) -> None:
    grouped_assets = defaultdict(list)
    for group, asset in iter_assets(groups):
        grouped_assets[group].append(asset)

    for group in groups:
        print(f"[{group}]")
        for asset in grouped_assets[group]:
            print(f"  {asset.relative_path}")
            print(f"    {asset.description}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and place EasyWan22 assets under /app/models or another model root."
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Asset group to download. Repeatable. Use `all` for every defined group.",
    )
    parser.add_argument(
        "--model-root",
        default="/app/models",
        help="Destination model root. Defaults to /app/models.",
    )
    parser.add_argument(
        "--civitai-model-root",
        default="/storage/ComfyUI/models",
        help="Destination root for Civitai assets. Defaults to /storage/ComfyUI/models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist at the destination.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max workers for same-repo Hugging Face downloads. Defaults to 8.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected assets without downloading them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue downloading other assets if one asset fails.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available groups and exit.",
    )
    parser.add_argument(
        "--list-assets",
        action="store_true",
        help="List assets in the selected groups and exit.",
    )
    args = parser.parse_args()

    if args.list_groups:
        print_group_list()
        return 0

    requested_groups = args.group or ["workflow-core"]
    groups = resolve_groups(requested_groups)

    if args.list_assets or args.dry_run:
        print_asset_list(groups)
        return 0

    hf_token = normalize_token(os.environ.get("HF_TOKEN"))
    civitai_token = normalize_token(
        os.environ.get("CIVITAI_API_KEY") or os.environ.get("CIVITAI_TOKEN")
    )
    model_root = Path(args.model_root)
    civitai_model_root = Path(args.civitai_model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    civitai_model_root.mkdir(parents=True, exist_ok=True)

    print(f"Model root: {model_root}")
    print(f"Civitai model root: {civitai_model_root}")
    print(f"Groups: {', '.join(groups)}")

    selected_assets = iter_assets(groups)

    hf_file_batches: dict[str, list[tuple[str, Asset]]] = {}
    other_assets: list[tuple[str, Asset]] = []
    for group, asset in selected_assets:
        if asset.source in {"hf_file", "hf_mirror_file"}:
            if asset.repo_id is None:
                raise SystemExit(f"Invalid hf_file asset definition: {asset.name}")
            batch_key = (asset.repo_id, asset.target_root)
            hf_file_batches.setdefault(batch_key, []).append((group, asset))
        else:
            other_assets.append((group, asset))

    for (repo_id, target_root), repo_assets in hf_file_batches.items():
        destination_root = model_root if target_root == "model" else civitai_model_root
        print(
            f"[hf:{repo_id}] {len(repo_assets)} assets "
            f"(max_workers={args.max_workers})"
        )
        results = download_hf_file_batch(
            repo_id=repo_id,
            assets=[asset for _, asset in repo_assets],
            destination_root=destination_root,
            token=hf_token,
            force=args.force,
            max_workers=args.max_workers,
            keep_going=args.keep_going,
        )
        for group, asset in repo_assets:
            destination = destination_root / asset.relative_path
            print(f"[{group}] {destination}")
            print(f"  {results[asset.relative_path]}")

    for group, asset in other_assets:
        destination = asset_model_root(asset, model_root, civitai_model_root) / asset.relative_path
        print(f"[{group}] {destination}")
        try:
            status = download_asset(
                asset,
                model_root,
                civitai_model_root,
                hf_token,
                civitai_token,
                args.force,
                args.max_workers,
            )
            print(f"  {status}")
        except Exception as exc:
            print(f"  error: {exc}")
            if not args.keep_going:
                raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
