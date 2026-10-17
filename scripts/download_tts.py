#!/usr/bin/env python3
"""Download pinned Japanese TTS weights (drama3-small, Irodori-TTS v4.1 Anime) for the unified ComfyUI.

The nodes and their Python packages are installed once into /storage/ComfyUI by
scripts/install_tts_nodes.sh; this script only fetches the per-session models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from download_minimax_h3 import Asset, download

DRAMA3_REV = "4a3078e885344bb9b24d511cec389d2726854d15"
ANIME_REV = "6b259f5baa5e236b3d14cbd1f8555ca87d92b530"
CODEC_REV = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
ANIME_DIR = "checkpoints/Irodori-TTS-v4.1-Anime"
# comfy-IrodoriTTS looks for the codec in its data/codecs/<repo>/, which links here.
CODEC_DIR = "irodori/codecs/Aratako_Semantic-DACVAE-Japanese-32dim"

ASSETS = {
    "drama3-small": Asset(
        "drama3/drama3-small-IQ3_XXS.gguf",
        "pentacoxian-dev/drama3-small-GGUF", DRAMA3_REV, "drama3-small-IQ3_XXS.gguf",
        10460748576, "97d773973d756141a65432b08de76fd4bedbb44e8b98805755130e02d7fe2561",
    ),
    "irodori-anime": Asset(
        f"{ANIME_DIR}/model.safetensors",
        "phasefield-audio/Irodori-TTS-v4.1-Anime", ANIME_REV, "model.safetensors",
        3064295596, "5630aa0a661930ff678a1d1f1893876e24520f4776d4fcc25ca54e6ab9f41f37",
    ),
    "irodori-anime-tokenizer": Asset(
        f"{ANIME_DIR}/tokenizer/tokenizer.json",
        "phasefield-audio/Irodori-TTS-v4.1-Anime", ANIME_REV, "tokenizer/tokenizer.json",
        6718495, "6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d",
    ),
    "irodori-anime-tokenizer-config": Asset(
        f"{ANIME_DIR}/tokenizer/tokenizer_config.json",
        "phasefield-audio/Irodori-TTS-v4.1-Anime", ANIME_REV, "tokenizer/tokenizer_config.json",
        668, "d229a271c64de1a7939d20d3665498e873fa91d5ee2edf135d73ec752cb9c9d3",
    ),
    "irodori-codec": Asset(
        f"{CODEC_DIR}/weights.pth",
        "Aratako/Semantic-DACVAE-Japanese-32dim", CODEC_REV, "weights.pth",
        429620065, "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5",
    ),
}

GROUPS = {
    "drama3-small": ("drama3-small",),
    "irodori-anime": ("irodori-anime", "irodori-anime-tokenizer", "irodori-anime-tokenizer-config", "irodori-codec"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=GROUPS, help="Repeat for several; default: all groups")
    parser.add_argument("--model-root", type=Path, default=Path("/app/models"))
    parser.add_argument("--dry-run", action="store_true", help="Show selected files without downloading or hashing")
    parser.add_argument("--verify-existing", action="store_true", help="SHA-256 check already complete files too")
    args = parser.parse_args()
    model_root = args.model_root.resolve()
    keys = list(dict.fromkeys(key for group in args.group or GROUPS for key in GROUPS[group]))
    total = sum(ASSETS[key].size for key in keys)
    for key in keys:
        destination = model_root / ASSETS[key].path
        if args.dry_run:
            print(f"PLAN {destination} ({ASSETS[key].size / 1e9:.2f} GB)", flush=True)
        else:
            download(ASSETS[key], destination, args.verify_existing)
    print(f"TOTAL selected: {total / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
