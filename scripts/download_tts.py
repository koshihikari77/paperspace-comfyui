#!/usr/bin/env python3
"""Download pinned Japanese TTS / voice-conversion weights (drama3-small, Irodori-TTS v4.1 Anime, Seed-VC,
and the RVC pretrains used by Applio).

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
# ComfyUI_Seed-VC reads ComfyUI/models/TTS/, which links to MODEL_ROOT/TTS (install_tts_nodes.sh).
SEEDVC_REV = "257283f9f41585055e8f858fba4fd044e5caed6e"
CAMPPLUS_REV = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"
RVC_REV = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
BIGVGAN22_REV = "633ff708ed5b74903e86ff1298cf4a98e921c513"
BIGVGAN44_REV = "95a9d1dcb12906c03edd938d77b9333d6ded7dfb"
WHISPER_SMALL_REV = "973afd24965f72e36ca33b3055d56a652f456b4d"
# Applio (/storage/tts-runtime/applio, its own venv) links rvc/models/{predictors,embedders/contentvec,
# pretraineds/hifi-gan} to MODEL_ROOT/rvc/ (install_rvc.sh).
APPLIO_REV = "e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14"

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
    "seedvc-f0-44k": Asset(
        "TTS/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
        "Plachta/Seed-VC", SEEDVC_REV, "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
        820864573, "883d19be9fe9c4bb2dce30d545cd77e5ed2b6326c7ed1c4d9a7ce8e555b163f9",
    ),
    "seedvc-speech-22k": Asset(
        "TTS/Seed-VC/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "Plachta/Seed-VC", SEEDVC_REV, "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        440312082, "8ec8841b20bb46df9f7e8e570a6946a4b87b940133c7f0e778487ff33841f720",
    ),
    "campplus": Asset(
        "TTS/Seed-VC/campplus_cn_common.bin", "funasr/campplus", CAMPPLUS_REV, "campplus_cn_common.bin",
        28036335, "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8",
    ),
    "rmvpe": Asset(
        "TTS/Seed-VC/rmvpe.pt", "lj1995/VoiceConversionWebUI", RVC_REV, "rmvpe.pt",
        181184272, "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
    ),
    "bigvgan-22k": Asset(
        "TTS/bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt",
        "nvidia/bigvgan_v2_22khz_80band_256x", BIGVGAN22_REV, "bigvgan_generator.pt",
        449228171, "e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced",
    ),
    "bigvgan-22k-config": Asset(
        "TTS/bigvgan_v2_22khz_80band_256x/config.json",
        "nvidia/bigvgan_v2_22khz_80band_256x", BIGVGAN22_REV, "config.json",
        1405, "88a1f47acf747db0b21e97a389d838566147f7a5464583ff5c8d819d870f03ee",
    ),
    "bigvgan-44k": Asset(
        "TTS/bigvgan_v2_44khz_128band_512x/bigvgan_generator.pt",
        "nvidia/bigvgan_v2_44khz_128band_512x", BIGVGAN44_REV, "bigvgan_generator.pt",
        489041291, "d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357",
    ),
    "bigvgan-44k-config": Asset(
        "TTS/bigvgan_v2_44khz_128band_512x/config.json",
        "nvidia/bigvgan_v2_44khz_128band_512x", BIGVGAN44_REV, "config.json",
        1403, "65b7f487bfaf15256056a75f6667ef5f640214c981922a111927873705ea4ddb",
    ),
}
for key, path, source, size, sha256 in [
    ("rvc-rmvpe", "predictors/rmvpe.pt", "predictors/rmvpe.pt", 181184272,
     "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"),
    ("rvc-fcpe", "predictors/fcpe.pt", "predictors/fcpe.pt", 43362881,
     "8544427eebbf2baef6213cc9a05057e46961617a8e5bd96975a0d42da6a09059"),
    ("rvc-contentvec", "embedders/contentvec/pytorch_model.bin", "embedders/contentvec/pytorch_model.bin", 378342945,
     "d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e"),
    ("rvc-contentvec-config", "embedders/contentvec/config.json", "embedders/contentvec/config.json", 1388,
     "2ddde063b795d38d9051a7215a092fecf4cfe148b54251e38de51d88d356898b"),
    ("rvc-g48k", "pretraineds/hifi-gan/f0G48k.pth", "pretrained_v2/f0G48k.pth", 75465569,
     "b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4"),
    ("rvc-d48k", "pretraineds/hifi-gan/f0D48k.pth", "pretrained_v2/f0D48k.pth", 142875703,
     "2269b73c7a4cf34da09aea99274dabf99b2ddb8a42cbfb065fb3c0aa9a2fc748"),
]:
    ASSETS[key] = Asset(f"rvc/{path}", "IAHispano/Applio", APPLIO_REV, f"Resources/{source}", size, sha256)
# Seed-VC's content encoder (WhisperModel.from_pretrained on a local folder).
for name, size, sha256 in [
    ("model.safetensors", 966995080, "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"),
    ("added_tokens.json", 34604, "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1"),
    ("config.json", 1967, "e6a2b489da1b5aed65a8eb8d1e7466fa867ad5643a8bc138ba708bd56b2875c4"),
    ("generation_config.json", 3868, "71565b8ef50d0bf7a1193ed4bbed195b94e70c18894d81bba2f1233dcec3ab53"),
    ("merges.txt", 493869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6"),
    ("normalizer.json", 52666, "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"),
    ("preprocessor_config.json", 184990, "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666"),
    ("special_tokens_map.json", 2194, "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691"),
    ("tokenizer.json", 2480466, "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566"),
    ("tokenizer_config.json", 282683, "2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce"),
    ("vocab.json", 835550, "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9"),
]:
    ASSETS[f"whisper-small/{name}"] = Asset(f"TTS/whisper-small/{name}", "openai/whisper-small",
                                            WHISPER_SMALL_REV, name, size, sha256)

GROUPS = {
    "drama3-small": ("drama3-small",),
    "irodori-anime": ("irodori-anime", "irodori-anime-tokenizer", "irodori-anime-tokenizer-config", "irodori-codec"),
    # The node loads both DiT models, both vocoders and whisper-small at once, so this group is all-or-nothing.
    "seed-vc": ("seedvc-f0-44k", "seedvc-speech-22k", "campplus", "rmvpe", "bigvgan-22k", "bigvgan-22k-config",
                "bigvgan-44k", "bigvgan-44k-config", *(k for k in ASSETS if k.startswith("whisper-small/"))),
    # RVC v2 48 kHz with f0 (RMVPE / FCPE), ContentVec embedder.
    "rvc": tuple(k for k in ASSETS if k.startswith("rvc-")),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=GROUPS, help="Repeat for several; default: drama3-small + irodori-anime")
    parser.add_argument("--model-root", type=Path, default=Path("/app/models"))
    parser.add_argument("--dry-run", action="store_true", help="Show selected files without downloading or hashing")
    parser.add_argument("--verify-existing", action="store_true", help="SHA-256 check already complete files too")
    args = parser.parse_args()
    model_root = args.model_root.resolve()
    keys = list(dict.fromkeys(key for group in args.group or ["drama3-small", "irodori-anime"] for key in GROUPS[group]))
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
