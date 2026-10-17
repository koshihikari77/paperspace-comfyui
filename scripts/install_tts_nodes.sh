#!/usr/bin/env bash
# Japanese TTS in ComfyUI: drama3-small (pure-PyTorch node, no audio.cpp) and Irodori-TTS v4.1 Anime.
# Nodes and their Python packages persist in /storage/ComfyUI; models are per-session in /app/models
# (re-run this script after a restart; installed parts are skipped). See
# docs/tts-drama3-small-vs-irodori-2026-10-16.md for why these two models.
set -euo pipefail

COMFYUI_DIR=${COMFYUI_DIR:-/storage/ComfyUI}
MODEL_ROOT=${MODEL_ROOT:-/app/models}
PY=${COMFYUI_PYTHON:-$COMFYUI_DIR/.venv/bin/python}
NODES=$COMFYUI_DIR/custom_nodes

DRAMA3_NODE_COMMIT=c7137d2c26333ba6aa12aeabb4fddfbf8e6a6e73   # pentaCoxian/ComfyUI-drama3-small
IRODORI_NODE_COMMIT=60b57245c2a6a9544c62c91166c8b80cca141ef7  # kjranyone/comfy-IrodoriTTS
DRAMA3_REV=4a3078e885344bb9b24d511cec389d2726854d15
ANIME_REV=6b259f5baa5e236b3d14cbd1f8555ca87d92b530
CODEC_REV=47376ee24834d7a05a48ebabfe3cde29b3c5e214

clone() {  # repo dir commit
  [ -d "$NODES/$2" ] || git clone -q "https://github.com/$1" "$NODES/$2"
  git -C "$NODES/$2" fetch -q origin "$3" 2>/dev/null || true
  git -C "$NODES/$2" checkout -q "$3"
}
clone pentaCoxian/ComfyUI-drama3-small ComfyUI-drama3-small "$DRAMA3_NODE_COMMIT"
clone kjranyone/comfy-IrodoriTTS comfy-IrodoriTTS "$IRODORI_NODE_COMMIT"

# drama3-small needs nothing beyond ComfyUI's own packages. Irodori's node needs dacvae, whose
# declared deps would pull a CUDA 13 torchaudio and protobuf<3.20; install only what is imported,
# with --no-deps, and torchaudio/torchcodec built for cu126 (they use torch's stable ABI, so it loads on 2.14).
if ! "$PY" -c "import dacvae, peft, torchaudio, torchcodec" 2>/dev/null; then
  "$PY" -m pip install -q --no-deps --index-url https://download.pytorch.org/whl/cu126 \
    torchaudio==2.11.0+cu126 torchcodec==0.17.0+cu126
  "$PY" -m pip install -q --no-deps \
    "dacvae @ git+https://github.com/facebookresearch/dacvae@414c20785fc3a28373073ea8ef7a1316eeeaca6e" \
    descript-audiotools==0.7.2 argbind==0.3.9 flatten-dict==0.5.0 julius==0.2.8 ffmpy==1.0.0 \
    importlib-resources==7.1.0 randomname==0.2.1 pyloudnorm==0.2.0 tensorboard==2.20.0 absl-py==2.5.1 soundfile==0.14.0 \
    peft==0.21.2 accelerate
fi
"$PY" -c "import torch, torchaudio, dacvae, peft; print('tts deps ok: torch', torch.__version__, 'torchaudio', torchaudio.__version__)"

# The Irodori node caches the codec under its own data/ dir; point that at MODEL_ROOT.
for sub in codecs tokenizers; do
  mkdir -p "$MODEL_ROOT/irodori/$sub"
  link=$NODES/comfy-IrodoriTTS/data/$sub
  [ -L "$link" ] || { rm -rf "$link"; ln -s "$MODEL_ROOT/irodori/$sub" "$link"; }
done

# Models (pinned). drama3 is found through extra_model_paths.yaml (write_extra_model_paths.py).
hf download pentacoxian-dev/drama3-small-GGUF drama3-small-IQ3_XXS.gguf --revision "$DRAMA3_REV" \
  --local-dir "$MODEL_ROOT/drama3"
hf download phasefield-audio/Irodori-TTS-v4.1-Anime model.safetensors tokenizer/tokenizer.json \
  tokenizer/tokenizer_config.json README.md --revision "$ANIME_REV" \
  --local-dir "$MODEL_ROOT/checkpoints/Irodori-TTS-v4.1-Anime"
hf download Aratako/Semantic-DACVAE-Japanese-32dim weights.pth --revision "$CODEC_REV" \
  --local-dir "$MODEL_ROOT/irodori/codecs/Aratako_Semantic-DACVAE-Japanese-32dim"
echo "tts nodes ready"
