#!/usr/bin/env bash
# Japanese TTS / voice conversion in ComfyUI: drama3-small (pure-PyTorch node, no audio.cpp),
# Irodori-TTS v4.1 Anime and Seed-VC.
# Installs the nodes and their Python packages once into /storage/ComfyUI (already-installed parts are
# skipped). The per-session models in /app/models come from scripts/download_tts.py, which start.ipynb
# runs when DOWNLOAD_TTS_MODELS=True. See
# docs/tts-drama3-small-vs-irodori-2026-10-16.md for why these two models.
set -euo pipefail

COMFYUI_DIR=${COMFYUI_DIR:-/storage/ComfyUI}
MODEL_ROOT=${MODEL_ROOT:-/app/models}
PY=${COMFYUI_PYTHON:-$COMFYUI_DIR/.venv/bin/python}
NODES=$COMFYUI_DIR/custom_nodes

DRAMA3_NODE_COMMIT=c7137d2c26333ba6aa12aeabb4fddfbf8e6a6e73   # pentaCoxian/ComfyUI-drama3-small
IRODORI_NODE_COMMIT=60b57245c2a6a9544c62c91166c8b80cca141ef7  # kjranyone/comfy-IrodoriTTS
SEEDVC_NODE_COMMIT=02c0cb8b05121dd9e391b4287c8e28e0eb4e79a4   # billwuhao/ComfyUI_Seed-VC (upstream seed-vc is archived)

clone() {  # repo dir commit
  [ -d "$NODES/$2" ] || git clone -q "https://github.com/$1" "$NODES/$2"
  git -C "$NODES/$2" fetch -q origin "$3" 2>/dev/null || true
  git -C "$NODES/$2" checkout -q "$3"
}
clone pentaCoxian/ComfyUI-drama3-small ComfyUI-drama3-small "$DRAMA3_NODE_COMMIT"
clone kjranyone/comfy-IrodoriTTS comfy-IrodoriTTS "$IRODORI_NODE_COMMIT"
clone billwuhao/ComfyUI_Seed-VC ComfyUI_Seed-VC "$SEEDVC_NODE_COMMIT"
# Its bundled BigVGAN expects huggingface_hub<1 (proxies/resume_download kwargs); make them optional.
PATCH=$(dirname "$0")/../patches/comfyui-seedvc-hfhub1.patch
git -C "$NODES/ComfyUI_Seed-VC" apply --check "$PATCH" 2>/dev/null && git -C "$NODES/ComfyUI_Seed-VC" apply "$PATCH"

# drama3-small needs nothing beyond ComfyUI's own packages. Irodori's node needs dacvae, whose
# declared deps would pull a CUDA 13 torchaudio and protobuf<3.20; install only what is imported,
# with --no-deps, and torchaudio/torchcodec built for cu126 (they use torch's stable ABI, so it loads on 2.14).
if ! "$PY" -c "import dacvae, peft, torchaudio, torchcodec" 2>/dev/null; then
  "$PY" -m pip install -q --no-deps --index-url https://download.pytorch.org/whl/cu126 \
    torchaudio==2.11.0+cu126 torchcodec==0.17.0+cu126
  "$PY" -m pip install -q --no-deps \
    "dacvae @ git+https://github.com/facebookresearch/dacvae@414c20785fc3a28373073ea8ef7a1316eeeaca6e" \
    descript-audiotools==0.7.2 argbind==0.3.9 docstring-parser==0.18.0 flatten-dict==0.5.0 julius==0.2.8 ffmpy==1.0.0 \
    importlib-resources==7.1.0 randomname==0.2.1 pyloudnorm==0.2.0 tensorboard==2.20.0 absl-py==2.5.1 soundfile==0.14.0 \
    peft==0.21.2 accelerate
fi
# Seed-VC imports librosa and munch only (its requirements.txt also lists funasr, hydra, sounddevice, ... unused).
"$PY" -c "import librosa, munch, docstring_parser" 2>/dev/null || "$PY" -m pip install -q librosa==1.0.0 munch==4.0.0 docstring-parser==0.18.0 \
  -c <("$PY" -m pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|torchcodec|triton|numpy|protobuf|transformers|scipy)==')
"$PY" -c "import torch, torchaudio, dacvae, peft, librosa; print('tts deps ok: torch', torch.__version__, 'torchaudio', torchaudio.__version__)"

# The Irodori node caches the codec under its own data/ dir; point that at MODEL_ROOT.
for sub in codecs tokenizers; do
  mkdir -p "$MODEL_ROOT/irodori/$sub"
  link=$NODES/comfy-IrodoriTTS/data/$sub
  [ -L "$link" ] || { rm -rf "$link"; ln -s "$MODEL_ROOT/irodori/$sub" "$link"; }
done

# ComfyUI_Seed-VC reads ComfyUI/models/TTS/ directly; keep those weights in MODEL_ROOT too.
mkdir -p "$MODEL_ROOT/TTS"
link=$COMFYUI_DIR/models/TTS
[ -L "$link" ] || { [ ! -e "$link" ] || rmdir "$link"; ln -s "$MODEL_ROOT/TTS" "$link"; }

# Seed-VC workflow for the ComfyUI sidebar (API form: workflows/seedvc_convert_api.json).
mkdir -p "$COMFYUI_DIR/user/default/workflows"
cp "$(dirname "$0")/../workflows/seedvc_convert.json" "$COMFYUI_DIR/user/default/workflows/seedvc_convert.json"

echo "tts nodes ready"
