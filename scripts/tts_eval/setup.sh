#!/usr/bin/env bash
# Reproducible setup for issue #13 (drama3-small vs Irodori-TTS) on Paperspace A6000.
# Models are per-session (/app/models/tts, re-downloaded every session, never /storage).
# The runtime (patched audio.cpp binaries, Irodori venv) persists in /storage/tts-runtime, so after
# a restart only the model downloads run. ComfyUI's venv is not touched.
set -euo pipefail

MODEL_ROOT=${TTS_MODEL_ROOT:-/app/models/tts}
BUILD=${TTS_BUILD_DIR:-/storage/tts-runtime}
UV=${UV:-$(command -v uv || echo /storage/.local/bin/uv)}
export HF_HOME=$MODEL_ROOT/hf
export UV_CACHE_DIR=${UV_CACHE_DIR:-/app/tts-build/uv-cache}
mkdir -p "$MODEL_ROOT" "$BUILD"
SERVER_BIN=$BUILD/audio.cpp/build/linux-cuda-release/bin/audiocpp_server

DRAMA3_REV=4a3078e885344bb9b24d511cec389d2726854d15
AUDIOCPP_COMMIT=9277495adfdc9b41004bd91db7b7a4e5d456c8df
PATCH_SHA256=cb71f86c81158db43e29a93626ae383c1b2f1d957304e700d7e0a174f6334119
IRODORI_COMMIT=89f9d8fbd4d51ea019867ee1197725ede1df13c5

# 1. CUDA 12.4 compiler + dev libraries, only needed to (re)build audio.cpp
#    (the image ships only the CUDA runtime, which the built binaries link against)
if [ ! -x "$SERVER_BIN" ] && [ ! -x /usr/local/cuda-12.4/bin/nvcc ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    cuda-nvcc-12-4 cuda-cudart-dev-12-4 libcublas-dev-12-4 cuda-cccl-12-4 cuda-nvrtc-dev-12-4 \
    cuda-driver-dev-12-4 libcufft-dev-12-4 libcurand-dev-12-4 cuda-nvtx-12-4 cuda-profiler-api-12-4
fi

# 2. drama3-small GGUF + patch (pinned revision)
hf download pentacoxian-dev/drama3-small-GGUF --revision "$DRAMA3_REV" --local-dir "$MODEL_ROOT/drama3-small-GGUF"
echo "$PATCH_SHA256  $MODEL_ROOT/drama3-small-GGUF/drama3-small-audiocpp.patch" | sha256sum -c -

# 3. audio.cpp at the author's commit (not on any branch; fetch by SHA) + patch, CUDA sm_86 build
if [ ! -x "$SERVER_BIN" ]; then
  [ -d "$BUILD/audio.cpp" ] || git clone -q https://github.com/JarodMica/audio.cpp "$BUILD/audio.cpp"
  git -C "$BUILD/audio.cpp" fetch -q origin "$AUDIOCPP_COMMIT"
  git -C "$BUILD/audio.cpp" checkout -q "$AUDIOCPP_COMMIT"
  git -C "$BUILD/audio.cpp" apply --check "$MODEL_ROOT/drama3-small-GGUF/drama3-small-audiocpp.patch" \
    && git -C "$BUILD/audio.cpp" apply "$MODEL_ROOT/drama3-small-GGUF/drama3-small-audiocpp.patch"
  (cd "$BUILD/audio.cpp" && PATH=/usr/local/cuda-12.4/bin:$PATH CUDACXX=/usr/local/cuda-12.4/bin/nvcc \
    scripts/build_linux.sh --cuda on --cuda-arch 86 --models fish_drama3)
fi

# 4. Irodori-TTS in its own uv venv (torch 2.10 cu128), kept in /storage
if [ ! -x "$BUILD/Irodori-TTS/.venv/bin/python" ]; then
  [ -d "$BUILD/Irodori-TTS" ] || git clone -q https://github.com/Aratako/Irodori-TTS "$BUILD/Irodori-TTS"
  git -C "$BUILD/Irodori-TTS" checkout -q "$IRODORI_COMMIT"
  (cd "$BUILD/Irodori-TTS" && "$UV" sync --extra cu128 && \
    "$UV" pip install --python .venv/bin/python jiwer pykakasi nvidia-ml-py psutil)
fi

# 5. Irodori checkpoints, codec, and the scoring models (Whisper, WavLM-SV), all pinned
hf download Aratako/Irodori-TTS-v4.1-Small --revision 2b28324dc263ed5e6638b3cf3dd94c82ead07b4b
hf download Aratako/Irodori-TTS-v4.1-Small-MF --revision ccc78f5d480b6e51b69b2d5042a14c4da04fea6e
hf download phasefield-audio/Irodori-TTS-v4.1-Anime --revision 6b259f5baa5e236b3d14cbd1f8555ca87d92b530 \
  README.md model.safetensors tokenizer/tokenizer.json tokenizer/tokenizer_config.json
hf download Aratako/Semantic-DACVAE-Japanese-32dim --revision 47376ee24834d7a05a48ebabfe3cde29b3c5e214
hf download openai/whisper-large-v3-turbo --revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 --exclude "*.bin" "*.msgpack" "*.h5"
hf download microsoft/wavlm-base-plus-sv --revision feb593a6c23c1cc3d9510425c29b0a14d2b07b1e

# 6. Reference speech sources (data, not models; kept under /app as well)
mkdir -p /app/tts-eval/src/jsut /app/tts-eval/src/cv17
hf download FluidInference/JSUT-basic5000 --repo-type dataset --revision cd13cc9c9fe70d8d4f62ffe6fbbb872aef79835c \
  --local-dir /app/tts-eval/src/jsut LICENCE.txt README_ja.txt basic5000/transcript_utf8.txt \
  $(for i in $(seq 4501 4530); do echo basic5000/BASIC5000_$i.wav; done)
hf download OpenSpeechHub/Common-Voice-17-Ja --repo-type dataset --revision a68c1bde8bbf3ca874bd67fd3dac1bc5de72296d \
  --local-dir /app/tts-eval/src/cv17 README.md data/train-00000-of-00352.parquet

echo "setup complete"
