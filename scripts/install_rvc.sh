#!/usr/bin/env bash
# RVC (issue #15) via Applio, in its own venv under /storage/tts-runtime (kept across restarts, separate
# from ComfyUI's venv). Pretrains are per-session in MODEL_ROOT/rvc: `download_tts.py --group rvc`
# (start.ipynb TTS_GROUPS). Trained models and logs stay in $APPLIO/logs/ on /storage.
set -euo pipefail

APPLIO=${APPLIO_DIR:-/storage/tts-runtime/applio}
MODEL_ROOT=${MODEL_ROOT:-/app/models}
UV=${UV:-$(command -v uv || echo /storage/.local/bin/uv)}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/app/tts-build/uv-cache}
APPLIO_COMMIT=f0a578575b3598a8c4d9e8deb30aca1474ab0ba9  # IAHispano/Applio 3.6.6
TORCH="torch==2.11.0+cu126 torchaudio==2.11.0+cu126"  # driver 550 = CUDA 12.4; PyPI's default torch is cu130

[ -d "$APPLIO" ] || git clone -q https://github.com/IAHispano/Applio "$APPLIO"
git -C "$APPLIO" fetch -q origin "$APPLIO_COMMIT" 2>/dev/null || true
git -C "$APPLIO" checkout -q "$APPLIO_COMMIT"

if [ ! -x "$APPLIO/.venv/bin/python" ]; then
  "$UV" venv -q --python 3.12 "$APPLIO/.venv"
  grep -vE '^(torch|torchaudio)==|sys_platform == "win32"|^#|^$' "$APPLIO/requirements-engine.txt" > "$APPLIO/.venv/requirements-linux.txt"
  # shellcheck disable=SC2086
  "$UV" pip install -q --python "$APPLIO/.venv/bin/python" -r "$APPLIO/.venv/requirements-linux.txt" $TORCH \
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
fi
"$APPLIO/.venv/bin/python" -c "import torch, faiss; print('applio ok: torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for sub in predictors embedders/contentvec pretraineds/hifi-gan; do
  mkdir -p "$MODEL_ROOT/rvc/$sub"
  link=$APPLIO/rvc/models/$sub
  [ -L "$link" ] || { rm -rf "$link"; ln -s "$MODEL_ROOT/rvc/$sub" "$link"; }
done
echo "rvc ready: $APPLIO"
