#!/usr/bin/env bash
# Clean speed/VRAM benchmark: one engine at a time on an otherwise idle GPU.
# Lines follow the drama3 model card's speed set (JA short, JA long, JA tags, JA+English).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/app/models/tts/hf}
export TTS_REF_DIR=${TTS_REF_DIR:-/notebooks/workspace/tts_eval/refs}
export TTS_EVAL_DIR=${TTS_EVAL_DIR:-/notebooks/workspace/tts_eval/bench}
PY=${PY:-/storage/tts-runtime/Irodori-TTS/.venv/bin/python}
LINES="d01 l01 n03 k03"
for refs in 30s 10s; do
  $PY run_drama3.py --refs $refs --voices jsut_f --seeds 0 1 2 --lines $LINES
  $PY run_irodori.py --model small --refs $refs --voices jsut_f --seeds 0 1 2 --lines $LINES
done
$PY run_drama3.py --mode streaming --refs 30s --voices jsut_f --seeds 0 1 2 --lines $LINES
$PY run_irodori.py --model small --precision bf16 --refs 30s --voices jsut_f --seeds 0 1 2 --lines $LINES
$PY run_irodori.py --model mf --refs 30s --voices jsut_f --seeds 0 1 2 --lines $LINES
$PY run_irodori.py --model anime --refs 30s --voices jsut_f --seeds 0 1 2 --lines $LINES
