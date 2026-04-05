#!/usr/bin/env bash

set -euo pipefail

mkdir -p /notebooks
cp -n /app/bootstrap/start.ipynb /notebooks/start.ipynb || true
cp -n /app/bootstrap/hf-repo.example.yaml /notebooks/hf-repo.example.yaml || true

exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.token='' \
  --ServerApp.password=''
