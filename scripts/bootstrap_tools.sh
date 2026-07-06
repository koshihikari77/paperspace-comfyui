#!/usr/bin/env bash
set -euo pipefail

template="/notebooks/env/env_tools.example.sh"
target="/storage/env_tools.sh"

if [[ ! -f "$template" ]]; then
  echo "template not found: $template" >&2
  exit 1
fi

if [[ -e "$target" ]]; then
  echo "already exists: $target"
  echo "edit it manually if you want to update it from the template"
  exit 0
fi

install -D -m 0644 "$template" "$target"
echo "created: $target"
echo "next:"
echo "  1. edit Git identity values"
echo "  2. run: source $target"
