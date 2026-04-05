#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_TOOL="${CONTAINER_TOOL:-podman}"

usage() {
    cat <<'EOF'
Usage:
  ./docker/build-and-push.sh <image[:tag]> [--push]

Examples:
  ./docker/build-and-push.sh yourname/comfyui-paperspace:latest
  ./docker/build-and-push.sh ghcr.io/yourname/comfyui-paperspace:2026-04-05 --push
EOF
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    usage
    exit 1
fi

IMAGE_REF="$1"
PUSH_IMAGE="false"

if [ "${2:-}" = "--push" ]; then
    PUSH_IMAGE="true"
elif [ $# -eq 2 ]; then
    usage
    exit 1
fi

echo "Building image with ${CONTAINER_TOOL}"
echo "Image: ${IMAGE_REF}"

"${CONTAINER_TOOL}" build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_REF}" \
    "${REPO_ROOT}"

echo
echo "Build completed: ${IMAGE_REF}"

if [ "${PUSH_IMAGE}" = "true" ]; then
    echo "Pushing image: ${IMAGE_REF}"
    "${CONTAINER_TOOL}" push "${IMAGE_REF}"
fi
