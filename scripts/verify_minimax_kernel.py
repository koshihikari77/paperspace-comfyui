#!/usr/bin/env python3
"""Verify the locally built CUDA 12.8 kernels before enabling H3 on A6000."""

import hashlib
import json
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda

MANIFEST = Path("/storage/h3-research/issue6/cuda128-build-manifest.json")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    module = Path(cuda._C.__file__)
    actual = hashlib.sha256(module.read_bytes()).hexdigest()
    if actual != manifest["extension_sha256"]:
        raise RuntimeError(f"CUDA kernel hash mismatch: {module}")
    tests = manifest["tests"]
    if tests["tests"] < 1 or any(tests[key] for key in ("failures", "errors", "skipped")):
        raise RuntimeError("CUDA kernel test manifest is not passing")
    if not torch.version.cuda or not torch.version.cuda.startswith("12."):
        raise RuntimeError(f"Expected CUDA 12 PyTorch, got {torch.version.cuda}")
    if torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("This CUDA kernel was built for A6000 SM86 only")
    print(f"Verified MiniMax CUDA 12.8 kernel: {module}")


if __name__ == "__main__":
    main()
