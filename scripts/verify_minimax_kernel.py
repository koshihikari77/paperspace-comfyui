#!/usr/bin/env python3
"""Verify the locally built CUDA 12.8 kernels before enabling H3 on A6000."""

import argparse
import importlib.util
import hashlib
import json
from pathlib import Path

MANIFEST = Path("/storage/h3-research/issue6/cuda128-build-manifest.json")


def prepare_library_dir(directory: Path) -> None:
    spec = importlib.util.find_spec("nvidia.cublas")
    if spec is None or spec.origin is None:
        raise RuntimeError("The ComfyUI Python environment has no nvidia.cublas package")
    library = Path(spec.origin).parent / "lib/libcublasLt.so.12"
    if not library.is_file():
        raise RuntimeError(f"CUDA 12 cuBLASLt library missing: {library}")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    alias = directory / "libcublasLt.so"
    if alias.is_symlink():
        if alias.resolve() != library.resolve():
            alias.unlink()
    elif alias.exists():
        raise RuntimeError(f"Refusing to replace a non-symlink library: {alias}")
    if not alias.is_symlink():
        alias.symlink_to(library.resolve())
    print(directory)


def verify() -> None:
    # Imported only after the parent launcher prepares LD_LIBRARY_PATH.
    import torch
    from comfy_kitchen.backends import cuda
    from comfy_kitchen.registry import registry

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
    if not cuda._CUBLASLT_AVAILABLE:
        raise RuntimeError("cuBLASLt unavailable: prepare libcublasLt.so and set LD_LIBRARY_PATH before launching Python")
    x = torch.ones((32, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((256, 512), device="cuda", dtype=torch.int8)
    scale = torch.full((256,), 1 / 512, device="cuda", dtype=torch.float32)
    kwargs = {"x": x, "weight": weight, "weight_scale": scale}
    implementation = registry.get_implementation("int8_linear", kwargs=kwargs)
    if implementation is not cuda.int8_linear:
        raise RuntimeError("int8_linear is not dispatched to CUDA; refusing a silent slower fallback")
    output = implementation(**kwargs)
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0.01, atol=0.01)
    rotated = implementation(**kwargs, convrot=True, convrot_groupsize=256)
    if not torch.isfinite(rotated).all().item():
        raise RuntimeError("CUDA INT8 ConvRot produced non-finite output")
    torch.cuda.synchronize()
    print(f"Verified MiniMax CUDA 12.8 kernel: {module}")
    print("Verified cuBLASLt and CUDA int8_linear dispatch/GEMM/ConvRot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-library-dir", type=Path, help="Create a process-local CUDA 12 cuBLASLt alias and print its directory; does not initialize CUDA")
    args = parser.parse_args()
    if args.prepare_library_dir:
        prepare_library_dir(args.prepare_library_dir)
    else:
        verify()


if __name__ == "__main__":
    main()
