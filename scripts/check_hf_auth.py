#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess


def run_whoami(extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    commands = (
        ["hf", "auth", "whoami", *extra_args],
        ["hf", "whoami", *extra_args],
    )
    last_error: subprocess.CalledProcessError | None = None
    for cmd in commands:
        try:
            return subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            last_error = exc

    if last_error is None:
        raise SystemExit("Unable to run Hugging Face CLI")
    raise last_error


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    extra_args = ["--token", token] if token else []
    try:
        whoami = run_whoami(extra_args)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise SystemExit(
            "Hugging Face authentication is not available. "
            "Run `hf auth login` once with `HF_HOME` pointed at persistent storage, "
            "or set HF_TOKEN.\n"
            f"{stderr}"
        ) from exc

    print(whoami.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
