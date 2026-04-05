#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Configure it in Paperspace Secrets or environment variables."
        )

    whoami = subprocess.run(
        ["hf", "whoami", "--token", token],
        check=True,
        capture_output=True,
        text=True,
    )
    print(whoami.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
