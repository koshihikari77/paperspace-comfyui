"""RVC (Applio) conversions of the #14 source clips into the trained synthetic voice tgt_f (issue #15).
Writes OUT/rvc/log.jsonl in the same format as run_seedvc.py, so score_vc.py and review_vc.py apply as is.

  python run_rvc.py [model_name]   (any python; it calls Applio's own venv)
"""
from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from run_seedvc import OUT, SOURCES

APPLIO = Path("/storage/tts-runtime/applio")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "vc_eval_tgt_f"
CONDITIONS = [  # (label, index_rate, protect)
    ("noindex_p033", 0.0, 0.33),
    ("index075_p033", 0.75, 0.33),
    ("index075_p050", 0.75, 0.5),  # protect 0.5 = no protection: keeps breath/unvoiced untouched
]


def main() -> None:
    logs = APPLIO / "logs" / MODEL
    weights = sorted(logs.glob(f"{MODEL}_*e_*s.pth"), key=lambda p: int(re.search(r"_(\d+)e_", p.name).group(1)))
    index = next(logs.glob("*.index"))
    out = OUT / "rvc"
    (out / "wav").mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    for pth, src, (label, index_rate, protect) in itertools.product(weights, SOURCES, CONDITIONS):
        epoch = int(re.search(r"_(\d+)e_", pth.name).group(1))
        dst = out / "wav" / f"{src}__tgt_f__e{epoch}_{label}.wav"
        t = time.time()
        subprocess.run([str(APPLIO / ".venv/bin/python"), "core.py", "infer", "--input-path",
                        str(OUT / "inputs" / f"{src}.flac"), "--output-path", str(dst), "--pth-path", str(pth),
                        "--index-path", str(index), "--index-rate", str(index_rate), "--protect", str(protect),
                        "--f0-method", "rmvpe", "--embedder-model", "contentvec", "--export-format", "WAV"],
                       cwd=APPLIO, check=True, capture_output=True)
        rec = dict(engine="rvc", source=src, target="tgt_f", condition=f"e{epoch}_{label}", epoch=epoch,
                   index_rate=index_rate, protect=protect, take=0, wall_sec=round(time.time() - t, 2), wav=str(dst))
        print(rec, flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
