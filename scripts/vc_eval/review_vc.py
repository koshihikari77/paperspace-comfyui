"""Listening page for the voice-conversion eval (OUT/review.html): per source clip, the source, both
target references and every converted take side by side with the automatic numbers. Players use
loudness-matched 48 kHz WAV copies in OUT/review_audio/. Run with the eval venv after score_vc.py.
"""
from __future__ import annotations

import csv
import html
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

from run_seedvc import OUT, SOURCES, TARGETS

SR = 48000
TARGET_DB = -23.0


def normalized(src: str) -> str:
    dst = OUT / "review_audio" / (Path(src).stem + ".wav")
    if not dst.exists():
        wav, sr = sf.read(src, dtype="float32", always_2d=True)
        wav = F.resample(torch.from_numpy(wav.mean(axis=1)), sr, SR).numpy() if sr != SR else wav.mean(axis=1)
        frames = wav[: len(wav) // 960 * 960].reshape(-1, 960)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        voiced = rms[rms > rms.max() * 10 ** (-30 / 20)]
        wav = wav * 10 ** ((TARGET_DB - 20 * np.log10(np.sqrt((voiced ** 2).mean()) + 1e-12)) / 20)
        wav = wav * min(1.0, 10 ** (-1 / 20) / (np.abs(wav).max() + 1e-12))
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, wav, SR, subtype="PCM_16")
    return os.path.relpath(dst, OUT)


def player(path: str, caption: str) -> str:
    return (f"<div class='take'><audio controls preload='none' src='{html.escape(normalized(path))}'></audio>"
            f"<br><span class='muted'>{caption}</span></div>")


def main() -> None:
    rows = list(csv.DictReader((OUT / "scores.csv").open(encoding="utf-8")))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["source"], r["target"], r["condition"])].append(r)
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    out = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width,initial-scale=1'><title>Seed-VC listening review</title><style>",
           ":root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--chip:#f2f2f7}",
           "@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#f2f2f7;--muted:#a1a1a6;--line:#3a3a3c;--chip:#2c2c2e}}",
           "body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 16px 64px}",
           "h1{font-size:20px}h2{font-size:16px;margin-top:32px;border-bottom:1px solid var(--line)}",
           "table{border-collapse:collapse;width:100%}td,th{border-top:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}",
           ".muted{color:var(--muted);font-size:12px}.take{margin-bottom:6px}audio{width:220px;height:32px}",
           "code{background:var(--chip);padding:1px 4px;border-radius:4px;font-size:12px}.wrap{overflow-x:auto}</style></head><body>",
           "<h1>Seed-VC listening review (issue #14)</h1>",
           "<p class='muted'>All voices are synthetic (drama3-small without reference audio). sim→tgt / sim→src: WavLM speaker "
           "similarity to the target reference / the source. F0: pitch-contour correlation with the source. HF: share of "
           "energy above 4 kHz relative to the source (breath kept ≈ 1). Players are loudness-matched copies.</p>",
           "<h2>Target references</h2><table><tr>"]
    out += [f"<td><b>{t}</b><br><span class='muted'>{html.escape(TARGETS[t]['desc'])}</span>"
            f"{player(str(OUT / 'inputs' / f'{t}.flac'), 'reference')}</td>" for t in TARGETS]
    out.append("</tr></table>")
    for src, text in SOURCES.items():
        out.append(f"<h2>{src}</h2><p><code>{html.escape(text)}</code></p>"
                   f"{player(str(OUT / 'inputs' / f'{src}.flac'), 'source')}<div class='wrap'><table><tr><th>target</th>")
        out += [f"<th>{c}</th>" for c in conditions]
        out.append("</tr>")
        for tgt in TARGETS:
            out.append(f"<tr><td>{tgt}</td>")
            for c in conditions:
                takes = sorted(cells[(src, tgt, c)], key=lambda r: r["take"])
                out.append("<td>" + "".join(player(r["wav"], f"t{r['take']} · sim→tgt {r['sim_target']} · sim→src {r['sim_source']}"
                                                              f" · F0 {r['f0_corr']} · HF {r['hf_ratio']}") for r in takes) + "</td>")
            out.append("</tr>")
        out.append("</table></div>")
    out.append("</body></html>")
    (OUT / "review.html").write_text("\n".join(out), encoding="utf-8")
    print(OUT / "review.html")


if __name__ == "__main__":
    main()
