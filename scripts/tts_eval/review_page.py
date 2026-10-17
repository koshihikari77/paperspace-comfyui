"""Build a listening page (EVAL_DIR/review.html) from scores.csv: one row per line, one column per
model, every seed playable side by side with its objective numbers. Open it from JupyterLab
(right click -> Open in New Browser Tab) so the relative audio paths resolve.
"""
from __future__ import annotations

import csv
import html
import os
from collections import defaultdict
from pathlib import Path

from common import EVAL_DIR, load_script

ORDER = ["drama3-small", "irodori-small", "irodori-mf", "irodori-anime", "drama3-small-nofix"]
SR = 48000
TARGET_DB = -23.0  # RMS of the voiced frames


def normalized_copy(src: str) -> Path:
    """Same loudness (voiced-frame RMS -23 dBFS, peak <= -1 dBFS) and 48 kHz for fair listening."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    dst = EVAL_DIR / "review_audio" / Path(src).relative_to(EVAL_DIR / "takes")
    if dst.exists():
        return dst
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = F.resample(torch.from_numpy(wav), sr, SR).numpy()
    hop = SR // 50
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    voiced = rms[rms > rms.max() * 10 ** (-30 / 20)] if rms.size else rms
    level = 20 * np.log10(np.sqrt((voiced**2).mean()) + 1e-12) if voiced.size else TARGET_DB
    wav = wav * 10 ** ((TARGET_DB - level) / 20)
    peak = np.abs(wav).max() + 1e-12
    if peak > 10 ** (-1 / 20):
        wav = wav * (10 ** (-1 / 20) / peak)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, wav, SR, subtype="PCM_16")
    return dst


def main() -> None:
    rows = list(csv.DictReader((EVAL_DIR / "scores.csv").open(encoding="utf-8")))
    lines = load_script()["lines"]
    cells = defaultdict(list)
    for r in rows:
        if r.get("wav"):
            cells[(r["voice"], r["refs"], r["line"], r["model"])].append(r)
    models = [m for m in ORDER if any(k[3] == m for k in cells)] + sorted(
        {k[3] for k in cells} - set(ORDER))
    combos = sorted({(k[0], k[1]) for k in cells}, key=lambda x: (x[1] != "30s", x[0]))

    out = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width,initial-scale=1'>",
           "<title>TTS listening review</title><style>",
           ":root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--bad:#c62828;--chip:#f2f2f7}",
           "@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#f2f2f7;--muted:#a1a1a6;--line:#3a3a3c;--bad:#ff6b6b;--chip:#2c2c2e}}",
           "body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 16px 64px}",
           "h1{font-size:20px}h2{font-size:16px;margin-top:32px;border-bottom:1px solid var(--line)}",
           "table{border-collapse:collapse;width:100%}td,th{border-top:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}",
           ".txt{max-width:280px}.muted{color:var(--muted);font-size:12px}.take{margin-bottom:6px}",
           ".bad{color:var(--bad);font-weight:600}audio{width:220px;height:32px}",
           "code{background:var(--chip);padding:1px 4px;border-radius:4px;font-size:12px}",
           ".wrap{overflow-x:auto}</style></head><body>",
           "<h1>drama3-small vs Irodori-TTS listening review</h1>",
           "<p class='muted'>kana CER / speaker similarity / fail flags are automatic (Whisper large-v3-turbo, WavLM-SV). "
           "Judge naturalness, acting and whether tags were followed by ear. "
           "Players use loudness-matched 48 kHz copies (review_audio/); the scores use the raw outputs.</p>"]
    for voice, refs in combos:
        out.append(f"<h2>{html.escape(voice)} · reference {html.escape(refs)}</h2><div class='wrap'><table><tr><th>line</th>")
        out += [f"<th>{html.escape(m)}</th>" for m in models]
        out.append("</tr>")
        for l in lines:
            out.append(f"<tr><td class='txt'><b>{l['id']}</b> <span class='muted'>{html.escape(l['category'])}</span><br>"
                       f"{html.escape(l['text'])}<br><span class='muted'>drama3: <code>{html.escape(l['drama3'])}</code><br>"
                       f"irodori: <code>{html.escape(l['irodori'])}</code></span></td>")
            for m in models:
                takes = sorted(cells.get((voice, refs, l["id"], m), []), key=lambda r: int(r["seed"]))
                out.append("<td>")
                for r in takes:
                    rel = os.path.relpath(normalized_copy(r["wav"]), EVAL_DIR)
                    flag = f" <span class='bad'>{html.escape(r['fail_reason'])}</span>" if r.get("fail_reason") else ""
                    out.append(f"<div class='take'><audio controls preload='none' src='{html.escape(rel)}'></audio><br>"
                               f"<span class='muted'>s{r['seed']} · {r['audio_sec']}s · CER {r['kana_cer']} · sim {r['spk_sim']}</span>{flag}</div>")
                out.append("</td>")
            out.append("</tr>")
        out.append("</table></div>")
    out.append("</body></html>")
    (EVAL_DIR / "review.html").write_text("\n".join(out), encoding="utf-8")
    print(EVAL_DIR / "review.html")


if __name__ == "__main__":
    main()
