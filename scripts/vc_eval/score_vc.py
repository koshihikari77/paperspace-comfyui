"""Score voice-conversion outputs (issue #14). Run with the eval venv (/storage/tts-runtime/Irodori-TTS/.venv).

Per output: WavLM speaker similarity to the target reference and to the source (lower = the source voice
was replaced), F0 contour correlation with the source (the delivery was kept), duration ratio, share of
energy above 4 kHz relative to the source (breath/noise kept), peak and voiced-frame loudness, and kana
CER against the source text for the speech clips.

  python score_vc.py [log.jsonl]     ->  scores.csv, summary.md next to the log
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tts_eval"))
from common import SpeakerEmbedder, Whisper, cer, cosine, load_audio, to_kana  # noqa: E402
from run_seedvc import OUT, SOURCES  # noqa: E402

SR = 16000


def f0_track(wav: np.ndarray) -> np.ndarray:
    f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=1000, sr=SR, frame_length=1024, hop_length=160)
    return np.where(voiced, f0, np.nan)


def f0_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of log-F0 over frames voiced in both; the tracks are aligned by stretching b to a."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    b = np.interp(np.linspace(0, len(b) - 1, len(a)), np.arange(len(b)), b)
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 10:
        return float("nan")
    return float(np.corrcoef(np.log(a[m]), np.log(b[m]))[0, 1])


def hf_share(wav: np.ndarray) -> float:
    spec = np.abs(librosa.stft(wav, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    return float(spec[freqs >= 4000].sum() / (spec.sum() + 1e-12))


def loudness(wav: np.ndarray) -> tuple[float, float]:
    frames = wav[: len(wav) // 320 * 320].reshape(-1, 320)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    voiced = rms[rms > rms.max() * 10 ** (-30 / 20)]
    return (round(float(20 * np.log10(np.abs(wav).max() + 1e-12)), 1),
            round(float(20 * np.log10(np.sqrt((voiced ** 2).mean()) + 1e-12)), 1))


def plain(text: str) -> str:
    return re.sub(r"\[[^\]]*\]|<[^>]*>", "", text)


def main() -> None:
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "log.jsonl"
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    sv, asr = SpeakerEmbedder(), Whisper()
    cache: dict[str, dict] = {}

    def clip(name: str) -> dict:
        if name not in cache:
            wav = load_audio(str(OUT / "inputs" / f"{name}.flac"))
            cache[name] = dict(wav=wav, emb=sv.embed(wav), f0=f0_track(wav), hf=hf_share(wav))
        return cache[name]

    rows = []
    for r in recs:
        src, tgt = clip(r["source"]), clip(r["target"])
        wav = load_audio(r["wav"])
        emb = sv.embed(wav)
        peak, level = loudness(wav)
        row = {k: r[k] for k in ("engine", "source", "target", "condition", "take", "wall_sec")}
        row.update(sim_target=round(float(cosine(emb, tgt["emb"])), 4), sim_source=round(float(cosine(emb, src["emb"])), 4),
                   src_sim_target=round(float(cosine(src["emb"], tgt["emb"])), 4),
                   f0_corr=round(f0_corr(src["f0"], f0_track(wav)), 3),
                   dur_ratio=round(len(wav) / len(src["wav"]), 3),
                   hf_ratio=round(hf_share(wav) / (src["hf"] + 1e-12), 3), peak_db=peak, level_db=level,
                   kana_cer="", wav=r["wav"])
        if r["source"].startswith("speech_"):
            row["kana_cer"] = round(float(cer(to_kana(plain(SOURCES[r["source"]])), to_kana(asr.transcribe(wav)))), 3)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with (log.parent / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def mean(xs):
        xs = [x for x in xs if x != "" and not (isinstance(x, float) and np.isnan(x))]
        return round(float(np.mean(xs)), 3) if xs else "-"

    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], "speech" if row["source"].startswith("speech_") else "non-verbal")].append(row)
    lines = ["| condition | source | n | sim→target | sim→source | source→target (before) | F0 corr | dur ratio | HF ratio | kana CER | sec/clip |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (cond, kind), g in sorted(groups.items()):
        lines.append(f"| {cond} | {kind} | {len(g)} | {mean(x['sim_target'] for x in g)} | {mean(x['sim_source'] for x in g)} | "
                     f"{mean(x['src_sim_target'] for x in g)} | {mean(x['f0_corr'] for x in g)} | {mean(x['dur_ratio'] for x in g)} | "
                     f"{mean(x['hf_ratio'] for x in g)} | {mean(x['kana_cer'] for x in g)} | {mean(x['wall_sec'] for x in g)} |")
    (log.parent / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
