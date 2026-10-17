"""Build the ~10 s / ~30 s reference sets for two voices.

jsut_f : JSUT basic5000 (one female speaker, anechoic; non-commercial research use only)
cv_m   : one Common Voice 17 ja male speaker (CC0), chosen by WavLM speaker consistency

Each set is written as individual clips (Irodori --ref-wavs) plus one concatenated wav
with its transcript (drama3 voice_ref + reference_text). Both models hear the same audio.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from common import REF_DIR, SpeakerEmbedder, cosine

SR = 48000
GAP = 0.3
TARGETS = {"10s": 10.0, "30s": 30.0}


def trim(wav: np.ndarray, sr: int, thresh_db: float = -40.0, pad: float = 0.08) -> np.ndarray:
    hop = int(sr * 0.01)
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    idx = np.flatnonzero(db > thresh_db)
    if idx.size == 0:
        return wav
    s = max(0, idx[0] * hop - int(pad * sr))
    e = min(len(wav), (idx[-1] + 1) * hop + int(pad * sr))
    return wav[s:e]


def snr_db(wav: np.ndarray, sr: int) -> float:
    hop = int(sr * 0.02)
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    return float(20 * np.log10(np.percentile(rms, 95) / (np.percentile(rms, 10) + 1e-9)))


def resample(wav: np.ndarray, sr: int) -> np.ndarray:
    import torch
    import torchaudio.functional as F

    return wav if sr == SR else F.resample(torch.from_numpy(wav), sr, SR).numpy()


def write_sets(name: str, clips: list[tuple[str, np.ndarray, str]], source: dict) -> None:
    for label, target in TARGETS.items():
        out = REF_DIR / f"{name}_{label}"
        (out / "clips").mkdir(parents=True, exist_ok=True)
        chosen, total = [], 0.0
        for cid, wav, text in clips:
            if total >= target - 0.5:
                break
            chosen.append((cid, wav, text))
            total += len(wav) / SR
        gap = np.zeros(int(GAP * SR), dtype=np.float32)
        parts, paths = [], []
        for i, (cid, wav, _) in enumerate(chosen):
            p = out / "clips" / f"{i:02d}_{cid}.wav"
            sf.write(p, wav, SR, subtype="PCM_16")
            paths.append(str(p))
            parts += [wav, gap]
        concat = np.concatenate(parts[:-1]).astype(np.float32)
        sf.write(out / "concat.wav", concat, SR, subtype="PCM_16")
        meta = {
            "voice": name, "set": label, "sample_rate": SR,
            "speech_seconds": round(total, 2), "concat_seconds": round(len(concat) / SR, 2),
            "clips": paths, "concat": str(out / "concat.wav"),
            "clip_ids": [c[0] for c in chosen],
            "reference_text": "".join(c[2] for c in chosen),
            "source": source,
        }
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}_{label}: {len(chosen)} clips, {total:.1f}s speech")


def jsut(src: Path) -> None:
    trans = {}
    for line in (src / "basic5000" / "transcript_utf8.txt").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition(":")
        trans[k] = v
    clips = []
    for p in sorted((src / "basic5000").glob("BASIC5000_*.wav")):
        if p.stem not in trans:
            continue
        wav, sr = sf.read(p, dtype="float32")
        clips.append((p.stem, trim(resample(wav, sr), SR), trans[p.stem]))
    write_sets("jsut_f", clips, {
        "dataset": "JSUT basic5000 (FluidInference/JSUT-basic5000 @ cd13cc9c)",
        "license": "audio: academic / non-commercial research / personal use only; no redistribution",
    })


def cv(parquet: Path, sv: SpeakerEmbedder) -> None:
    import pyarrow.parquet as pq

    df = pq.read_table(parquet).to_pandas()
    key = list(zip(df.age, df.gender))
    runs, start = [], 0
    for i in range(1, len(df) + 1):
        if i == len(df) or key[i] != key[start]:
            runs.append((start, i))
            start = i
    best = None
    for s, e in runs:
        if df.gender[s] != "male_masculine" or e - s < 12:
            continue
        items = []
        for j in range(s, e):
            if int(df.down_votes[j]) > 0:
                continue
            wav, sr = sf.read(io.BytesIO(df.audio[j]["bytes"]), dtype="float32", always_2d=True)
            wav = trim(resample(wav.mean(axis=1), sr), SR)
            if len(wav) / SR < 2.0:
                continue
            items.append((df.id[j], wav, df.text[j], snr_db(wav, SR)))
        if len(items) < 10:
            continue
        embs = [sv.embed(load_audio_arr(w)) for _, w, _, _ in items]
        centroid = np.mean(embs, axis=0)
        sims = [cosine(x, centroid) for x in embs]
        # keep clips that clearly belong to the run's dominant speaker
        keep = [(it, sm) for it, sm in zip(items, sims) if sm > 0.9 and it[3] > 25]
        speech = sum(len(it[0][1]) / SR for it in keep)
        score = (np.mean([sm for _, sm in keep]) if keep else 0) + min(speech, 40) / 400 + (
            np.median([it[0][3] for it in keep]) / 200 if keep else 0)
        print(f"run {s}-{e} {key[s]} n={len(items)} keep={len(keep)} speech={speech:.1f}s "
              f"min_sim={min(sims):.3f} snr_med={np.median([i[3] for i in items]):.1f} score={score:.3f}")
        if speech >= 32 and (best is None or score > best[0]):
            best = (score, s, e, keep)
    assert best, "no suitable Common Voice male speaker run"
    _, s, e, keep = best
    keep.sort(key=lambda x: -x[0][3])  # cleanest clips first
    clips = [(cid, wav, text) for (cid, wav, text, _), _ in keep]
    write_sets("cv_m", clips, {
        "dataset": "Common Voice 17 ja (OpenSpeechHub/Common-Voice-17-Ja @ a68c1bde), shard 0",
        "rows": [s, e], "age_gender": list(key[s]),
        "license": "CC0-1.0",
        "speaker_check": "consecutive same age/gender rows; clips with WavLM cosine>0.9 to the run centroid",
    })


def load_audio_arr(wav48: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(wav48), SR, 16000).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsut", type=Path, default=Path("/app/tts-eval/src/jsut"))
    ap.add_argument("--cv-parquet", type=Path,
                    default=Path("/app/tts-eval/src/cv17/data/train-00000-of-00352.parquet"))
    args = ap.parse_args()
    jsut(args.jsut)
    cv(args.cv_parquet, SpeakerEmbedder())
