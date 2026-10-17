"""Objective scoring of every take: Whisper CER, WavLM speaker similarity, pauses, failure flags.

  /storage/tts-runtime/Irodori-TTS/.venv/bin/python scripts/tts_eval/score.py
Writes EVAL_DIR/scores.csv (one row per take) and EVAL_DIR/summary.md.
CER is auxiliary: Whisper normalises numbers/kanji differently, so kana CER is the main number.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (EVAL_DIR, REF_DIR, SpeakerEmbedder, Whisper, cer, cosine, load_audio,
                    load_script, silence_stats, to_kana, to_surface)

# Words a model may read aloud when it fails to treat a control tag as an instruction.
TAG_WORDS = re.compile(r"laugh|sigh|whisper|angr|excit|soft|shy|sad|tear|calm|teas|panic|gentl|determin|"
                       r"flust|confus|relie|voice|shout|nostalg|笑い|ため息|ささや|囁", re.I)
FAIL_KANA_CER = 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--rescore", action="store_true", help="ignore the cache and recompute everything")
    args = ap.parse_args()

    lines = {l["id"]: l for l in load_script()["lines"]}
    cache_path = args.eval_dir / "score_cache.jsonl"
    cache = {}
    if cache_path.exists() and not args.rescore:
        for row in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(row)
            cache[r["wav"]] = r

    takes = []
    for log in sorted((args.eval_dir / "takes").glob("*/log_*.jsonl")):
        for row in log.read_text(encoding="utf-8").splitlines():
            takes.append(json.loads(row))

    asr = sv = None
    ref_emb = {}
    out_cache = cache_path.open("a", encoding="utf-8")
    rows = []
    for t in takes:
        row = {k: t.get(k) for k in ("model", "voice", "refs", "line", "category", "seed", "ok", "cold",
                                     "audio_sec", "wall_sec", "rtf", "xrt", "ttfa_sec", "gpu_used_peak_mib", "gpu_util_mean",
                                     "torch_peak_mib", "rss_peak_mib", "server_rss_peak_mib", "load_sec",
                                     "load_gpu_mib", "wav", "error")}
        if not t.get("ok"):
            row["fail_reason"] = "error"
            rows.append(row)
            continue
        if t["wav"] in cache:
            row.update(cache[t["wav"]])
            rows.append(row)
            continue
        if asr is None:
            asr, sv = Whisper(), SpeakerEmbedder()
        if t["voice"] not in ref_emb:
            # identity target = the voice's 30 s set, so 10 s and 30 s takes are judged against the same voice
            meta = json.loads((REF_DIR / f"{t['voice']}_30s" / "meta.json").read_text(encoding="utf-8"))
            ref_emb[t["voice"]] = sv.embed(load_audio(meta["concat"]))
        wav = load_audio(t["wav"])
        hyp = asr.transcribe(wav)
        target = lines[t["line"]]["text"]
        st = silence_stats(wav, 16000)
        emb = sv.embed(wav)
        sc = {
            "wav": t["wav"], "asr": hyp,
            "kana_cer": round(cer(to_kana(target), to_kana(hyp)), 4),
            "cer": round(cer(to_surface(target), to_surface(hyp)), 4),
            "spk_sim": round(cosine(emb, ref_emb[t["voice"]]), 4),
            "longest_pause_s": round(st["longest_pause_s"], 2), "tail_s": round(st["tail_s"], 2),
            "audible_ratio": round(st["audible_ratio"], 3), "peak_dbfs": round(st["peak_dbfs"], 1),
            "tag_read": bool(TAG_WORDS.search(hyp)) and not TAG_WORDS.search(target),
            "emb": emb.round(5).tolist(),
        }
        out_cache.write(json.dumps(sc, ensure_ascii=False) + "\n")
        out_cache.flush()
        row.update(sc)
        rows.append(row)
        print(f"{t['model']} {t['voice']}_{t['refs']} {t['line']} s{t['seed']} kana_cer={sc['kana_cer']} "
              f"sim={sc['spk_sim']} | {hyp}", flush=True)
    out_cache.close()

    # expected length per line = median Irodori-Small duration (its duration predictor is deterministic
    # given text + reference), used only to flag takes that run far past the spoken content
    durs = defaultdict(list)
    for r in rows:
        if r.get("model") == "irodori-small" and r.get("audio_sec"):
            durs[(r["voice"], r["line"])].append(float(r["audio_sec"]))
    expected = {k: float(np.median(v)) for k, v in durs.items()}
    for r in rows:
        if r.get("ok") and "kana_cer" in r:
            chars = len(to_kana(lines[r["line"]]["text"]))
            exp = expected.get((r["voice"], r["line"]))
            r["dur_ratio"] = round(r["audio_sec"] / exp, 3) if exp else float("nan")
            r["sec_per_kana"] = round(r["audio_sec"] / max(chars, 1), 3)
            reasons = []
            if r["kana_cer"] > FAIL_KANA_CER:
                reasons.append("content")  # misread / dropped / repeated
            if r["longest_pause_s"] > 3.0:
                reasons.append("long_pause")
            if r["sec_per_kana"] > 0.6:
                reasons.append("runaway")
            elif exp and r["dur_ratio"] > 1.8:
                reasons.append("overlong")
            if r["tag_read"]:
                reasons.append("tag_read")
            if r.get("peak_dbfs") is not None and r["peak_dbfs"] < -35:
                reasons.append("inaudible")
            r["fail_reason"] = "+".join(reasons)

    # take-to-take speaker consistency: mean pairwise cosine across seeds of the same line
    groups = defaultdict(list)
    for r in rows:
        if r.get("emb"):
            groups[(r["model"], r["voice"], r["refs"], r["line"])].append(np.array(r["emb"]))
    consistency = {}
    for k, embs in groups.items():
        if len(embs) > 1:
            consistency[k] = float(np.mean([cosine(a, b) for a, b in itertools.combinations(embs, 2)]))
    for r in rows:
        r["seed_consistency"] = round(consistency.get((r["model"], r["voice"], r["refs"], r["line"]), float("nan")), 4)

    fields = ["model", "voice", "refs", "line", "category", "seed", "ok", "fail_reason", "kana_cer", "cer",
              "spk_sim", "seed_consistency", "dur_ratio", "audio_sec", "wall_sec", "rtf", "xrt", "ttfa_sec", "sec_per_kana",
              "longest_pause_s", "tail_s", "audible_ratio", "peak_dbfs", "tag_read", "gpu_used_peak_mib", "gpu_util_mean", "torch_peak_mib", "rss_peak_mib",
              "server_rss_peak_mib", "cold", "load_sec", "load_gpu_mib", "asr", "wav", "error"]
    with (args.eval_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, args.eval_dir / "summary.md")


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def write_summary(rows: list[dict], path: Path) -> None:
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["refs"])].append(r)
    out = ["# drama3-small vs Irodori-TTS: objective summary", "",
           "Warm takes only for speed (the first request after load is excluded). "
           "fail = kana CER > 0.25, internal pause > 3 s, > 0.6 s per kana (runaway), a tag read aloud, peak below -35 dBFS (inaudible), or > 1.8x the median Irodori-Small length of the same line (overlong).", "",
           "| model | refs | takes | fail | fail % | kana CER | CER | spk sim | seed cons. | xRT warm | RTF warm | wall s (mean) | GPU peak MiB |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (model, refs), rs in sorted(by.items()):
        ok = [r for r in rs if r.get("kana_cer") is not None]
        fails = [r for r in rs if r.get("fail_reason")]
        warm = [r for r in ok if not r.get("cold")]
        out.append(
            f"| {model} | {refs} | {len(rs)} | {len(fails)} | {100 * len(fails) / max(len(rs), 1):.1f} | "
            f"{_mean([r['kana_cer'] for r in ok]):.3f} | {_mean([r['cer'] for r in ok]):.3f} | "
            f"{_mean([r['spk_sim'] for r in ok]):.3f} | {_mean([r['seed_consistency'] for r in ok]):.3f} | "
            f"{_mean([r['xrt'] for r in warm]):.2f} | {_mean([r['rtf'] for r in warm]):.3f} | "
            f"{_mean([r['wall_sec'] for r in warm]):.2f} | {max([r.get('gpu_used_peak_mib') or 0 for r in rs])} |")
    out += ["", "## Failures by category (count of failed takes / takes)", ""]
    cats = sorted({r["category"] for r in rows})
    keys = sorted(by)
    out.append("| category | " + " | ".join(f"{m} {rf}" for m, rf in keys) + " |")
    out.append("|---|" + "---:|" * len(keys))
    for c in cats:
        cells = []
        for k in keys:
            rs = [r for r in by[k] if r["category"] == c]
            cells.append(f"{sum(1 for r in rs if r.get('fail_reason'))}/{len(rs)}")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out += ["", "## Failure reasons", ""]
    for k in keys:
        reasons = defaultdict(int)
        for r in by[k]:
            for x in (r.get("fail_reason") or "").split("+"):
                if x:
                    reasons[x] += 1
        out.append(f"- {k[0]} {k[1]}: " + (", ".join(f"{a}={b}" for a, b in sorted(reasons.items())) or "none"))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
