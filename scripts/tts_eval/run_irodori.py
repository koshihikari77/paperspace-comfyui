"""Generate the comparison takes with Irodori-TTS v4.1 (Small / Small-MF / Anime).

Run with the Irodori venv:
  /storage/tts-runtime/Irodori-TTS/.venv/bin/python scripts/tts_eval/run_irodori.py --model small --refs 30s
The model is loaded once; the first request is a separate warm-up and is logged as cold.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from common import EVAL_DIR, REF_DIR, GpuMemSampler, load_script, peak_rss_mib

IRODORI_SRC = Path(os.environ.get("IRODORI_SRC", "/storage/tts-runtime/Irodori-TTS"))
sys.path.insert(0, str(IRODORI_SRC))

MODELS = {
    "small": ("Aratako/Irodori-TTS-v4.1-Small", "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b"),
    "mf": ("Aratako/Irodori-TTS-v4.1-Small-MF", "ccc78f5d480b6e51b69b2d5042a14c4da04fea6e"),
    "anime": ("phasefield-audio/Irodori-TTS-v4.1-Anime", "6b259f5baa5e236b3d14cbd1f8555ca87d92b530"),
}
CODEC = ("Aratako/Semantic-DACVAE-Japanese-32dim", "47376ee24834d7a05a48ebabfe3cde29b3c5e214")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--refs", default="30s", help="reference set label (10s / 30s)")
    ap.add_argument("--voices", nargs="+", default=["jsut_f", "cv_m"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lines", nargs="*", default=None, help="line ids (default: all)")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--num-steps", type=int, default=None, help="default: 40 RF / 4 MeanFlow")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download

    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav

    repo, rev = MODELS[args.model]
    snap = Path(snapshot_download(repo, revision=rev, allow_patterns=["model.safetensors", "tokenizer/*", "*.md"]))
    snapshot_download(CODEC[0], revision=CODEC[1])

    name = f"irodori-{args.model}" + (f"-{args.precision}" if args.precision != "fp32" else "") + args.tag
    out_root = EVAL_DIR / "takes" / name
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"log_{args.refs}.jsonl"

    script = load_script()["lines"]
    if args.lines:
        script = [l for l in script if l["id"] in set(args.lines)]

    t0 = time.perf_counter()
    with GpuMemSampler() as load_mem:
        runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=str(snap / "model.safetensors"), model_device="cuda",
            codec_repo=CODEC[0], model_precision=args.precision,
            codec_device="cuda", codec_precision="fp32",
        ))
    load_sec = time.perf_counter() - t0

    def request(text: str, caption: str | None, clips: list[str], seed: int) -> SamplingRequest:
        return SamplingRequest(text=text, caption=caption, ref_wavs=clips, seed=seed,
                               num_steps=args.num_steps)

    log = log_path.open("a", encoding="utf-8")
    first = True
    for voice in args.voices:
        meta = json.loads((REF_DIR / f"{voice}_{args.refs}" / "meta.json").read_text(encoding="utf-8"))
        out_dir = out_root / f"{voice}_{args.refs}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for line in script:
            for seed in args.seeds:
                rec = {"model": name, "repo": repo, "revision": rev, "voice": voice, "refs": args.refs,
                       "line": line["id"], "category": line["category"], "seed": seed,
                       "prompt": line["irodori"], "caption": line.get("irodori_caption"),
                       "precision": args.precision, "num_steps": args.num_steps,
                       "cold": first, "load_sec": round(load_sec, 2) if first else None,
                       "load_gpu_mib": load_mem.peak_mib if first else None}
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                try:
                    with GpuMemSampler() as mem:
                        res = runtime.synthesize(request(line["irodori"], line.get("irodori_caption"),
                                                         meta["clips"], seed))
                    wall = time.perf_counter() - t
                    wav_path = out_dir / f"{line['id']}_s{seed}.wav"
                    save_wav(wav_path, res.audio, res.sample_rate)
                    audio_sec = res.audio.shape[-1] / res.sample_rate
                    rec.update({
                        "ok": True, "wav": str(wav_path), "sample_rate": res.sample_rate,
                        "audio_sec": round(audio_sec, 3), "wall_sec": round(wall, 3),
                        "rtf": round(wall / audio_sec, 4), "xrt": round(audio_sec / wall, 3),
                        "stages": {k: round(v, 4) for k, v in res.stage_timings},
                        "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                        "gpu_used_peak_mib": mem.peak_mib, "gpu_util_mean": mem.util_mean, "used_seed": res.used_seed,
                        "messages": res.messages[-3:],
                    })
                except Exception as exc:  # record failures, keep going
                    rec.update({"ok": False, "error": repr(exc), "wall_sec": round(time.perf_counter() - t, 3)})
                rec["rss_peak_mib"] = round(peak_rss_mib(os.getpid()))
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                print(f"{name} {voice}_{args.refs} {line['id']} s{seed} "
                      f"{'ok' if rec['ok'] else 'FAIL'} audio={rec.get('audio_sec')} wall={rec.get('wall_sec')}",
                      flush=True)
                first = False
    log.close()


if __name__ == "__main__":
    main()
