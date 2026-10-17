"""Long-form (~1 min) voice-drift test.

drama3-small: one request; audio.cpp chunks the text itself (text_chunk_size=200, tag_aware).
Irodori: one request per sentence (its output cap is 30 s), concatenated with 0.25 s gaps.
Drift = WavLM similarity of each ~8 s window to the 30 s reference, first vs last window.

  python longform.py --engine irodori-small   (Irodori venv)
  python longform.py --engine drama3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from common import EVAL_DIR, REF_DIR

PARAGRAPH = (
    "あの夏のことは、今でもはっきり覚えている。"
    "駅のホームで待っていた君は、白いワンピースを着て、少し緊張した顔をしていた。"
    "僕たちは海沿いの古い町を歩いて、名前も知らない小さな神社にお参りをした。"
    "夕方になると、空が少しずつ橙色に染まって、波の音だけが静かに響いていた。"
    "君は突然立ち止まって、来年もまたここに来ようねと笑った。"
    "その約束は、結局果たせなかったけれど、僕は毎年この季節になると、一人であの浜辺を訪れている。"
    "潮風の匂いをかぐたびに、あの日の君の声が、すぐそばで聞こえる気がするんだ。"
)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[。！？])", text) if s]


def run_irodori(model: str, voices: list[str], seeds: list[int], out_root: Path) -> list[dict]:
    sys.path.insert(0, os.environ.get("IRODORI_SRC", "/storage/tts-runtime/Irodori-TTS"))
    from huggingface_hub import snapshot_download

    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    from run_irodori import CODEC, MODELS

    repo, rev = MODELS[model]
    snap = Path(snapshot_download(repo, revision=rev, allow_patterns=["model.safetensors", "tokenizer/*"]))
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(snap / "model.safetensors"), model_device="cuda",
                                              codec_repo=CODEC[0], codec_device="cuda"))
    recs = []
    for voice in voices:
        meta = json.loads((REF_DIR / f"{voice}_30s" / "meta.json").read_text(encoding="utf-8"))
        for seed in seeds:
            t = time.perf_counter()
            parts, sr = [], 48000
            for i, s in enumerate(sentences(PARAGRAPH)):
                res = rt.synthesize(SamplingRequest(text=s, ref_wavs=meta["clips"], seed=seed * 100 + i))
                sr = res.sample_rate
                parts += [res.audio[0].float().cpu().numpy(), np.zeros(int(0.25 * sr), dtype=np.float32)]
            wav = np.concatenate(parts[:-1])
            path = out_root / f"irodori-{model}_{voice}_s{seed}.wav"
            sf.write(path, wav, sr)
            recs.append({"engine": f"irodori-{model}", "voice": voice, "seed": seed, "wav": str(path),
                         "audio_sec": round(len(wav) / sr, 2), "wall_sec": round(time.perf_counter() - t, 2),
                         "requests": len(sentences(PARAGRAPH))})
            print(recs[-1], flush=True)
    return recs


def run_drama3(voices: list[str], seeds: list[int], out_root: Path) -> list[dict]:
    import run_drama3 as rd

    cfg = out_root / "server.json"
    rd.write_config(cfg, "offline", None)
    cfg_data = json.loads(cfg.read_text())
    cfg_data["models"][0]["default_request_options"] = {"max_tokens": 1024}
    cfg.write_text(json.dumps(cfg_data, indent=2))
    env = dict(os.environ, AUDIOCPP_DRAMA3_MAX_PAUSE_FRAMES="10", AUDIOCPP_DRAMA3_SILENCE_CODES=str(rd.SILENCE))
    proc = subprocess.Popen([str(rd.SERVER_BIN), "--config", str(cfg)], cwd=str(rd.AUDIOCPP),
                            stdout=(out_root / "server.log").open("w"), stderr=subprocess.STDOUT, env=env)
    recs = []
    try:
        rd.wait_ready(proc)
        for voice in voices:
            meta = json.loads((REF_DIR / f"{voice}_30s" / "meta.json").read_text(encoding="utf-8"))
            for seed in seeds:
                body = {"model": "drama3-small", "input": PARAGRAPH, "seed": seed, "voice_ref": meta["concat"],
                        "reference_text": meta["reference_text"], **rd.SAMPLING,
                        "options": dict(rd.PAUSE_FIX)}
                t = time.perf_counter()
                data = rd.speak(body, timeout=1200)
                path = out_root / f"drama3-small_{voice}_s{seed}.wav"
                path.write_bytes(data)
                sec, _ = rd.wav_seconds(data)
                recs.append({"engine": "drama3-small", "voice": voice, "seed": seed, "wav": str(path),
                             "audio_sec": round(sec, 2), "wall_sec": round(time.perf_counter() - t, 2), "requests": 1})
                print(recs[-1], flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    return recs


def score(out_root: Path) -> None:
    from common import SpeakerEmbedder, Whisper, cer, cosine, load_audio, silence_stats, to_kana

    asr, sv = Whisper(), SpeakerEmbedder()
    refs = {}
    rows = []
    for line in (out_root / "log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["voice"] not in refs:
            meta = json.loads((REF_DIR / f"{r['voice']}_30s" / "meta.json").read_text(encoding="utf-8"))
            refs[r["voice"]] = sv.embed(load_audio(meta["concat"]))
        wav = load_audio(r["wav"])
        win = 8 * 16000
        sims = [cosine(sv.embed(wav[i:i + win]), refs[r["voice"]])
                for i in range(0, max(len(wav) - win // 2, 1), win)]
        hyp = asr.transcribe(wav)
        r.update({"kana_cer": round(cer(to_kana(PARAGRAPH), to_kana(hyp)), 4),
                  "sim_first": round(sims[0], 4), "sim_last": round(sims[-1], 4),
                  "sim_min": round(min(sims), 4), "sim_windows": [round(s, 3) for s in sims],
                  "longest_pause_s": round(silence_stats(wav, 16000)["longest_pause_s"], 2), "asr": hyp})
        r.pop("text", None)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    (out_root / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="drama3 | irodori-small | irodori-mf | irodori-anime | score")
    ap.add_argument("--voices", nargs="+", default=["jsut_f", "cv_m"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    out_root = EVAL_DIR / "longform"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.engine == "score":
        score(out_root)
        sys.exit(0)
    if args.engine == "drama3":
        recs = run_drama3(args.voices, args.seeds, out_root)
    else:
        recs = run_irodori(args.engine.split("-", 1)[1], args.voices, args.seeds, out_root)
    with (out_root / "log.jsonl").open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({**r, "text": PARAGRAPH}, ensure_ascii=False) + "\n")
