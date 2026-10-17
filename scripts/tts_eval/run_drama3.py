"""Generate the comparison takes with drama3-small through audiocpp_server (/v1/audio/speech).

  python scripts/tts_eval/run_drama3.py --refs 30s
The server loads the GGUF once at startup (load time logged); every request re-encodes the
reference clip, as in the model card's speed table. Pause fix and sampling follow the README.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import EVAL_DIR, MODEL_ROOT, REF_DIR, GpuMemSampler, load_script, peak_rss_mib

AUDIOCPP = Path(os.environ.get("AUDIOCPP_DIR", "/storage/tts-runtime/audio.cpp"))
SERVER_BIN = AUDIOCPP / "build/linux-cuda-release/bin/audiocpp_server"
GGUF_DIR = MODEL_ROOT / "drama3-small-GGUF"
GGUF = GGUF_DIR / "drama3-small-IQ3_XXS.gguf"
SILENCE = GGUF_DIR / "silence_codes.json"
HF_REV = "4a3078e885344bb9b24d511cec389d2726854d15"
AUDIOCPP_COMMIT = "9277495adfdc9b41004bd91db7b7a4e5d456c8df"
PORT = int(os.environ.get("DRAMA3_PORT", "8035"))

SAMPLING = {"temperature": 0.7, "top_p": 0.7, "top_k": 30}
PAUSE_FIX = {"max_pause_frames": "10", "silence_codes": str(SILENCE)}


def write_config(path: Path, mode: str, max_pause: str | None) -> None:
    cfg = {
        "host": "127.0.0.1", "port": PORT, "backend": "cuda", "device": 0, "threads": 8,
        "lazy_load": False, "max_loaded_models": 1,
        "models": [{
            "id": "drama3-small", "family": "fish_drama3", "task": "clon", "mode": mode,
            "path": str(GGUF),
            "session_options": {"fish_drama3.cpu_expert_layers": 0, "fish_drama3.context_length": 4096},
            "default_request_options": {"max_tokens": 645},
        }],
    }
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def wait_ready(proc: subprocess.Popen, timeout: float = 900) -> None:
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("server not ready")


def speak(body: dict, timeout: float = 600) -> bytes:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/audio/speech", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def speak_stream(body: dict, timeout: float = 600) -> tuple[bytes, float, int]:
    """SSE streaming; returns (pcm16 bytes, seconds to first audio delta, sample rate)."""
    body = {**body, "response_format": "pcm", "stream_format": "sse"}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/audio/speech", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    ttfa, pcm, sr = None, bytearray(), 44100
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("type") == "speech.audio.delta":
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                pcm += base64.b64decode(ev["audio"])
                sr = int(ev.get("sample_rate", sr))
    return bytes(pcm), ttfa if ttfa is not None else float("nan"), sr


def wav_seconds(data: bytes) -> tuple[float, int]:
    import io

    import soundfile as sf

    info = sf.info(io.BytesIO(data))
    return info.frames / info.samplerate, info.samplerate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="30s")
    ap.add_argument("--voices", nargs="+", default=["jsut_f", "cv_m"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lines", nargs="*", default=None)
    ap.add_argument("--mode", choices=["offline", "streaming"], default="offline")
    ap.add_argument("--no-pause-fix", action="store_true")
    ap.add_argument("--no-ref-text", action="store_true", help="omit reference_text")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    name = "drama3-small" + ("-stream" if args.mode == "streaming" else "") + \
        ("-nofix" if args.no_pause_fix else "") + ("-noreftext" if args.no_ref_text else "") + args.tag
    out_root = EVAL_DIR / "takes" / name
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_path = out_root / "server.json"
    write_config(cfg_path, args.mode, None)

    env = dict(os.environ)
    if not args.no_pause_fix:
        env.update({"AUDIOCPP_DRAMA3_MAX_PAUSE_FRAMES": "10", "AUDIOCPP_DRAMA3_SILENCE_CODES": str(SILENCE)})
    else:
        env["AUDIOCPP_DRAMA3_MAX_PAUSE_FRAMES"] = "0"
    server_log = (out_root / f"server_{args.refs}.log").open("w")
    t0 = time.perf_counter()
    with GpuMemSampler() as load_mem:
        proc = subprocess.Popen([str(SERVER_BIN), "--config", str(cfg_path)], cwd=str(AUDIOCPP),
                                stdout=server_log, stderr=subprocess.STDOUT, env=env)
        wait_ready(proc)
    load_sec = time.perf_counter() - t0
    print(f"server ready in {load_sec:.1f}s, gpu {load_mem.peak_mib} MiB", flush=True)

    script = load_script()["lines"]
    if args.lines:
        script = [l for l in script if l["id"] in set(args.lines)]
    log = (out_root / f"log_{args.refs}.jsonl").open("a", encoding="utf-8")
    first = True
    try:
        for voice in args.voices:
            meta = json.loads((REF_DIR / f"{voice}_{args.refs}" / "meta.json").read_text(encoding="utf-8"))
            out_dir = out_root / f"{voice}_{args.refs}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for line in script:
                for seed in args.seeds:
                    options = {} if args.no_pause_fix else dict(PAUSE_FIX)
                    if args.no_pause_fix:
                        options["max_pause_frames"] = "0"
                    body = {"model": "drama3-small", "input": line["drama3"], "seed": seed,
                            "voice_ref": meta["concat"], **SAMPLING, "options": options}
                    if not args.no_ref_text:
                        body["reference_text"] = meta["reference_text"]
                    rec = {"model": name, "repo": "pentacoxian-dev/drama3-small-GGUF", "revision": HF_REV,
                           "audiocpp_commit": AUDIOCPP_COMMIT, "voice": voice, "refs": args.refs,
                           "line": line["id"], "category": line["category"], "seed": seed,
                           "prompt": line["drama3"], "mode": args.mode, "sampling": SAMPLING,
                           "options": options, "reference_text": not args.no_ref_text,
                           "cold": first, "load_sec": round(load_sec, 2) if first else None,
                           "load_gpu_mib": load_mem.peak_mib if first else None}
                    t = time.perf_counter()
                    try:
                        with GpuMemSampler() as mem:
                            if args.mode == "streaming":
                                pcm, ttfa, sr = speak_stream(body)
                            else:
                                data = speak(body)
                        wall = time.perf_counter() - t
                        wav_path = out_dir / f"{line['id']}_s{seed}.wav"
                        if args.mode == "streaming":
                            import numpy as np
                            import soundfile as sf

                            sf.write(wav_path, np.frombuffer(pcm, dtype="<i2"), sr, subtype="PCM_16")
                            audio_sec = len(pcm) / 2 / sr
                            rec["ttfa_sec"] = round(ttfa, 3)
                        else:
                            wav_path.write_bytes(data)
                            audio_sec, sr = wav_seconds(data)
                        rec.update({"ok": True, "wav": str(wav_path), "sample_rate": sr,
                                    "audio_sec": round(audio_sec, 3), "wall_sec": round(wall, 3),
                                    "rtf": round(wall / audio_sec, 4) if audio_sec else None,
                                    "xrt": round(audio_sec / wall, 3), "gpu_used_peak_mib": mem.peak_mib, "gpu_util_mean": mem.util_mean})
                    except Exception as exc:
                        detail = exc.read().decode(errors="replace")[:500] if isinstance(exc, urllib.error.HTTPError) else ""
                        rec.update({"ok": False, "error": repr(exc) + " " + detail,
                                    "wall_sec": round(time.perf_counter() - t, 3)})
                        if proc.poll() is not None:
                            raise
                    rec["server_rss_peak_mib"] = round(peak_rss_mib(proc.pid))
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log.flush()
                    print(f"{name} {voice}_{args.refs} {line['id']} s{seed} "
                          f"{'ok' if rec['ok'] else 'FAIL'} audio={rec.get('audio_sec')} wall={rec.get('wall_sec')}",
                          flush=True)
                    first = False
    finally:
        log.close()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        server_log.close()


if __name__ == "__main__":
    main()
