"""Seed-VC smoke/eval through the ComfyUI API (issue #14).

Every voice here is synthetic: drama3-small invents a voice from a seed and an English description
when it gets no reference audio, so no real person's voice is cloned or converted.

  python run_seedvc.py prepare   # target voices + source clips (speech and non-verbal sounds) via drama3-small
  python run_seedvc.py convert   # Seed-VC over every source x target x condition
"""
from __future__ import annotations

import itertools
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:6006"
COMFY = Path("/storage/ComfyUI")
OUT = Path("/notebooks/workspace/vc_eval")
WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/seedvc_convert_api.json"

# Target speakers (fictional, generated without reference audio).
TARGETS = {
    "tgt_f": dict(seed=11, desc="a young, bright, slightly high-pitched female anime voice",
                  text="はじめまして。今日からこのお店で働くことになりました。わからないことばかりですが、どうぞよろしくお願いします。"),
    "tgt_m": dict(seed=23, desc="a calm, low, mature male voice",
                  text="はじめまして。今日からこの研究室に配属になりました。わからないことも多いですが、よろしくお願いします。"),
}
# Source clips in a third synthetic voice: speech plus non-verbal sounds (breath, sighs, laughs).
SRC_VOICE = dict(seed=37, desc="a soft, gentle female voice")
SOURCES = {
    "speech_daily": "おはよう。今日はちょっと早く起きちゃったから、一緒に朝ごはん食べない？",
    "speech_emotion": "[excited, overjoyed] やったー！ついに合格したよ！ずっと頑張ってきて本当によかった！",
    "nv_sigh": "[sighs deeply] はぁ……。[sighs] もう、今日は本当に疲れちゃった。",
    "nv_laugh": "[laughs] あはは、ちょっと待って、[laughing] それ面白すぎるって！",
    "nv_breath": "[breathing heavily, out of breath after running] はぁ、はぁ……ま、間に合った……はぁ……。",
    "nv_gasp_whisper": "[gasps] えっ……<whisper>ちょっと、静かにして。誰か来るかも。</whisper>",
}
CONDITIONS = [  # (label, steps, f0_condition, auto_f0_adjust)
    ("f0_s25", 25, True, True),
    ("f0_s50", 50, True, True),
    ("speech_s25", 25, False, True),
]
TAKES = [0, 1]  # the node has no seed input, so takes are repeated runs (on per-take input copies,
               # otherwise ComfyUI returns the cached first take)


def post(prompt: dict) -> tuple[float, list[dict]]:
    req = urllib.request.Request(f"{API}/prompt", json.dumps({"prompt": prompt}).encode(),
                                 {"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    t = time.time()
    while True:
        h = json.load(urllib.request.urlopen(f"{API}/history/{pid}")).get(pid)
        if h and (h["status"].get("completed") or h["status"].get("status_str") == "error"):
            break
        time.sleep(0.5)
    if h["status"]["status_str"] != "success":
        raise RuntimeError(json.dumps(h["status"]["messages"])[-2000:])
    return time.time() - t, [f for o in h["outputs"].values() for f in o.get("audio", [])]


def output_path(f: dict) -> Path:
    return COMFY / "output" / f["subfolder"] / f["filename"]


def drama3_prompt(text: str, seed: int, desc: str, prefix: str) -> dict:
    return {
        "1": {"class_type": "Drama3SmallLoader", "inputs": {"gguf": "drama3-small-IQ3_XXS.gguf", "device": "auto",
                                                            "precision": "auto", "expert_offload": "none"}},
        "2": {"class_type": "Drama3SmallTTS", "inputs": {
            "model": ["1", 0], "text": text, "voice_description": desc, "seed": seed, "temperature": 0.7,
            "top_p": 0.7, "top_k": 30, "max_seconds": 45.0, "max_pause_seconds": 0.45, "keep_model_loaded": True}},
        "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": prefix}},
    }


def prepare() -> None:
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    jobs = [(name, t["text"], t["seed"], t["desc"]) for name, t in TARGETS.items()]
    jobs += [(name, text, SRC_VOICE["seed"], SRC_VOICE["desc"]) for name, text in SOURCES.items()]
    for name, text, seed, desc in jobs:
        wall, files = post(drama3_prompt(text, seed, desc, f"vc_eval/inputs/{name}"))
        src = output_path(files[0])
        shutil.copy(src, OUT / "inputs" / f"{name}.flac")
        shutil.copy(src, COMFY / "input" / f"vc_eval_{name}.flac")  # LoadAudio lists the top of input/
        print(name, round(wall, 1), flush=True)


def convert() -> None:
    graph = json.loads(WORKFLOW.read_text())
    log = (OUT / "log.jsonl").open("a")
    for src, tgt, (label, steps, f0, auto_f0), take in itertools.product(SOURCES, TARGETS, CONDITIONS, TAKES):
        g = json.loads(json.dumps(graph))
        for role, name in (("source", src), ("reference", tgt)):
            copy = COMFY / "input" / f"vc_eval_{name}_t{take}.flac"
            if not copy.exists():
                shutil.copy(COMFY / "input" / f"vc_eval_{name}.flac", copy)
            g[role]["inputs"]["audio"] = copy.name
        g["convert"]["inputs"].update(steps=steps, f0_condition=f0, auto_f0_adjust=auto_f0, unload_model=False)
        g["save"]["inputs"]["filename_prefix"] = f"vc_eval/seedvc/{src}__{tgt}__{label}__t{take}"
        wall, files = post(g)
        rec = dict(engine="seed-vc", source=src, target=tgt, condition=label, steps=steps, f0_condition=f0,
                   auto_f0_adjust=auto_f0, take=take, wall_sec=round(wall, 2), wav=str(output_path(files[0])))
        print(rec, flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()


if __name__ == "__main__":
    {"prepare": prepare, "convert": convert}[sys.argv[1]]()
