"""Shared helpers for the drama3-small / Irodori-TTS comparison (issue #13).

Models are read from HF_HOME under /app/models/tts (per-session, never /storage).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np

MODEL_ROOT = Path(os.environ.get("TTS_MODEL_ROOT", "/app/models/tts"))
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf"))
REPO_DIR = Path(__file__).resolve().parent
SCRIPT_JSON = Path(os.environ.get("TTS_SCRIPT_JSON", str(REPO_DIR / "script_ja.json")))
EVAL_DIR = Path(os.environ.get("TTS_EVAL_DIR", "/notebooks/workspace/tts_eval"))
REF_DIR = Path(os.environ.get("TTS_REF_DIR", str(EVAL_DIR / "refs")))

WHISPER_REPO = "openai/whisper-large-v3-turbo"
WHISPER_REV = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
WAVLM_SV_REPO = "microsoft/wavlm-base-plus-sv"
WAVLM_SV_REV = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"


def load_script() -> dict:
    return json.loads(SCRIPT_JSON.read_text(encoding="utf-8"))


def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if file_sr != sr:
        mono = F.resample(torch.from_numpy(mono), file_sr, sr).numpy()
    return mono


class GpuMemSampler:
    """Polls nvidia-smi used memory for the whole GPU (works for any process, incl. audio.cpp)."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_mib = 0
        self.util_samples: list[int] = []
        self.baseline_mib = self._read()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read() -> int:
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetMemoryInfo(h).used // (1024 * 1024))
        except Exception:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()
            return int(out[0]) if out else 0

    @staticmethod
    def _util() -> int | None:
        try:
            import pynvml

            return int(pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(0)).gpu)
        except Exception:
            return None

    @property
    def util_mean(self) -> float | None:
        return round(sum(self.util_samples) / len(self.util_samples), 1) if self.util_samples else None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._read())
            u = self._util()
            if u is not None:
                self.util_samples.append(u)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, self._read())


def peak_rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


# ---------- text normalisation for CER ----------

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⌀-⏿⬀-⯿️‍]"
)
_TAG_RE = re.compile(r"\[[^\]]*\]|<[^>]*>")
_PUNCT_RE = re.compile(r"[\s、。，．,.!！?？…・「」『』（）()\-ー〜~：:;；\"'“”‘’%％]")


def strip_controls(text: str) -> str:
    return _EMOJI_RE.sub("", _TAG_RE.sub("", text))


def _kakasi():
    import pykakasi

    if not hasattr(_kakasi, "k"):
        _kakasi.k = pykakasi.kakasi()
    return _kakasi.k


def to_kana(text: str) -> str:
    text = unicodedata.normalize("NFKC", strip_controls(text))
    hira = "".join(item["hira"] for item in _kakasi().convert(text))
    return _PUNCT_RE.sub("", hira)


def to_surface(text: str) -> str:
    text = unicodedata.normalize("NFKC", strip_controls(text))
    return _PUNCT_RE.sub("", text)


def cer(ref: str, hyp: str) -> float:
    import jiwer

    if not ref:
        return float("nan")
    return float(jiwer.cer(ref, hyp if hyp else " "))


# ---------- ASR / speaker models ----------

class Whisper:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(WHISPER_REPO, revision=WHISPER_REV)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_REPO, revision=WHISPER_REV, torch_dtype=torch.float16
        ).to(device).eval()

    def transcribe(self, wav16k: np.ndarray) -> str:
        import torch

        feats = self.processor(
            wav16k, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        if feats.input_features.shape[-1] < 3000:
            feats = self.processor(
                wav16k, sampling_rate=16000, return_tensors="pt",
                padding="max_length", return_attention_mask=True,
            )
        long_form = len(wav16k) > 30 * 16000
        if long_form:
            feats = self.processor(
                wav16k, sampling_rate=16000, return_tensors="pt", truncation=False,
                padding="longest", return_attention_mask=True,
            )
        with torch.inference_mode():
            # temperature fallback + compression-ratio check (OpenAI defaults) stop Whisper's
            # repetition loops ("あ、あ、あ…") from being scored as TTS content errors
            ids = self.model.generate(
                feats.input_features.to(self.device, torch.float16),
                attention_mask=feats.attention_mask.to(self.device),
                language="ja", task="transcribe",
                return_timestamps=long_form,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                compression_ratio_threshold=1.8 if not long_form else 2.4,
                logprob_threshold=-1.0, do_sample=False,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class SpeakerEmbedder:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(WAVLM_SV_REPO, revision=WAVLM_SV_REV)
        self.model = WavLMForXVector.from_pretrained(WAVLM_SV_REPO, revision=WAVLM_SV_REV).to(device).eval()

    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        import torch

        inputs = self.fe(wav16k, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            emb = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).embeddings
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb[0].cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------- simple signal checks ----------

def silence_stats(wav: np.ndarray, sr: int, frame_ms: float = 20.0, thresh_db: float = -45.0) -> dict:
    """Longest internal silence (s) and leading/trailing silence, relative to the clip's peak."""
    hop = int(sr * frame_ms / 1000)
    if len(wav) < hop:
        return {"longest_pause_s": 0.0, "lead_s": 0.0, "tail_s": 0.0}
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    voiced = db > thresh_db
    idx = np.flatnonzero(voiced)
    if idx.size == 0:
        dur = len(wav) / sr
        return {"longest_pause_s": dur, "lead_s": dur, "tail_s": dur}
    lead, tail = idx[0], len(voiced) - 1 - idx[-1]
    inner = voiced[idx[0] : idx[-1] + 1]
    longest = run = 0
    for v in inner:
        run = 0 if v else run + 1
        longest = max(longest, run)
    f = frame_ms / 1000
    # share of frames above an absolute -40 dBFS: catches takes that are a quiet murmur throughout
    abs_db = 20 * np.log10(rms + 1e-12)
    return {"longest_pause_s": longest * f, "lead_s": lead * f, "tail_s": tail * f,
            "audible_ratio": float((abs_db > -40).mean()), "peak_dbfs": float(abs_db.max())}
