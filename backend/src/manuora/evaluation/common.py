"""Plumbing shared by all experiments: file I/O, local LLM calls, GPU sampling.

Keep this file boring: no experiment logic belongs here. Gold scoring lives in
`manuora.evaluation.coverage`, stage-wise reporting in `manuora.evaluation.stages`.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
import unicodedata
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
EVAL = ROOT / "data/evaluation"


@lru_cache(maxsize=1)
def settings() -> dict:
    """config/assistant.yaml, the same file the app reads."""
    return yaml.safe_load((ROOT / "config/assistant.yaml").read_text(encoding="utf-8"))


# --- files -----------------------------------------------------------------
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(p):
    """Read a JSONL file."""
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def append(path, row):
    """Append one JSONL row (used for resumable answer caches)."""
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(x):
    """Casefolded, punctuation-free text for literal quote matching."""
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", x).casefold())


# --- local generator (Docker Model Runner, OpenAI-compatible) ---------------
def generation_config():
    return settings()["generation"]


def call(messages, max_tokens=360):
    """One deterministic chat completion against the configured local model."""
    import urllib.request

    cfg = generation_config()
    tick = time.perf_counter()
    payload = dict(model=cfg["model"], messages=messages, temperature=0, max_tokens=max_tokens, stream=False)
    req = urllib.request.Request(cfg["url"], json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.load(response)
    return dict(
        text=result["choices"][0]["message"]["content"],
        finish_reason=result["choices"][0].get("finish_reason"),
        usage=result.get("usage"),
        seconds=time.perf_counter() - tick,
    )


def prompt_tokens(tok, messages):
    """Qwen3 ChatML prompt length (thinking disabled) for context-budget asserts."""
    serial = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages)
    serial += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return len(tok.encode(serial, add_special_tokens=False))


@contextmanager
def model_context(model, size=4096, after=None):
    """Temporarily run the local model with a larger context, then always restore it.

    Refuses to run while the live app has the model loaded. Yields the runtime
    configuration from before the change; writes the restored one to `after`.
    """
    from manuora.generation.model_control import _docker, is_loaded, unload_model

    assert not is_loaded(model), "Live model is loaded; do not interrupt it."
    before = json.loads(_docker("configure", "show") or "[]")
    entry = next((r for r in before if r["Model"] == model), None)
    try:
        _docker("configure", "--context-size", str(size), "--think=false", model)
        yield before
    finally:
        unload_model(model)
        # Restore an explicit earlier context. With no earlier entry there is nothing to
        # restore to, so the run's setting (thinking off) stays and is recorded in `after`.
        if entry is not None:
            _docker("configure", "--context-size", str(entry["Config"]["context-size"]), "--think=false", model)
        if after is not None:
            write(after, dict(before=before, after=json.loads(_docker("configure", "show") or "[]")))


# --- measurement -----------------------------------------------------------
class GpuSampler:
    """Observe device-wide memory/utilization; values are not process attribution."""

    def __init__(self):
        self.samples: list[tuple[int, int]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        used, utilization = [int(value.strip()) for value in line.split(",")[:2]]
                        self.samples.append((used, utilization))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(2)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> dict:
        self.stop.set()
        self.thread.join(timeout=6)
        return {"samples": len(self.samples),
                "peak_device_memory_mib": max((row[0] for row in self.samples), default=None),
                "peak_device_utilization_percent": max((row[1] for row in self.samples), default=None)}
