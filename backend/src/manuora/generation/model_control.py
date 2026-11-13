"""Local Docker Model Runner controls for the chat page."""

from __future__ import annotations

import json
import subprocess
import urllib.request


def _docker(*args: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(["docker", "model", *args], capture_output=True,
                                text=True, timeout=timeout,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Docker Model Runner command failed: {exc}") from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Docker Model Runner is unavailable")
    return result.stdout


def is_loaded(model: str) -> bool:
    running = _docker("ps", timeout=10)
    return any(line.split()[0].casefold() == model.casefold()
               for line in running.splitlines()[1:] if line.split())


def load_model(model: str, url: str) -> None:
    # A one-token local request loads the already-downloaded model into VRAM.
    payload = {"model": model, "messages": [{"role": "user", "content": "Ready."}],
               "max_tokens": 1, "stream": False}
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()


def unload_model(model: str) -> None:
    if is_loaded(model):
        _docker("unload", model)
