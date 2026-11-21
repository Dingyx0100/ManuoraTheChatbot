"""Start the router model in llama-server, for conversation.context_router_method: llm.

    python scripts/router/serve_router.py            # starts in the background, waits until healthy
    python scripts/router/serve_router.py --stop

The app works without it: when the server is unreachable, the router falls back to the TF-IDF classifier
(`fallback_classifier` in config/assistant.yaml). llama-server comes with Docker Desktop's model runner; pass
--server if it lives elsewhere. Small context (-c 2048) so it fits on the GPU beside the 8B chat model.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data/models/router_qwen3-0.6b-mix_q8_0.gguf"
SERVER = Path(os.path.expanduser("~")) / ".docker/bin/inference/llama-server.exe"
PID_FILE = ROOT / "data/models/router_server.pid"


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as response:
            return b"ok" in response.read()
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--server", default=str(SERVER))
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if args.stop:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text())
            subprocess.run(["taskkill", "/PID", str(pid), "/F"] if os.name == "nt" else ["kill", str(pid)], check=False)
            PID_FILE.unlink()
            print(f"stopped router server (pid {pid})")
        else:
            print("no router server started by this script")
        return
    if healthy(url):
        print(f"router server already running at {url}")
        return
    log = ROOT / "data/models/llama_server_router.log"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([args.server, "-m", args.model, "-c", "2048", "-np", "1", "-ngl", "999",
                                "--host", "127.0.0.1", "--port", str(args.port), "--log-file", str(log)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    PID_FILE.write_text(str(process.pid))
    for _ in range(60):
        if healthy(url):
            print(f"router server ready at {url} (pid {process.pid})")
            return
        if process.poll() is not None:
            sys.exit(f"llama-server exited with code {process.returncode}; see {log}")
        time.sleep(1)
    sys.exit(f"router server not healthy after 60 s; see {log}")


if __name__ == "__main__":
    main()
