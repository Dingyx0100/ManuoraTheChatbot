"""Local-only products (config/local.yaml `products:`) must never appear in files tracked by git.

The names come from the local file, so this test never has to spell them out. It skips on a fresh
clone or in CI, where config/local.yaml does not exist.
"""
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "config/local.yaml"


def test_no_local_only_product_in_tracked_files():
    if not LOCAL.is_file():
        pytest.skip("no config/local.yaml on this machine")
    products = (yaml.safe_load(LOCAL.read_text(encoding="utf-8")) or {}).get("products") or {}
    names = {str(n).casefold() for pair in products.items() for n in pair if n}
    if not names:
        pytest.skip("config/local.yaml lists no local-only products")
    try:
        tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    hits = []
    for rel in filter(None, tracked.split("\0")):
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", "ignore").casefold()
        hits += [f"{rel}: {name}" for name in names if name in text]
    assert not hits, "local-only product named in tracked files:\n" + "\n".join(hits)
