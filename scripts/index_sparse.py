"""Build the BM25 SQLite index from retrieval chunks."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from manuora.retrieval.sparse import build_sparse_index  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(build_sparse_index(ROOT, ROOT / "data" / "sources"), indent=2))
