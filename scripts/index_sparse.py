"""Build the BM25 SQLite index from retrieval chunks."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from manuora.retrieval.sparse import build_sparse_index  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="unicode61",
                        help='FTS5 tokenize spec; "porter unicode61" adds English stemming (own index file)')
    args = parser.parse_args()
    print(json.dumps(build_sparse_index(ROOT, ROOT / "data" / "sources", tokenizer=args.tokenizer), indent=2))
