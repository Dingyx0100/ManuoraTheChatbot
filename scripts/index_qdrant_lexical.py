"""Experiment L: build the Qdrant BM25 copy of the production dense collection.

    python scripts/index_qdrant_lexical.py            # uses embedding_config + dense_collection from config/assistant.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from manuora.retrieval.indexing import IndexConfig  # noqa: E402
from manuora.retrieval.qdrant_lexical import build_lexical_collection  # noqa: E402


if __name__ == "__main__":
    settings = yaml.safe_load((ROOT / "config" / "assistant.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=settings["embedding_config"])
    parser.add_argument("--collection", default=settings["dense_collection"], help="source dense collection")
    args = parser.parse_args()
    config = IndexConfig.load(ROOT / args.config, ROOT)
    print(json.dumps(build_lexical_collection(config, args.collection), indent=2))
