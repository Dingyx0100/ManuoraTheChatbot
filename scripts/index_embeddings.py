"""Build or query a model-isolated Qdrant dense index over retrieval chunks."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from manuora.retrieval.indexing import IndexConfig, build_index, search_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "embeddings.yaml")
    command = parser.add_subparsers(dest="command", required=True)
    build = command.add_parser("build", help="Split as needed, embed, and upsert chunks")
    build.add_argument("--source-root", type=Path, default=ROOT / "data" / "sources")
    build.add_argument("--select", help="Folder under source root, e.g. QGIS/3.44")
    build.add_argument("--max-chunks", type=int, help="Small reproducible pilot subset")
    build.add_argument("--device", help="Runtime device override, e.g. cuda for indexing; not part of the vector space")
    search = command.add_parser("search", help="Dense retrieval from one managed collection")
    search.add_argument("--collection", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--product-id")
    search.add_argument("--product-version")
    search.add_argument("--document-id")
    listing = command.add_parser("list", help="Show local index manifests")
    _ = listing
    args = parser.parse_args()
    config = IndexConfig.load(args.config, ROOT)
    if args.command == "build":
        encoder = None
        if args.device:
            from manuora.retrieval.indexing import SentenceEncoder
            encoder = SentenceEncoder({**config.model, "device": args.device})
        result = build_index(config, args.source_root.resolve(),
                             select=args.select, max_chunks=args.max_chunks,
                             encoder=encoder)
    elif args.command == "search":
        if args.limit < 1:
            parser.error("--limit must be positive")
        result = search_index(config, args.collection, args.query, limit=args.limit,
                              product_id=args.product_id,
                              product_version=args.product_version,
                              document_id=args.document_id)
    else:
        result = [json.loads(path.read_text(encoding="utf-8"))
                  for path in sorted(config.location("registry_path").glob("*.json"))]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
