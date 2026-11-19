"""Discover source PDFs and run selected manifest, parse, and chunk stages."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import yaml

from manuora.ingestion.naming import document_id, inferred_type, inferred_version, slug  # noqa: F401
from manuora.ingestion.files import (
    document_metadata_sha256, resolve_output_dir, resolve_source_path, sha256_file,
)
from manuora.ingestion.visual_regions import CLASSIFIER_VERSION
from manuora.knowledge.manifests import load_manifest
from manuora.knowledge.schemas import ProductManifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "data" / "sources"
RUN_LOG = ROOT / "data" / "metrics" / "ingestion_runs.jsonl"
STAGES = ("manifest", "parse", "chunks", "regions")


def find_pdf_folders(source_root: Path, selection: str | None) -> list[Path]:
    selected = (source_root / selection).resolve() if selection else source_root.resolve()
    if not selected.is_relative_to(source_root.resolve()) or not selected.is_dir():
        raise ValueError(f"selection must be a directory below {source_root}: {selection}")
    folders = {
        pdf.parent for pdf in selected.rglob("*")
        if pdf.is_file() and pdf.suffix.casefold() == ".pdf"
        and not {"processed", "debug"}.intersection(pdf.relative_to(source_root).parts)
    }
    return sorted(folders)


def prepare_folder(folder: Path, source_root: Path, write: bool,
                   accept_inferred: bool) -> None:
    manifest_path = folder / "manifest.yaml"
    if manifest_path.is_file():
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest = load_manifest(manifest_path)
        product_id = manifest.product_id
        version = manifest.version
    else:
        parts = folder.relative_to(source_root).parts
        if not parts:
            raise ValueError(f"put new PDFs under a product folder: {folder}")
        product_id = slug(parts[0])
        version = inferred_version(parts)
        data = {
            "product_id": product_id,
            "display_name": parts[0],
            "version": version,
            "language": "en",
            "documents": [],
        }
    existing = {str(doc["path"]).casefold() for doc in data["documents"]}
    used_ids = {doc["id"] for doc in data["documents"]}
    additions = []
    for pdf in sorted(path for path in folder.iterdir()
                      if path.is_file() and path.suffix.casefold() == ".pdf"):
        if pdf.name.casefold() in existing:
            continue
        kind = inferred_type(pdf.stem)
        additions.append({
            "id": document_id(product_id, pdf.stem, version, used_ids),
            "type": kind,
            "title": pdf.stem.replace("_", " ").replace("-", " ").strip(),
            "path": pdf.name,
            "authority": "primary" if accept_inferred else "unreviewed",
        })
    if not additions:
        print(f"unchanged {manifest_path}")
        return
    data["documents"].extend(additions)
    ProductManifest.model_validate(data)
    for doc in additions:
        print(f"{'write' if write else 'would write'} {manifest_path}: "
              f"{doc['id']} ({doc['type']}, {doc['authority']})")
    if write:
        manifest_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def current_parse(output_dir: Path, source_path: Path) -> bool:
    required = ("profile.json", "parsed.json", "normalized_sections.jsonl", "quality_report.json")
    if not all((output_dir / name).is_file() for name in required):
        return False
    profile = json.loads((output_dir / "profile.json").read_text(encoding="utf-8"))
    return profile.get("source_sha256") == sha256_file(source_path)


def current_chunks(output_dir: Path, manifest: ProductManifest, document) -> bool:
    required = ("parsed.json", "normalized_sections.jsonl",
                "retrieval_chunks.jsonl", "retrieval_chunk_report.json")
    if not all((output_dir / name).is_file() for name in required):
        return False
    report = json.loads((output_dir / "retrieval_chunk_report.json").read_text(encoding="utf-8"))
    tables_path = output_dir / "normalized_tables.jsonl"
    return (
        report.get("assembly_version") == "2"
        and report.get("source_to_chunk_preservation", {}).get("metric_version") == "2"
        and not report.get("source_to_chunk_preservation", {}).get("legacy_provenance_chunk_ids")
        and report.get("retrieval_chunks_sha256") == sha256_file(output_dir / "retrieval_chunks.jsonl")
        and report.get("parsed_sha256") == sha256_file(output_dir / "parsed.json")
        and report.get("normalized_sections_sha256") == sha256_file(output_dir / "normalized_sections.jsonl")
        and report.get("normalized_tables_sha256") == (
            sha256_file(tables_path) if tables_path.is_file() else None
        )
        and report.get("document_metadata_sha256") == document_metadata_sha256(manifest, document)
    )


def current_regions(output_dir: Path, source_path: Path,
                    manifest: ProductManifest, document) -> bool:
    required = ("parsed.json", "normalized_sections.jsonl",
                "visual_regions.jsonl", "visual_region_report.json")
    if not all((output_dir / name).is_file() for name in required):
        return False
    report = json.loads((output_dir / "visual_region_report.json").read_text(encoding="utf-8"))
    return (
        report.get("classifier_version") == CLASSIFIER_VERSION
        and report.get("document_metadata_sha256") ==
        document_metadata_sha256(manifest, document)
        and report.get("source_sha256") == sha256_file(source_path)
        and report.get("parsed_sha256") == sha256_file(output_dir / "parsed.json")
        and report.get("normalized_sections_sha256") ==
        sha256_file(output_dir / "normalized_sections.jsonl")
        and report.get("inventoried_pages") == report.get("source_pages")
    )


def run_stage(manifest_path: Path, manifest: ProductManifest, document, stage: str, run: bool,
              force: bool, debug: bool, parse_pending: bool = False) -> bool:
    source_path = resolve_source_path(manifest_path, document)
    output_dir = resolve_output_dir(source_path, document)
    if document.authority == "unreviewed" or document.type == "unknown":
        print(f"needs manifest review {document.id}: {manifest_path}")
        return False
    if stage == "parse":
        ready = current_parse(output_dir, source_path)
        script = ROOT / "scripts" / "parse_document.py"
    elif stage == "chunks":
        if not parse_pending and not current_parse(output_dir, source_path):
            print(f"needs parse {document.id}: no current parse output")
            return False
        ready = current_chunks(output_dir, manifest, document) if not parse_pending else False
        script = ROOT / "scripts" / "assemble_retrieval_chunks.py"
    else:
        if not parse_pending and not current_parse(output_dir, source_path):
            print(f"needs parse {document.id}: no current parse output")
            return False
        ready = current_regions(output_dir, source_path, manifest, document) if not parse_pending else False
        script = ROOT / "scripts" / "inventory_regions.py"
    if ready and not force:
        print(f"skip {document.id}: existing {stage} output")
        return False
    command = [sys.executable, str(script), "--manifest", str(manifest_path),
               "--document-id", document.id]
    if stage == "parse" and debug:
        command.append("--debug")
    print(f"{'run' if run else 'would run'} {stage} {document.id} -> {output_dir}", flush=True)
    if run:
        started = perf_counter()
        result = subprocess.run(command, cwd=ROOT)
        artifact_bytes = sum(path.stat().st_size for path in output_dir.glob("*") if path.is_file())
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "document_id": document.id,
            "source_pdf": str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT)
                              else source_path),
            "source_bytes": source_path.stat().st_size,
            "wall_seconds": round(perf_counter() - started, 2),
            "output_bytes": artifact_bytes,
            "exit_code": result.returncode,
            "gpu_usage": None,
        }
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("a", encoding="utf-8") as target:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--select", help="Relative product or release folder, e.g. KiCAD/v9")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--run", action="store_true", help="Write manifests and run costly stages")
    parser.add_argument("--debug", action="store_true", help="Write parser inspection exports")
    parser.add_argument("--force", action="store_true", help="Rebuild existing parse or chunks")
    parser.add_argument("--accept-inferred", action="store_true",
                        help="Treat inferred metadata for new PDFs as reviewed")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        parser.error(f"source root does not exist: {source_root}")
    folders = find_pdf_folders(source_root, args.select)
    if not folders:
        parser.error("no PDFs found in selected source folders")
    stages = [stage for stage in STAGES if stage in args.stages]
    for folder in folders:
        if "manifest" in stages:
            prepare_folder(folder, source_root, args.run, args.accept_inferred)
        manifest_path = folder / "manifest.yaml"
        if not any(stage in stages for stage in ("parse", "chunks", "regions")):
            continue
        if not manifest_path.is_file():
            print(f"needs manifest {folder}: run the manifest stage first")
            continue
        manifest = load_manifest(manifest_path)
        for document in manifest.documents:
            parse_pending = False
            if "parse" in stages:
                parse_pending = run_stage(manifest_path, manifest, document, "parse", args.run,
                                          args.force, args.debug)
            if "chunks" in stages:
                run_stage(manifest_path, manifest, document, "chunks", args.run,
                          args.force, args.debug,
                          parse_pending=parse_pending and not args.run)
            if "regions" in stages:
                run_stage(manifest_path, manifest, document, "regions", args.run,
                          args.force, args.debug,
                          parse_pending=parse_pending and not args.run)


if __name__ == "__main__":
    main()
