"""File operations shared by ingestion stages."""

from hashlib import sha256
import json
from pathlib import Path

from manuora.knowledge.schemas import DocumentManifest, ProductManifest


def resolve_source_path(manifest_path: Path, document: DocumentManifest) -> Path:
    """Find a manifest's document inside the manifest's directory."""
    directory = manifest_path.parent.resolve()
    source_path = (directory / document.path).resolve()
    if not source_path.is_relative_to(directory):
        raise ValueError(f"document path leaves manifest directory: {document.path}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    return source_path


def resolve_output_dir(source_path: Path, document: DocumentManifest) -> Path:
    """Use isolated outputs, retaining a matching legacy parse beside its PDF."""
    if document.id in {".", ".."} or "/" in document.id or "\\" in document.id:
        raise ValueError(f"document ID is not a directory name: {document.id}")
    legacy_profile = source_path.parent / "profile.json"
    if legacy_profile.is_file():
        try:
            profile = json.loads(legacy_profile.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            profile = {}
        if profile.get("document_id") == document.id and profile.get("file_name") == source_path.name:
            return source_path.parent
    return source_path.parent / "processed" / document.id


def sha256_file(path: Path) -> str:
    """Hash a source file without loading the entire file into memory."""
    digest = sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def document_metadata_sha256(manifest: ProductManifest, document: DocumentManifest) -> str:
    """Fingerprint metadata copied into each retrieval chunk."""
    payload = {
        "product_id": manifest.product_id,
        "product_version": manifest.version,
        "language": manifest.language,
        "document": document.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()
