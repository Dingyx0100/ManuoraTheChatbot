"""Read the source manifests (data/sources/**/manifest.yaml): products, versions and manual titles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Manual:
    product_id: str
    display_name: str
    version: str | None
    document_id: str
    title: str
    type: str
    path: str
    manifest: Path

    @property
    def folder(self) -> Path:
        return self.manifest.parent


def manifests(source_root: Path) -> list[Path]:
    return sorted(path for path in source_root.rglob("manifest.yaml")
                  if not {"processed", "debug"}.intersection(path.relative_to(source_root).parts))


def manuals(source_root: Path) -> list[Manual]:
    found = []
    for path in manifests(source_root):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for document in data.get("documents") or []:
            found.append(Manual(
                product_id=str(data["product_id"]), display_name=str(data.get("display_name") or data["product_id"]),
                version=None if data.get("version") is None else str(data["version"]),
                document_id=str(document["id"]), title=str(document.get("title") or document["id"]),
                type=str(document.get("type") or "unknown"), path=str(document["path"]), manifest=path))
    return found


def names(source_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """({product_id: display name}, {document_id: title})."""
    found = manuals(source_root)
    return ({m.product_id: m.display_name for m in found}, {m.document_id: m.title for m in found})
