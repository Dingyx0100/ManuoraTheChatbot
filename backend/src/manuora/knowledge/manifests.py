"""Read product manifests from YAML files."""

from pathlib import Path

import yaml

from manuora.knowledge.schemas import ProductManifest


def load_manifest(path: Path) -> ProductManifest:
    """Read YAML and validate it against the manifest schema."""
    with path.open("r", encoding="utf-8") as source:
        data = yaml.safe_load(source)
    return ProductManifest.model_validate(data)
