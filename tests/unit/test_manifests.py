from pathlib import Path
import subprocess
import sys

import yaml

import pytest
from pydantic import ValidationError

from manuora.knowledge.schemas import ProductManifest
from manuora.ingestion.files import resolve_output_dir


def test_manifest_requires_distinct_document_ids() -> None:
    source = {
        "product_id": "freecad",
        "display_name": "FreeCAD",
        "version": "1.0",
        "documents": [
            {
                "id": "user_manual",
                "type": "user_manual",
                "title": "User Manual",
                "path": "FreeCAD_UserManual.pdf",
            }
        ],
    }

    manifest = ProductManifest.model_validate(source)
    assert manifest.version == "1.0"
    assert manifest.documents[0].path == Path("FreeCAD_UserManual.pdf")

    source["documents"].append(source["documents"][0].copy())
    with pytest.raises(ValidationError, match="document IDs must be unique"):
        ProductManifest.model_validate(source)


def test_document_outputs_do_not_collide_and_legacy_parse_is_reused(tmp_path: Path) -> None:
    user_pdf = tmp_path / "user.pdf"
    admin_pdf = tmp_path / "admin.pdf"
    user_pdf.touch()
    admin_pdf.touch()
    documents = ProductManifest.model_validate({
        "product_id": "nextcloud", "display_name": "Nextcloud", "version": "34",
        "documents": [
            {"id": "nextcloud_user_manual_34", "type": "user_manual",
             "title": "User Manual", "path": user_pdf.name},
            {"id": "nextcloud_administration_manual_34", "type": "administration_manual",
             "title": "Administration Manual", "path": admin_pdf.name},
        ],
    }).documents
    assert resolve_output_dir(user_pdf, documents[0]) != resolve_output_dir(admin_pdf, documents[1])
    (tmp_path / "profile.json").write_text(
        '{"document_id":"nextcloud_user_manual_34","file_name":"user.pdf"}', encoding="utf-8"
    )
    assert resolve_output_dir(user_pdf, documents[0]) == tmp_path
    assert resolve_output_dir(admin_pdf, documents[1]) == tmp_path / "processed" / documents[1].id


def test_generic_intake_creates_reviewable_manifest_for_new_product(tmp_path: Path) -> None:
    folder = tmp_path / "KiCAD" / "v9"
    folder.mkdir(parents=True)
    (folder / "KiCAD_User_Guide.pdf").touch()
    script = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"
    command = [sys.executable, str(script), "--source-root", str(tmp_path),
               "--select", "KiCAD/v9", "--stages", "manifest"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert not (folder / "manifest.yaml").exists()
    subprocess.run([*command, "--run"], check=True, capture_output=True, text=True)
    data = yaml.safe_load((folder / "manifest.yaml").read_text(encoding="utf-8"))
    assert data["product_id"] == "kicad"
    assert data["version"] == "9"
    assert data["documents"][0]["type"] == "user_manual"
    assert data["documents"][0]["authority"] == "unreviewed"
