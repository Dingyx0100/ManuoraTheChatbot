"""Create reviewable table corrections beside an unchanged Docling parse."""

import json
import re
from pathlib import Path

import yaml
from docling_core.types.doc import DoclingDocument

from manuora.ingestion.chunk_context import map_item_context
from manuora.ingestion.files import sha256_file


def format_corrected_table_row(table: dict, row: dict) -> str:
    if table["kind"] == "configuration":
        return (f"{row['variable']}. Default: {row['default']}. "
                f"Meaning: {row['meaning']}")
    if table["kind"] == "status_icons":
        return (f"{row['state']}. Status icon: {row['status']} "
                f"Mono tray icon: {row['mono_tray']} "
                f"Color tray icon: {row['color_tray']}")
    raise ValueError(f"unknown corrected table kind: {table['kind']}")


def _text(cell: dict | None) -> str:
    return (cell or {}).get("text", "").strip()


def _clean_prose(value: str) -> str:
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _variable(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("-", ""))


def _default(value: str, spill: str) -> str:
    value, spill = _clean_prose(value), _clean_prose(spill)
    if not spill or spill == value:
        return value
    if spill.startswith("(") and ")" in value and "(" not in value:
        number, suffix = value.split(" ", 1)
        return f"{number} {spill} {suffix}"
    return f"{value} {spill}".strip()


def _config_rows(raw_grid: list[list[dict]], spec: dict) -> list[dict]:
    """Collapse Docling's spurious fourth column and wrapped source rows."""
    grid = [[_text(cell) for cell in line] for line in raw_grid]
    header_index = next(
        (i for i, line in enumerate(grid) if line[:2] == ["Variable", "Default"]),
        None,
    )
    if header_index is None or len(grid[header_index]) != 4:
        raise ValueError("configuration table no longer has the expected raw header")
    rows: list[dict] = []
    expected = spec["expected_variables"]
    overrides = spec.get("overrides", {})
    if not set(overrides) <= set(expected):
        raise ValueError("configuration override names differ from expected variables")
    for raw_index, cells in enumerate(grid[header_index + 1:], start=header_index + 1):
        raw_name, raw_default, spill, raw_meaning = cells
        name = _variable(raw_name)
        name = spec.get("variable_aliases", {}).get(name, name)
        default = _default(raw_default, spill)
        meaning = _clean_prose(raw_meaning)
        next_name = expected[len(rows)] if len(rows) < len(expected) else None
        if name != next_name and not (name and next_name and next_name.startswith(name)):
            # Some wrapped rows contain only the end of the preceding name,
            # default or meaning. They must not become standalone settings.
            if not rows or (name and rows[-1]["variable"] + name != rows[-1]["expected_name"]):
                if not rows or raw_name.strip() not in {"", "Interval"}:
                    raise ValueError(f"unexpected configuration row: {raw_name!r}")
            previous = rows[-1]
            previous["raw_row_indices"].append(raw_index)
            if name:
                previous["variable"] += name
            if default:
                previous["default"] += f" {default}"
            if meaning:
                previous["meaning"] += f" {meaning}"
            continue
        rows.append({
            "variable": name,
            "default": default,
            "meaning": meaning,
            "expected_name": next_name,
            "raw_row_indices": [raw_index],
        })

    for row in rows:
        row.pop("expected_name")
        if patch := overrides.get(row["variable"]):
            row.update(patch)
        row["meaning"] = _clean_prose(row["meaning"])
    if [row["variable"] for row in rows] != expected:
        raise ValueError("configuration variable list differs from reviewed source")
    if any(not row["variable"] or not row["default"] or not row["meaning"] for row in rows):
        raise ValueError("corrected configuration table contains an empty field")
    return rows


def _icon_rows(raw_grid: list[list[dict]], spec: dict) -> list[dict]:
    if [_text(cell) for cell in raw_grid[0]] != ["State", "Status", "Mono tray", "Color tray"]:
        raise ValueError("icon table header differs from reviewed source")
    states = [_text(line[0]) for line in raw_grid[1:]]
    rows = spec["rows"]
    if states != [row["state"] for row in rows]:
        raise ValueError("icon states differ from reviewed source")
    if any(set(row) != {"state", "status", "mono_tray", "color_tray"} for row in rows):
        raise ValueError("each icon row needs descriptions for all three icons")
    return [{**row, "raw_row_indices": [index + 1]} for index, row in enumerate(rows)]


def _markdown(headers: list[str], rows: list[dict], keys: list[str]) -> str:
    def safe(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(safe(row[key]) for key in keys) + " |" for row in rows)
    return "\n".join(lines)


def export_normalized_tables(
    parsed_path: Path, pdf_path: Path, sections_path: Path, annotations_path: Path
) -> list[dict]:
    """Validate source-specific corrections and save canonical table content."""
    annotations = yaml.safe_load(annotations_path.read_text(encoding="utf-8"))
    if annotations.get("schema_version") != 1:
        raise ValueError("unsupported table annotation schema")
    source_hash = sha256_file(pdf_path)
    if annotations["source_sha256"] != source_hash:
        raise ValueError("table annotations belong to a different PDF")
    raw = json.loads(parsed_path.read_text(encoding="utf-8"))
    document = DoclingDocument.load_from_json(parsed_path)
    sections = [json.loads(line) for line in sections_path.read_text(encoding="utf-8").splitlines()]
    contexts = map_item_context(document, sections)
    tables = {table["self_ref"]: table for table in raw["tables"]}
    records = []
    for spec in annotations["tables"]:
        ref = spec["ref"]
        table = tables[ref]
        page = table["prov"][0]["page_no"]
        grid = table["data"]["grid"]
        shape = [table["data"]["num_rows"], table["data"]["num_cols"]]
        if page != spec["pdf_page"] or shape != spec["raw_shape"]:
            raise ValueError(f"raw table {ref} changed page or shape")
        if spec["kind"] == "configuration":
            headers = ["Variable", "Default", "Meaning"]
            keys = ["variable", "default", "meaning"]
            rows = _config_rows(grid, spec)
        elif spec["kind"] == "status_icons":
            headers = ["State", "Status icon", "Mono tray icon", "Color tray icon"]
            keys = ["state", "status", "mono_tray", "color_tray"]
            rows = _icon_rows(grid, spec)
        else:
            raise ValueError(f"unknown correction kind: {spec['kind']}")
        records.append({
            "table_ref": ref,
            "source_pdf": pdf_path.name,
            "pdf_page": page,
            "title": spec["title"],
            "kind": spec["kind"],
            "section_path": contexts[ref]["normalized_heading_path"],
            "source_pdf_sha256": source_hash,
            "parsed_sha256": sha256_file(parsed_path),
            "raw_shape": shape,
            "headers": headers,
            "rows": rows,
            "markdown": _markdown(headers, rows, keys),
            "normalization_source": annotations_path.name,
            "review_status": "first_pass_verified",
        })
    output_path = parsed_path.parent / "normalized_tables.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records
