"""Names inferred from a manual's path or filename: ids, versions and document types."""

import re


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def inferred_version(parts: tuple[str, ...]) -> str | None:
    if len(parts) < 2:
        return None
    label = parts[1]
    if re.fullmatch(r"v\d+(?:\.\d+)*", label, re.IGNORECASE):
        return label[1:]
    if re.fullmatch(r"\d+(?:\.\d+)*", label):
        return label
    return None


def inferred_type(stem: str) -> str:
    name = slug(stem)
    if "admin" in name or "administrator" in name:
        return "administration_manual"
    if "server" in name and ("manual" in name or "guide" in name):
        return "server_manual"
    if "developer" in name or "cookbook" in name:
        return "developer_manual"
    if "getting_started" in name:
        return "getting_started_guide"
    if "user" in name and ("manual" in name or "guide" in name):
        return "user_manual"
    if "install" in name and ("manual" in name or "guide" in name):
        return "installation_manual"
    return "unknown"


def document_id(product_id: str, stem: str, version: str | None, used: set[str]) -> str:
    name = slug(stem)
    if name.startswith(f"{product_id}_"):
        name = name[len(product_id) + 1:]
    version_slug = slug(version or "")
    if version_slug and name.startswith(f"{version_slug}_"):
        name = name[len(version_slug) + 1:]
    base = "_".join(filter(None, (product_id, name, version_slug)))
    result = base
    number = 2
    while result in used:
        result = f"{base}_{number}"
        number += 1
    used.add(result)
    return result
