"""Manifest data supplied before a document is ingested."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentManifest(BaseModel):
    """Identify one source document and its basic publication metadata."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    path: Path
    authority: str = Field(default="primary", min_length=1)
    revision: str | None = None
    publication_date: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None


class ProductManifest(BaseModel):
    """Describe one product version and the documents that support it."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    language: str = Field(default="en", min_length=1)
    publication_date: date | None = None
    documents: list[DocumentManifest] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_document_ids(self) -> ProductManifest:
        ids = [document.id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document IDs must be unique within a product manifest")
        return self
