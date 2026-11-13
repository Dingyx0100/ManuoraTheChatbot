"""Typed, reproducible diagnostics for a source PDF and its parse."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProfilingThresholds(BaseModel):
    min_body_text_chars: int = 40
    image_heavy_ratio: float = 0.50
    ocr_image_ratio: float = 0.35
    blank_image_ratio: float = 0.02
    blank_drawing_ratio: float = 0.02
    body_margin_ratio: float = 0.08


class PageProfile(BaseModel):
    page_number: int
    text_char_count: int
    body_text_char_count: int
    text_object_count: int
    image_count: int
    image_area_ratio: float
    drawing_count: int
    drawing_area_ratio: float
    column_count: int | None
    toc_numbered_rows: int
    numbered_line_count: int
    likely_blank: bool
    likely_image_heavy: bool
    likely_ocr_needed: bool
    likely_toc: bool


class DocumentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profiler_version: str = "1"
    source_sha256: str
    source_bytes: int
    pdf_engine: str
    file_name: str
    product_id: str | None = None
    product_version: str | None = None
    document_id: str | None = None
    page_count: int
    native_text_page_ratio: float
    image_heavy_page_ratio: float
    blank_page_ratio: float
    ocr_candidate_page_ratio: float
    multi_column_page_ratio: float
    toc_detected: bool
    toc_pages: list[int]
    numbered_heading_candidate_count: int
    numbered_heading_candidate_page_ratio: float
    thresholds: ProfilingThresholds
    page_profiles: list[PageProfile]
    warnings: list[str] = Field(default_factory=list)


class ParsingPlan(BaseModel):
    schema_version: int = 1
    planner_version: str = "1"
    source_sha256: str
    status: Literal["advisory"] = "advisory"
    parser: Literal["docling"] = "docling"
    ocr_mode: Literal["off", "selective", "full"]
    ocr_pages: list[int]
    toc_mode: Literal["standard", "multi_column", "ignore"]
    heading_strategy: list[str]
    warnings: list[str] = Field(default_factory=list)


class PageQuality(BaseModel):
    page_number: int
    native_body_text_chars: int
    parsed_text_chars: int
    likely_ocr_needed: bool
    parsed_text_present: bool | None = None  # OCR presence only; no minimum length gate.
    # Whole-page counts, separate from the legacy body/presence diagnostic.
    native_text_chars: int | None = None
    parsed_coverage_chars: int | None = None
    native_text_tokens: int | None = None
    parsed_text_tokens: int | None = None
    native_character_coverage_ratio: float | None = None
    native_token_coverage_ratio: float | None = None
    coverage_status: Literal["PASS", "WARN", "FAIL", "NOT_APPLICABLE", "UNMEASURED"] = "UNMEASURED"
    coverage_notes: list[str] = Field(default_factory=list)


class TextCoverageSettings(BaseModel):
    """Count retention thresholds, not independently calibrated fidelity scores."""

    fail_below: float = Field(default=0.80, ge=0, le=1)
    warn_below: float = Field(default=0.95, ge=0, le=1)
    tokenizer_id: str = "Qwen/Qwen3-Embedding-0.6B"
    tokenizer_revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


class ParsingQualityReport(BaseModel):
    schema_version: int = 1
    source_sha256: str
    status: Literal["PASS", "WARN", "FAIL"]
    retrieval_ready: bool
    page_count: int
    native_character_coverage_ratio: float | None = None
    native_token_coverage_ratio: float | None = None
    text_coverage_settings: TextCoverageSettings | None = None
    text_coverage_scope: str | None = None
    token_counting_status: str | None = None
    text_coverage_failed_pages: list[int] = Field(default_factory=list)
    text_coverage_warning_pages: list[int] = Field(default_factory=list)
    text_coverage_unmeasured_pages: list[int] = Field(default_factory=list)
    text_coverage_notes: list[str] = Field(default_factory=list)
    parsed_sha256: str | None = None
    ocr_candidate_pages: int
    ocr_candidate_uncovered_pages: list[int]
    ocr_execution_complete: bool | None = None
    ocr_accepted_pages: int | None = None
    section_count: int
    matched_section_count: int
    heading_resolution_rate: float | None
    outline_heading_coverage: float | None = None
    section_specificity: float | None = None
    outline_quality: dict | None = None
    section_specificity_details: dict | None = None
    outline_evaluation_wall_seconds: float | None = None
    body_item_context_rate: float | None = None
    orphan_block_ratio: float | None
    duplicate_text_ratio: float | None
    chunk_context_resolution_rate: float | None
    parsed_table_count: int
    toc_alignment_rate: float | None = None
    table_parse_success_rate: float | None = None
    reasons: list[str]
    page_quality: list[PageQuality]
    failure_detection: dict | None = None
