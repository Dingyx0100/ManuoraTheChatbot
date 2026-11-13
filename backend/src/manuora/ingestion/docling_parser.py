"""Convert a text-based PDF to Docling's structured document format."""

from pathlib import Path
import json
from importlib.metadata import version

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, OcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from manuora.ingestion.files import sha256_file


def save_conversion_confidence(result, source_path: Path, parsed_path: Path) -> None:
    """Persist the inexpensive conversion signal, bound to exactly this parse.

    model_dump_json converts unavailable NaN scores to JSON null. Selective OCR
    changes invalidate this hash; consumers must not reuse stale confidence.
    """
    record = dict(schema_version=1, docling_version=version("docling"),
                  source_sha256=sha256_file(source_path), parsed_sha256=sha256_file(parsed_path),
                  scores=json.loads(result.confidence.model_dump_json()))
    parsed_path.with_suffix(".confidence.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_pdf_converter(ocr_options: OcrOptions | None = None) -> DocumentConverter:
    """Build the shared PDF pipeline used by full parses and OCR page passes."""
    options = PdfPipelineOptions()
    options.do_ocr = ocr_options is not None
    if ocr_options is not None:
        options.ocr_options = ocr_options
    options.heading_hierarchy_options = HeadingHierarchyOptions(
        enabled=True,
        use_style=False,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_pdf(
    source_path: Path,
    output_path: Path,
    page_range: tuple[int, int] | None = None,
    ocr_options: OcrOptions | None = None,
) -> None:
    """Parse a PDF; OCR stays off unless options are explicitly provided."""
    converter = create_pdf_converter(ocr_options)
    convert_options = {"page_range": page_range} if page_range else {}
    result = converter.convert(source_path, **convert_options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.document.save_as_json(output_path, image_mode=ImageRefMode.PLACEHOLDER)
    save_conversion_confidence(result, source_path, output_path)
