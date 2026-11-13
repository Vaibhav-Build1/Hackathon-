from __future__ import annotations

from pathlib import Path
from typing import Any

from src.document_processing.docling_parser import parse_with_docling
from src.document_processing.ocr_service import OCRService
from src.document_processing.text_extractor import extract_native_text_with_quality


def _merge_page_text(
    native_text: dict[int, str],
    docling_text: dict[int, str],
    ocr_text: dict[int, str],
) -> tuple[dict[int, str], dict[int, str]]:
    merged: dict[int, str] = {}
    provenance: dict[int, str] = {}
    pages = sorted(set(native_text.keys()) | set(docling_text.keys()) | set(ocr_text.keys()))
    for p in pages:
        o = (ocr_text.get(p) or "").strip()
        d = (docling_text.get(p) or "").strip()
        n = (native_text.get(p) or "").strip()

        if o:
            merged[p] = o
            provenance[p] = "ocr_fused"
        elif d:
            merged[p] = d
            provenance[p] = "docling"
        else:
            merged[p] = n
            provenance[p] = "native"
    return merged, provenance


def parse_document_hybrid(
    file_path: Path,
    *,
    parser_mode: str = "smart_hybrid",
    force_all_ocr_engines: bool = False,
    enable_docling: bool = True,
    ocr_service: OCRService | None = None,
) -> dict[str, Any]:
    """
    Hybrid parser orchestrator.

    Modes:
    - smart_hybrid: Docling/native first, OCR where needed
    - forced_hybrid: always run Docling + OCR on all pages
    """
    ext = file_path.suffix.lower()
    if ext != ".pdf":
        return {
            "parser_engine": "native_non_pdf",
            "merged_text_by_page": {1: ""},
            "merged_text": "",
            "page_provenance": {1: "native_non_pdf"},
            "docling": {"ok": False, "engine": "not_applicable"},
            "ocr": {"text_by_page": {}, "diagnostics": {}, "engine_summary": "none"},
            "native": {"text_by_page": {}, "quality": {}},
        }

    native_text_by_page, native_quality = extract_native_text_with_quality(file_path)

    # Simple mode path: native extraction only (PyMuPDF/pdfplumber strategies), no Docling/OCR.
    if parser_mode == "smart_hybrid":
        docling_result = {
            "ok": False,
            "engine": "disabled_in_simple_mode",
            "text_by_page": {},
            "merged_text": "",
            "tables": {},
            "blocks": [],
            "confidence": 0.0,
            "diagnostics": {"disabled_in_simple_mode": True},
        }
    elif enable_docling:
        docling_result = parse_with_docling(file_path)
    else:
        docling_result = {
            "ok": False,
            "engine": "disabled",
            "text_by_page": {},
            "merged_text": "",
            "tables": {},
            "blocks": [],
            "confidence": 0.0,
            "diagnostics": {"docling_available": False, "disabled": True},
        }

    docling_text_by_page = docling_result.get("text_by_page", {}) or {}

    ocr_text_by_page: dict[int, str] = {}
    ocr_diagnostics: dict[str, Any] = {}
    ocr_summary = "none"

    if ocr_service is not None and parser_mode == "forced_hybrid":
        total_pages = max(
            [0]
            + list(native_text_by_page.keys())
            + list(docling_text_by_page.keys())
        )

        if parser_mode == "forced_hybrid" or force_all_ocr_engines:
            ocr_pages = list(range(1, total_pages + 1))
        else:
            ocr_pages = [
                p
                for p, txt in native_text_by_page.items()
                if not (txt or "").strip()
                or bool((native_quality.get(p, {}) or {}).get("low_quality"))
            ]

        if ocr_pages:
            ocr_text_by_page = ocr_service.run_on_pdf_pages(file_path, ocr_pages)
            ocr_diagnostics = ocr_service.get_last_diagnostics()
            ocr_summary = ocr_service.summarize_last_run()

    merged_text_by_page, page_provenance = _merge_page_text(
        native_text=native_text_by_page,
        docling_text=docling_text_by_page,
        ocr_text=ocr_text_by_page,
    )

    merged_text = "\n".join((merged_text_by_page.get(p) or "") for p in sorted(merged_text_by_page.keys())).strip()

    parser_engine = "forced_hybrid" if parser_mode == "forced_hybrid" else "smart_hybrid"

    return {
        "parser_engine": parser_engine,
        "merged_text_by_page": merged_text_by_page,
        "merged_text": merged_text,
        "page_provenance": page_provenance,
        "docling": docling_result,
        "ocr": {
            "text_by_page": ocr_text_by_page,
            "diagnostics": ocr_diagnostics,
            "engine_summary": ocr_summary,
        },
        "native": {
            "text_by_page": native_text_by_page,
            "quality": native_quality,
        },
    }
