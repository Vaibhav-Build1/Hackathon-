from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf as fitz


def _fallback_pdf_parse(path: Path) -> dict[str, Any]:
    text_by_page: dict[int, str] = {}
    tables: dict[str, list[dict[str, Any]]] = {}
    total_chars = 0

    with fitz.open(path) as doc:
        for i in range(doc.page_count):
            txt = (doc.load_page(i).get_text("text") or "").strip()
            text_by_page[i + 1] = txt
            total_chars += len(txt)

    confidence = 0.9 if total_chars > 200 else 0.5 if total_chars > 0 else 0.0
    return {
        "ok": True,
        "engine": "fallback_pymupdf",
        "text_by_page": text_by_page,
        "merged_text": "\n".join(v for _, v in sorted(text_by_page.items())),
        "tables": tables,
        "blocks": [],
        "confidence": confidence,
        "diagnostics": {
            "docling_available": False,
            "docling_error": "docling_not_installed_or_not_used",
            "fallback_used": True,
            "total_chars": total_chars,
        },
    }


def parse_with_docling(path: Path) -> dict[str, Any]:
    """
    Safe parser wrapper:
    - Tries Docling if available.
    - Falls back to PyMuPDF native parse.
    Returns a unified structure expected by the hybrid router.
    """
    if not path.exists():
        return {
            "ok": False,
            "engine": "none",
            "text_by_page": {},
            "merged_text": "",
            "tables": {},
            "blocks": [],
            "confidence": 0.0,
            "diagnostics": {"error": "file_not_found"},
        }

    if path.suffix.lower() != ".pdf":
        return _fallback_pdf_parse(path)

    try:
        # Optional import: keep app working even if docling isn't installed.
        # The exact Docling API can vary by version; we guard this call.
        from docling.document_converter import DocumentConverter  # type: ignore

        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = getattr(result, "document", None)
        if doc is None:
            return _fallback_pdf_parse(path)

        text_by_page: dict[int, str] = {}
        blocks: list[dict[str, Any]] = []
        tables: dict[str, list[dict[str, Any]]] = {}

        # Conservative extraction with version-tolerant guards.
        page_count = 0
        if hasattr(doc, "pages") and doc.pages:
            for p_idx, page in enumerate(doc.pages, start=1):
                page_count += 1
                page_text = ""
                if hasattr(page, "text"):
                    page_text = str(getattr(page, "text") or "").strip()
                text_by_page[p_idx] = page_text

        if not text_by_page:
            # Fallback to global markdown/text export
            page_text = ""
            if hasattr(doc, "export_to_markdown"):
                page_text = str(doc.export_to_markdown() or "")
            elif hasattr(doc, "text"):
                page_text = str(getattr(doc, "text") or "")
            text_by_page = {1: page_text.strip()}
            page_count = 1

        merged_text = "\n".join(v for _, v in sorted(text_by_page.items()))
        char_count = len(merged_text.strip())
        confidence = 0.95 if char_count > 300 else 0.65 if char_count > 0 else 0.0

        return {
            "ok": True,
            "engine": "docling",
            "text_by_page": text_by_page,
            "merged_text": merged_text,
            "tables": tables,
            "blocks": blocks,
            "confidence": confidence,
            "diagnostics": {
                "docling_available": True,
                "fallback_used": False,
                "pages_detected": page_count,
                "char_count": char_count,
            },
        }
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_pdf_parse(path)
        fallback["diagnostics"]["docling_error"] = str(exc)
        return fallback
