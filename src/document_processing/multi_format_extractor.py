from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf as fitz
from docx import Document

from intelligence_plugin.classifiers import classify_document
from intelligence_plugin.extractors import extract_structured_data
from src.nlp.doc_classifier import classify_document_ensemble
from src.nlp.entity_extractor import extract_entities
from src.nlp.quality_report import build_quality_report
from src.nlp.text_normalizer import normalize_text


def _kv_from_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                out[k] = v
    return out


def _extract_pdf(path: Path) -> dict[str, Any]:
    doc_type_legacy = classify_document(path)
    structured = extract_structured_data(path, doc_type_legacy)

    native_text_by_page: dict[int, str] = {}
    with fitz.open(path) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            native_text_by_page[i + 1] = (page.get_text("text") or "").strip()

    merged_text = normalize_text("\n".join(native_text_by_page.values()))
    ensemble = classify_document_ensemble(merged_text) if merged_text else None
    doc_type = ensemble.document_type if ensemble else doc_type_legacy

    nlp_entities = extract_entities(
        document_type=doc_type,
        text=merged_text or structured.get("source_text_preview", ""),
        fallback_fields=structured.get("fields", {}),
    )
    fields = nlp_entities.get("fields", structured.get("fields", {}))
    quality = build_quality_report(doc_type, fields)

    return {
        "document_type": doc_type,
        "document_type_confidence": float(ensemble.confidence) if ensemble else 0.0,
        "classification_score_breakdown": ensemble.score_breakdown if ensemble else {},
        "fields": fields,
        "field_confidence": nlp_entities.get("field_confidence", {}),
        "extraction_evidence": nlp_entities.get("extraction_evidence", {}),
        "quality_flags": quality.get("quality_flags", []),
        "requires_review": quality.get("requires_review", False),
        "raw_key_values": structured.get("raw_key_values", {}),
        "tables": structured.get("tables", {}),
        "native_text_by_page": native_text_by_page,
        "ocr_text_by_page": {},
        "pdf_type": "pdf",
        "total_pages": len(native_text_by_page),
    }


def _extract_image(path: Path) -> dict[str, Any]:
    # Keep extraction text container; OCR handled by existing OCR flow.
    return {
        "document_type": "generic",
        "fields": {},
        "raw_key_values": {},
        "tables": {},
        "native_text_by_page": {1: ""},
        "ocr_text_by_page": {},
        "pdf_type": "image",
        "total_pages": 1,
    }


def _extract_docx(path: Path) -> dict[str, Any]:
    doc = Document(path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    text = normalize_text("\n".join(lines))

    tables: dict[str, list[dict[str, str]]] = {}
    for idx, table in enumerate(doc.tables, start=1):
        rows = []
        if not table.rows:
            continue
        headers = [c.text.strip() or f"col_{i+1}" for i, c in enumerate(table.rows[0].cells)]
        for r in table.rows[1:]:
            vals = [c.text.strip() for c in r.cells]
            row = {headers[i] if i < len(headers) else f"col_{i+1}": vals[i] if i < len(vals) else "" for i in range(max(len(headers), len(vals)))}
            rows.append(row)
        tables[f"table_{idx}"] = rows

    ensemble = classify_document_ensemble(text) if text else None
    doc_type = ensemble.document_type if ensemble else "generic"
    nlp_entities = extract_entities(document_type=doc_type, text=text, fallback_fields={})
    fields = nlp_entities.get("fields", {})
    quality = build_quality_report(doc_type, fields)

    return {
        "document_type": doc_type,
        "document_type_confidence": float(ensemble.confidence) if ensemble else 0.0,
        "classification_score_breakdown": ensemble.score_breakdown if ensemble else {},
        "fields": fields,
        "field_confidence": nlp_entities.get("field_confidence", {}),
        "extraction_evidence": nlp_entities.get("extraction_evidence", {}),
        "quality_flags": quality.get("quality_flags", []),
        "requires_review": quality.get("requires_review", False),
        "raw_key_values": _kv_from_lines(text),
        "tables": tables,
        "native_text_by_page": {1: text},
        "ocr_text_by_page": {},
        "pdf_type": "word",
        "total_pages": 1,
    }


def _extract_excel(path: Path) -> dict[str, Any]:
    xl = pd.ExcelFile(path)
    text_lines = []
    tables: dict[str, list[dict[str, Any]]] = {}

    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        safe = sheet[:31] or "Sheet"
        tables[safe] = df.fillna("").to_dict(orient="records")
        text_lines.append(f"[Sheet: {sheet}]")
        if not df.empty:
            text_lines.append(df.to_string(index=False))
        text_lines.append("")

    text = normalize_text("\n".join(text_lines).strip())
    ensemble = classify_document_ensemble(text) if text else None
    doc_type = ensemble.document_type if ensemble else "generic"
    nlp_entities = extract_entities(document_type=doc_type, text=text, fallback_fields={})
    fields = nlp_entities.get("fields", {})
    quality = build_quality_report(doc_type, fields)

    return {
        "document_type": doc_type,
        "document_type_confidence": float(ensemble.confidence) if ensemble else 0.0,
        "classification_score_breakdown": ensemble.score_breakdown if ensemble else {},
        "fields": fields,
        "field_confidence": nlp_entities.get("field_confidence", {}),
        "extraction_evidence": nlp_entities.get("extraction_evidence", {}),
        "quality_flags": quality.get("quality_flags", []),
        "requires_review": quality.get("requires_review", False),
        "raw_key_values": _kv_from_lines(text),
        "tables": tables,
        "native_text_by_page": {1: text},
        "ocr_text_by_page": {},
        "pdf_type": "excel",
        "total_pages": 1,
    }


def extract_any_document(path: Path) -> dict[str, Any]:
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in {".png", ".jpg", ".jpeg"}:
        return _extract_image(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in {".xlsx", ".xls"}:
        return _extract_excel(path)

    raise ValueError(f"Unsupported file type: {ext}")
