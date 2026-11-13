from __future__ import annotations

import re
from typing import Any

from src.nlp.text_normalizer import normalize_text


def _first_match(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _evidence(text: str, value: str) -> str:
    if not value:
        return ""
    low = text.lower()
    idx = low.find(value.lower())
    if idx < 0:
        return value
    s = max(0, idx - 40)
    e = min(len(text), idx + len(value) + 60)
    return text[s:e].replace("\n", " ").strip()


def extract_entities(document_type: str, text: str, fallback_fields: dict[str, Any] | None = None) -> dict[str, Any]:
    text = normalize_text(text)
    fallback_fields = fallback_fields or {}

    fields: dict[str, Any] = dict(fallback_fields)
    field_confidence: dict[str, float] = {}
    extraction_evidence: dict[str, str] = {}

    if document_type == "invoice_receipt":
        candidates = {
            "invoice_number": _first_match(r"(?:invoice\s*(?:no|number|#)\s*[:#]?\s*)([A-Za-z0-9\-_/]+)", text),
            "date": _first_match(
                r"(?:invoice\s*date|date)\s*[:#]?\s*([0-9]{1,2}[\-/][0-9]{1,2}[\-/][0-9]{2,4}|[A-Za-z]+\s+[0-9]{1,2},\s*[0-9]{4})",
                text,
            ),
            "total_amount": _first_match(r"(?:total\s*(?:due|amount)?\s*[:$]?\s*)([\$€£]?\s?[0-9][0-9,]*\.?[0-9]{0,2})", text),
            "tax_amount": _first_match(r"(?:tax\s*[:$]?\s*)([\$€£]?\s?[0-9][0-9,]*\.?[0-9]{0,2})", text),
            "vendor": _first_match(r"(?:from|seller|vendor|bill\s*from)\s*[:]?\s*([^\n]+)", text),
            "bill_to": _first_match(r"(?:bill\s*to)\s*[:]?\s*([^\n]+)", text),
        }
    elif document_type == "resume":
        candidates = {
            "candidate_name": (text.splitlines()[0].strip() if text.splitlines() else ""),
            "email": _first_match(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text),
            "phone": _first_match(r"(\+?\d[\d\s().\-]{7,}\d)", text),
            "linkedin": _first_match(r"(https?://(?:www\.)?linkedin\.com/[^\s]+)", text),
        }
    else:
        candidates = {}

    for k, v in candidates.items():
        if v:
            fields[k] = v
            field_confidence[k] = 0.92
            extraction_evidence[k] = _evidence(text, v)

    for k, v in fields.items():
        if k not in field_confidence:
            vv = str(v).strip() if v is not None else ""
            field_confidence[k] = 0.65 if vv else 0.0
            extraction_evidence[k] = _evidence(text, vv) if vv else ""

    return {
        "fields": fields,
        "field_confidence": field_confidence,
        "extraction_evidence": extraction_evidence,
    }
