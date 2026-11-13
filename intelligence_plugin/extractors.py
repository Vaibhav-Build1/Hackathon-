import re
from pathlib import Path

import pdfplumber


def _extract_text_and_tables(pdf_path: Path) -> tuple[str, list[list[list[str]]]]:
    pages_text = []
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
            page_tables = page.extract_tables() or []
            tables.extend(page_tables)
    return "\n".join(pages_text), tables


def _kv_from_lines(text: str) -> dict:
    data = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                data[key] = value
    return data


def _invoice_fields(text: str) -> dict:
    out = {}
    patterns = {
        "invoice_number": r"(?:invoice\s*(?:no|number|#)\s*[:#]?\s*)([A-Za-z0-9\-_/]+)",
        "date": r"(?:invoice\s*date|date)\s*[:#]?\s*([0-9]{1,2}[\-/][0-9]{1,2}[\-/][0-9]{2,4}|[A-Za-z]+\s+[0-9]{1,2},\s*[0-9]{4})",
        "total_amount": r"(?:total\s*(?:due|amount)?\s*[:$]?\s*)([\$€£]?\s?[0-9][0-9,]*\.?[0-9]{0,2})",
        "tax_amount": r"(?:tax\s*[:$]?\s*)([\$€£]?\s?[0-9][0-9,]*\.?[0-9]{0,2})",
        "vendor": r"(?:from|seller|vendor|bill\s*from)\s*[:]?\s*([^\n]+)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            out[field] = match.group(1).strip()
    return out


def _resume_fields(text: str) -> dict:
    out = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        out["candidate_name"] = lines[0]

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(\+?\d[\d\s().\-]{7,}\d)", text)
    linkedin_match = re.search(r"(https?://(?:www\.)?linkedin\.com/[^\s]+)", text, flags=re.IGNORECASE)

    if email_match:
        out["email"] = email_match.group(0)
    if phone_match:
        out["phone"] = phone_match.group(0)
    if linkedin_match:
        out["linkedin"] = linkedin_match.group(1)

    return out


def _normalize_tables(raw_tables: list[list[list[str]]]) -> dict:
    normalized = {}
    for idx, table in enumerate(raw_tables, start=1):
        if not table:
            continue
        headers = [str(cell or "").strip() for cell in table[0]]
        rows = []
        for row in table[1:]:
            values = [str(cell or "").strip() for cell in row]
            row_dict = {headers[i] if i < len(headers) and headers[i] else f"col_{i+1}": values[i] if i < len(values) else "" for i in range(max(len(headers), len(values)))}
            rows.append(row_dict)
        normalized[f"table_{idx}"] = rows
    return normalized


def extract_structured_data(pdf_path: Path, document_type: str) -> dict:
    text, tables = _extract_text_and_tables(pdf_path)
    result = {
        "document_type": document_type,
        "fields": {},
        "raw_key_values": _kv_from_lines(text),
        "tables": _normalize_tables(tables),
        "source_text_preview": "\n".join(text.splitlines()[:20]),
    }

    if document_type == "invoice_receipt":
        result["fields"] = _invoice_fields(text)
    elif document_type == "resume":
        result["fields"] = _resume_fields(text)

    return result
