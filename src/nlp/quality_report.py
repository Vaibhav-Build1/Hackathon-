from __future__ import annotations

from typing import Any


def build_quality_report(document_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []

    if document_type == "invoice_receipt":
        required = ["invoice_number", "date", "total_amount"]
        missing = [k for k in required if not str(fields.get(k, "")).strip()]
        if missing:
            flags.append(f"missing_required_fields:{','.join(missing)}")

    if document_type == "resume":
        required = ["candidate_name", "email"]
        missing = [k for k in required if not str(fields.get(k, "")).strip()]
        if missing:
            flags.append(f"missing_required_fields:{','.join(missing)}")

    numeric_like = ["total_amount", "tax_amount", "subtotal"]
    for k in numeric_like:
        val = str(fields.get(k, "")).strip()
        if val and not any(ch.isdigit() for ch in val):
            flags.append(f"invalid_numeric_field:{k}")

    requires_review = len(flags) > 0

    return {
        "quality_flags": flags,
        "requires_review": requires_review,
    }
