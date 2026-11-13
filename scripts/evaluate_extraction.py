from __future__ import annotations

import json
import os
from pathlib import Path

from src.document_processing.multi_format_extractor import extract_any_document
from src.workflows.document_workflow import run_document_workflow


def evaluate_file(path: Path) -> dict:
    ext = path.suffix.lower()
    if ext == ".pdf":
        result = run_document_workflow(str(path))
        extracted = result.get("extraction_result", {})
        doc_type = result.get("document_type", "generic")
        parser_metadata = result.get("parser_metadata", {}) or {}
    else:
        extracted = extract_any_document(path)
        doc_type = extracted.get("document_type", "generic")
        parser_metadata = {}

    fields = extracted.get("fields", {}) or {}
    field_conf = extracted.get("field_confidence", {}) or {}
    quality_flags = extracted.get("quality_flags", []) or []
    requires_review = bool(extracted.get("requires_review", False))

    filled_fields = sum(1 for _, v in fields.items() if str(v).strip())
    avg_conf = (sum(float(v) for v in field_conf.values()) / len(field_conf)) if field_conf else 0.0

    return {
        "file": str(path),
        "document_type": doc_type,
        "field_count": len(fields),
        "filled_field_count": filled_fields,
        "avg_field_confidence": round(avg_conf, 4),
        "quality_flag_count": len(quality_flags),
        "requires_review": requires_review,
        "parser_engine": parser_metadata.get("parser_engine", ""),
        "ocr_engine_summary": (parser_metadata.get("ocr", {}) or {}).get("engine_summary", ""),
        "docling_engine": (parser_metadata.get("docling", {}) or {}).get("engine", ""),
        "docling_confidence": (parser_metadata.get("docling", {}) or {}).get("confidence", 0.0),
    }


def run_eval_with_mode(files: list[Path], parser_mode: str, force_all_ocr_engines: bool) -> list[dict]:
    os.environ["PARSER_MODE"] = parser_mode
    os.environ["FORCE_ALL_OCR_ENGINES"] = "true" if force_all_ocr_engines else "false"

    report: list[dict] = []
    for p in files:
        row = evaluate_file(p)
        row["eval_parser_mode"] = parser_mode
        row["eval_force_all_ocr_engines"] = force_all_ocr_engines
        report.append(row)
    return report


def main() -> None:
    base = Path("storage/uploads")
    files = []
    for pattern in ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.docx", "*.xlsx", "*.xls"]:
        files.extend(base.glob(pattern))

    files = sorted(files)
    if not files:
        print("No files found under storage/uploads")
        return

    smart_report = run_eval_with_mode(files, parser_mode="smart_hybrid", force_all_ocr_engines=False)
    forced_report = run_eval_with_mode(files, parser_mode="forced_hybrid", force_all_ocr_engines=True)

    out = {
        "total_files": len(files),
        "smart_hybrid": smart_report,
        "forced_hybrid": forced_report,
    }

    out_path = Path("outputs/nlp_eval_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report: {out_path}")


if __name__ == "__main__":
    main()
