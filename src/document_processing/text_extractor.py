from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
import pymupdf as fitz


DOC_KEYWORDS = {
    "invoice_receipt": ("invoice", "total", "tax", "subtotal", "bill to", "amount due"),
    "resume": ("experience", "education", "skills", "linkedin", "curriculum vitae"),
}


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _score_text(text: str) -> float:
    if not text:
        return 0.0

    length_score = min(len(text) / 3000.0, 1.0)

    printable = sum(ch.isprintable() for ch in text)
    printable_ratio = printable / max(len(text), 1)

    alnum = sum(ch.isalnum() for ch in text)
    alnum_ratio = alnum / max(len(text), 1)

    keyword_hits = 0
    low = text.lower()
    for kws in DOC_KEYWORDS.values():
        for kw in kws:
            if kw in low:
                keyword_hits += 1
    keyword_score = min(keyword_hits / 8.0, 1.0)

    score = (0.45 * length_score) + (0.2 * printable_ratio) + (0.2 * alnum_ratio) + (0.15 * keyword_score)
    return round(score, 4)


def _extract_page_text_strategies_fitz(page: fitz.Page) -> dict[str, str]:
    outputs = {
        "fitz_text": _normalize_text(page.get_text("text") or ""),
        "fitz_blocks": _normalize_text("\n".join([b[4] for b in (page.get_text("blocks") or []) if len(b) > 4 and str(b[4]).strip()])),
        "fitz_words": _normalize_text(" ".join([w[4] for w in (page.get_text("words") or []) if len(w) > 4 and str(w[4]).strip()])),
    }
    return outputs


def extract_native_text_with_quality(pdf_path: Path) -> tuple[dict[int, str], dict[int, dict]]:
    text_by_page: dict[int, str] = {}
    quality_by_page: dict[int, dict] = {}

    with fitz.open(pdf_path) as doc, pdfplumber.open(pdf_path) as plumber_pdf:
        for idx in range(doc.page_count):
            page_num = idx + 1
            fitz_page = doc.load_page(idx)
            candidates = _extract_page_text_strategies_fitz(fitz_page)

            plumber_text = ""
            if idx < len(plumber_pdf.pages):
                plumber_text = _normalize_text(plumber_pdf.pages[idx].extract_text() or "")
            candidates["pdfplumber_text"] = plumber_text

            scored = {name: _score_text(txt) for name, txt in candidates.items()}
            winner = max(scored, key=scored.get)
            best_text = candidates[winner]

            text_by_page[page_num] = best_text
            quality_by_page[page_num] = {
                "selected_strategy": winner,
                "selected_score": scored[winner],
                "candidate_scores": scored,
                "char_count": len(best_text),
                "low_quality": scored[winner] < 0.2 or len(best_text) < 40,
            }

    return text_by_page, quality_by_page


def extract_native_text(pdf_path: Path) -> dict[int, str]:
    text_by_page, _ = extract_native_text_with_quality(pdf_path)
    return text_by_page
