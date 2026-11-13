from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
from pydantic import BaseModel, Field


class PageInspection(BaseModel):
    page_number: int
    text_length: int
    alpha_char_count: int
    image_count: int
    text_density: float
    requires_ocr: bool


class PDFInspectionResult(BaseModel):
    pdf_type: str
    total_pages: int
    native_text_pages: int
    ocr_pages: int
    mixed_pages: int
    extracted_characters: int
    pages_requiring_ocr: list[int] = Field(default_factory=list)
    page_details: list[PageInspection] = Field(default_factory=list)


def _estimate_text_density(text: str, width: float, height: float) -> float:
    area = max(width * height, 1.0)
    return round(len(text) / area, 6)


def inspect_pdf(pdf_path: Path, min_text_chars_per_page: int = 30) -> PDFInspectionResult:
    page_details: list[PageInspection] = []
    total_chars = 0
    native_text_pages = 0
    pages_requiring_ocr: list[int] = []

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count

        for idx in range(total_pages):
            page = doc.load_page(idx)
            text = page.get_text("text") or ""
            text_length = len(text)
            alpha_count = sum(1 for ch in text if ch.isalpha())
            image_count = len(page.get_images(full=True))

            rect = page.rect
            density = _estimate_text_density(text, rect.width, rect.height)

            has_sufficient_text = text_length >= min_text_chars_per_page and alpha_count >= 15
            requires_ocr = not has_sufficient_text

            if has_sufficient_text:
                native_text_pages += 1
            else:
                pages_requiring_ocr.append(idx + 1)

            total_chars += text_length
            page_details.append(
                PageInspection(
                    page_number=idx + 1,
                    text_length=text_length,
                    alpha_char_count=alpha_count,
                    image_count=image_count,
                    text_density=density,
                    requires_ocr=requires_ocr,
                )
            )

    ocr_pages = len(pages_requiring_ocr)
    mixed_pages = min(native_text_pages, ocr_pages)

    if native_text_pages == total_pages:
        pdf_type = "native_text_pdf"
    elif ocr_pages == total_pages:
        pdf_type = "scanned_image_pdf"
    else:
        pdf_type = "mixed_pdf"

    return PDFInspectionResult(
        pdf_type=pdf_type,
        total_pages=total_pages,
        native_text_pages=native_text_pages,
        ocr_pages=ocr_pages,
        mixed_pages=mixed_pages,
        extracted_characters=total_chars,
        pages_requiring_ocr=pages_requiring_ocr,
        page_details=page_details,
    )
