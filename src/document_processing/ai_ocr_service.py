from __future__ import annotations
import truststore

truststore.inject_into_ssl()

from pathlib import Path

import google.generativeai as genai
import pymupdf as fitz

from src.config.settings import AppSettings


VISION_OCR_PROMPT = (
    "You are an OCR engine. Extract ALL readable text from this document "
    "image exactly as it appears, preserving line breaks where meaningful. "
    "Do not summarize, translate, or add commentary. Return only the "
    "extracted text."
)


class AIOCRService:
    """OCR fallback using Gemini vision-capable model.

    Used only when:
    - Tesseract is unavailable/blocked, AND
    - The user enabled AI Enhanced Mode, AND
    - GEMINI_API_KEY is configured.
    """

    def __init__(self, settings: AppSettings, dpi: int = 220) -> None:
        self.settings = settings
        self.dpi = dpi
        self._configured = False
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self._configured = True

    def is_available(self) -> tuple[bool, str]:
        if not self._configured:
            return False, "Gemini API key is not configured (GEMINI_API_KEY)."
        return True, "Gemini vision OCR is configured."

    def _render_page_png(self, page: fitz.Page) -> bytes:
        matrix = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")

    def run_on_pdf_pages(self, pdf_path: Path, pages: list[int]) -> dict[int, str]:
        results: dict[int, str] = {}
        if not pages:
            return results

        model = genai.GenerativeModel(self.settings.gemini_vision_model)

        with fitz.open(pdf_path) as doc:
            for page_num in pages:
                if page_num < 1 or page_num > doc.page_count:
                    continue
                try:
                    page = doc.load_page(page_num - 1)
                    image_bytes = self._render_page_png(page)
                    response = model.generate_content(
                        [
                            VISION_OCR_PROMPT,
                            {"mime_type": "image/png", "data": image_bytes},
                        ]
                    )
                    text = (getattr(response, "text", "") or "").strip()
                    results[page_num] = text
                except Exception as exc:  # noqa: BLE001
                    results[page_num] = f"[AI OCR error: {exc}]"

        return results
