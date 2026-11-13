from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union

import cv2
import easyocr
import numpy as np
import pymupdf as fitz
from PIL import Image


class OCRService:
    def __init__(self, tesseract_cmd: str | None, language: str = "eng", dpi: int = 400, preprocess: bool = True) -> None:
        # Keep signature for compatibility with existing app flow.
        self._legacy_cmd = tesseract_cmd.strip() if tesseract_cmd else None
        self.language = language
        self.dpi = dpi
        self.preprocess = preprocess

        self._easy_reader = None
        self._easy_init_error = None
        self._last_diagnostics: dict[str, Any] = {}

        try:
            lang_map = {"eng": "en", "en": "en"}
            easy_lang = lang_map.get((self.language or "eng").lower(), "en")
            self._easy_reader = easyocr.Reader([easy_lang], gpu=False)
        except Exception as exc:  # noqa: BLE001
            self._easy_init_error = str(exc)

    def is_available(self) -> tuple[bool, str]:
        if self._easy_reader is None:
            return False, f"EasyOCR unavailable: {self._easy_init_error}"
        return True, "Available OCR engines: easyocr"

    def get_last_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def _pix_to_ndarray(self, pix: fitz.Pixmap) -> np.ndarray:
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        if pix.n >= 3:
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return bgr

        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.equalizeHist(gray)
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    def _normalize_line(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text.lower()

    def _score_lines(self, lines: list[str], confidences: list[float] | None = None) -> float:
        if not lines:
            return 0.0
        unique = len({self._normalize_line(x) for x in lines if x.strip()})
        char_count = sum(len(x.strip()) for x in lines)
        conf = float(np.mean(confidences)) if confidences else 0.5
        return unique * 1.5 + char_count * 0.02 + conf * 10

    def _dedupe_preserve_order(self, lines: list[str]) -> list[str]:
        seen = set()
        out = []
        for line in lines:
            norm = self._normalize_line(line)
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            out.append(line.strip())
        return out

    def _easyocr_lines(self, image: np.ndarray) -> tuple[list[str], list[float]]:
        if self._easy_reader is None:
            return [], []
        try:
            raw = self._easy_reader.readtext(image, detail=1, paragraph=False)
            lines, confs = [], []
            for item in raw:
                if len(item) >= 3:
                    txt = str(item[1]).strip()
                    cf = float(item[2]) if item[2] is not None else 0.5
                    if txt:
                        lines.append(txt)
                        confs.append(cf)
            return lines, confs
        except Exception:
            return [], []

    def _merge_easyocr(self, image_bgr: np.ndarray, context_key: str = "default") -> str:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        processed = self._preprocess_image(image_bgr) if self.preprocess else gray
        upscaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        variants = [gray, processed, upscaled]
        variant_outputs: list[dict[str, Any]] = []

        for idx, var in enumerate(variants, start=1):
            e_lines, e_conf = self._easyocr_lines(var)
            variant_outputs.append(
                {
                    "engine": f"easyocr_v{idx}",
                    "variant": idx,
                    "lines": e_lines,
                    "conf": e_conf,
                }
            )

        for block in variant_outputs:
            block["score"] = self._score_lines(block["lines"], block["conf"])
            block["line_count"] = len(block["lines"])

        ranked = sorted(variant_outputs, key=lambda x: x["score"], reverse=True)

        merged_lines: list[str] = []
        for block in ranked:
            if not block["lines"]:
                continue
            merged_lines.extend(block["lines"])

        merged_lines = self._dedupe_preserve_order(merged_lines)

        per_engine_counts = {"easyocr": 0}
        for block in variant_outputs:
            per_engine_counts["easyocr"] += int(block["line_count"])

        top = ranked[0] if ranked else None
        self._last_diagnostics[context_key] = {
            "available_engines": {
                "easyocr": self._easy_reader is not None,
            },
            "per_engine_line_counts": per_engine_counts,
            "top_engine_variant": top["engine"] if top else "none",
            "top_engine_score": float(top["score"]) if top else 0.0,
            "merged_line_count": len(merged_lines),
            "engine_variants": [
                {
                    "engine": b["engine"],
                    "score": float(b["score"]),
                    "line_count": int(b["line_count"]),
                }
                for b in ranked
            ],
        }

        if not merged_lines and ranked:
            for block in ranked:
                if block["lines"]:
                    return "\n".join(block["lines"]).strip()

        return "\n".join(merged_lines).strip()

    def summarize_last_run(self) -> str:
        if not self._last_diagnostics:
            return "none"

        easy_lines = 0
        for _, diag in self._last_diagnostics.items():
            counts = diag.get("per_engine_line_counts", {})
            easy_lines += int(counts.get("easyocr", 0))

        if easy_lines > 0:
            return "easyocr"

        sample_diag = next(iter(self._last_diagnostics.values()))
        if sample_diag.get("available_engines", {}).get("easyocr"):
            return "available(easyocr)"
        return "none"

    def _ocr_page(self, page: fitz.Page, page_num: int) -> str:
        matrix = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = self._pix_to_ndarray(pix)
        return self._merge_easyocr(img, context_key=f"page_{page_num}")

    def run_on_pdf_pages(self, pdf_path: Path, pages: list[int]) -> dict[int, str]:
        results: dict[int, str] = {}
        self._last_diagnostics = {}
        if not pages:
            return results

        with fitz.open(pdf_path) as doc:
            for page_num in pages:
                if page_num < 1 or page_num > doc.page_count:
                    continue
                try:
                    page = doc.load_page(page_num - 1)
                    results[page_num] = self._ocr_page(page, page_num)
                except Exception:
                    results[page_num] = ""

        return results

    def run_on_image(self, image_path: Union[str, Path]) -> dict[int, str]:
        path = Path(image_path)
        self._last_diagnostics = {}
        if not path.exists():
            return {1: ""}

        try:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            text = self._merge_easyocr(img, context_key="page_1")
            return {1: text}
        except Exception:
            return {1: ""}
