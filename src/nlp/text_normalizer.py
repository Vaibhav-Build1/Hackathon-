from __future__ import annotations

import re
from typing import Iterable


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dehyphenate_line_breaks(text: str) -> str:
    # "invo-\nice" -> "invoice"
    return re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)


def normalize_punctuation(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[ ]+([,:;.!?])", r"\1", text)
    text = re.sub(r"([,:;])([^\s])", r"\1 \2", text)
    return text


def normalize_common_ocr_noise(text: str) -> str:
    # Conservative replacements to avoid over-correction.
    replacements = {
        " ltem ": " item ",
        " tota1 ": " total ",
        " lnvoice ": " invoice ",
        " bi11 ": " bill ",
    }
    padded = f" {text} "
    for bad, good in replacements.items():
        padded = padded.replace(bad, good)
    return padded.strip()


def normalize_text(text: str) -> str:
    out = text or ""
    out = dehyphenate_line_breaks(out)
    out = normalize_punctuation(out)
    out = normalize_common_ocr_noise(out)
    out = normalize_whitespace(out)
    return out


def merge_text_pages(native_text_by_page: dict[int, str], ocr_text_by_page: dict[int, str]) -> str:
    pages = sorted(set(native_text_by_page.keys()) | set(ocr_text_by_page.keys()))
    chunks: list[str] = []
    for p in pages:
        native = (native_text_by_page.get(p) or "").strip()
        ocr = (ocr_text_by_page.get(p) or "").strip()

        if native and ocr:
            base = native if len(native) >= len(ocr) else ocr
        else:
            base = native or ocr

        if base:
            chunks.append(f"[Page {p}]\n{base}")

    return normalize_text("\n\n".join(chunks))


def tokenize_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    return [ln for ln in lines if ln]


def sample_for_semantics(lines: Iterable[str], max_chars: int = 6000) -> str:
    buf: list[str] = []
    count = 0
    for ln in lines:
        if count >= max_chars:
            break
        piece = ln[:400]
        if count + len(piece) > max_chars:
            piece = piece[: max_chars - count]
        buf.append(piece)
        count += len(piece) + 1
    return "\n".join(buf).strip()
