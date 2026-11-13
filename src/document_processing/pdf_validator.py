from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
from pydantic import BaseModel

from src.config.settings import AppSettings


class PDFValidationError(Exception):
    def __init__(self, message: str, code: str = "pdf_validation_error") -> None:
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class PDFValidationResult(BaseModel):
    is_valid: bool
    file_name: str
    file_size_bytes: int
    page_count: int
    mime_type_hint: str
    starts_with_pdf_signature: bool
    is_password_protected: bool


def _guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def validate_pdf_file(pdf_path: Path, settings: AppSettings) -> PDFValidationResult:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise PDFValidationError("Uploaded file is missing or inaccessible.", code="file_missing")

    if pdf_path.suffix.lower() != ".pdf":
        raise PDFValidationError("Only .pdf files are allowed.", code="invalid_extension")

    file_size = pdf_path.stat().st_size
    if file_size <= 0:
        raise PDFValidationError("The uploaded PDF is empty.", code="empty_file")

    if file_size > settings.max_upload_bytes:
        raise PDFValidationError(
            f"File size exceeds limit ({settings.max_upload_mb} MB).",
            code="file_too_large",
        )

    with pdf_path.open("rb") as f:
        signature = f.read(5)
    has_pdf_signature = signature == b"%PDF-"
    if not has_pdf_signature:
        raise PDFValidationError("Invalid PDF signature. File may be corrupt or not a PDF.", code="invalid_signature")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        raise PDFValidationError("PDF cannot be opened. File may be corrupt.", code="open_failed") from exc

    try:
        is_encrypted = doc.is_encrypted
        if is_encrypted:
            raise PDFValidationError("Password-protected PDFs are not supported.", code="password_protected")

        page_count = doc.page_count
        if page_count <= 0:
            raise PDFValidationError("PDF has no pages.", code="no_pages")

        if page_count > settings.max_pdf_pages:
            raise PDFValidationError(
                f"PDF page count exceeds limit ({settings.max_pdf_pages} pages).",
                code="page_limit_exceeded",
            )
    finally:
        doc.close()

    return PDFValidationResult(
        is_valid=True,
        file_name=pdf_path.name,
        file_size_bytes=file_size,
        page_count=page_count,
        mime_type_hint=_guess_mime_type(pdf_path),
        starts_with_pdf_signature=has_pdf_signature,
        is_password_protected=False,
    )
