import re
from pathlib import Path

import pdfplumber


RULES = {
    "invoice_receipt": [r"invoice", r"receipt", r"total", r"tax", r"subtotal", r"bill to"],
    "resume": [r"experience", r"education", r"skills", r"linkedin", r"curriculum vitae", r"objective"],
}


def _read_text(pdf_path: Path, max_pages: int = 3) -> str:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks).lower()


def classify_document(pdf_path: Path) -> str:
    text = _read_text(pdf_path)
    scores = {name: 0 for name in RULES}

    for doc_type, patterns in RULES.items():
        for pattern in patterns:
            scores[doc_type] += len(re.findall(pattern, text, flags=re.IGNORECASE))

    winner = max(scores, key=scores.get)
    if scores[winner] == 0:
        return "generic"
    return winner
