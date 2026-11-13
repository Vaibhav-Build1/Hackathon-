from pathlib import Path

from intelligence_plugin.classifiers import classify_document
from intelligence_plugin.extractors import extract_structured_data
from intelligence_plugin.publishers import publish


def run_pipeline(pdf_path: Path, output_dir: Path, targets: list[str]) -> dict:
    doc_type = classify_document(pdf_path)
    structured = extract_structured_data(pdf_path, doc_type)
    published = publish(structured_data=structured, output_dir=output_dir, targets=targets, source_pdf=pdf_path)

    return {
        "document_type": doc_type,
        "structured_data": structured,
        "published": published,
        "api_payload": {
            "source": str(pdf_path),
            "document_type": doc_type,
            "data": structured,
        },
    }
