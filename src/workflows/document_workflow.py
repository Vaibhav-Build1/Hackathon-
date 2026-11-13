from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from intelligence_plugin.classifiers import classify_document
from intelligence_plugin.extractors import extract_structured_data
from langgraph.graph import END, START, StateGraph
from src.config.settings import get_settings
from src.document_processing.ocr_service import OCRService
from src.document_processing.parser_router import parse_document_hybrid
from src.nlp.doc_classifier import classify_document_ensemble
from src.nlp.entity_extractor import extract_entities
from src.nlp.quality_report import build_quality_report
from src.nlp.text_normalizer import normalize_text


class WorkflowState(TypedDict, total=False):
    pdf_path: str
    document_type: str
    extraction_result: dict
    normalized_text: str
    document_type_confidence: float
    classification_score_breakdown: dict
    parser_metadata: dict
    error: str


def _classify_node(state: WorkflowState) -> WorkflowState:
    try:
        settings = get_settings()

        parser_mode = str(state.get("parser_mode", settings.parser_mode))
        force_all_ocr_engines = bool(state.get("force_all_ocr_engines", settings.force_all_ocr_engines))
        enable_docling = bool(state.get("enable_docling", settings.enable_docling))
        ocr_dpi = int(state.get("ocr_dpi", settings.ocr_dpi))
        ocr_preprocess = bool(state.get("ocr_preprocess", True))

        if parser_mode != "forced_hybrid":
            force_all_ocr_engines = False

        ocr_service = OCRService(
            tesseract_cmd=settings.tesseract_cmd,
            language=settings.ocr_language,
            dpi=ocr_dpi,
            preprocess=ocr_preprocess,
        )

        hybrid = parse_document_hybrid(
            Path(state["pdf_path"]),
            parser_mode=parser_mode,
            force_all_ocr_engines=force_all_ocr_engines,
            enable_docling=enable_docling,
            ocr_service=ocr_service,
        )

        merged_text = hybrid.get("merged_text", "")
        normalized = normalize_text(merged_text)
        legacy_type = classify_document(Path(state["pdf_path"]))

        ensemble = classify_document_ensemble(normalized) if normalized else None
        doc_type = ensemble.document_type if ensemble else legacy_type
        conf = float(ensemble.confidence) if ensemble else 0.0
        breakdown = ensemble.score_breakdown if ensemble else {}

        parser_metadata = {
            "parser_engine": hybrid.get("parser_engine", settings.parser_mode),
            "page_provenance": hybrid.get("page_provenance", {}),
            "docling": hybrid.get("docling", {}),
            "ocr": {
                "engine_summary": (hybrid.get("ocr", {}) or {}).get("engine_summary", "none"),
                "diagnostics": (hybrid.get("ocr", {}) or {}).get("diagnostics", {}),
            },
            "native_quality": (hybrid.get("native", {}) or {}).get("quality", {}),
        }

        return {
            **state,
            "document_type": doc_type or legacy_type,
            "normalized_text": normalized,
            "document_type_confidence": conf,
            "classification_score_breakdown": breakdown,
            "parser_metadata": parser_metadata,
        }
    except Exception as exc:  # noqa: BLE001
        return {**state, "document_type": "generic", "error": f"classification_error: {exc}"}


def _extract_node(state: WorkflowState) -> WorkflowState:
    try:
        doc_type = state.get("document_type", "generic")
        extracted = extract_structured_data(Path(state["pdf_path"]), doc_type)

        normalized_text = state.get("normalized_text", "")
        nlp_entities = extract_entities(
            document_type=doc_type,
            text=normalized_text or extracted.get("source_text_preview", ""),
            fallback_fields=extracted.get("fields", {}),
        )

        fields = nlp_entities.get("fields", extracted.get("fields", {}))
        quality = build_quality_report(doc_type, fields)

        extracted["fields"] = fields
        extracted["field_confidence"] = nlp_entities.get("field_confidence", {})
        extracted["extraction_evidence"] = nlp_entities.get("extraction_evidence", {})
        extracted["quality_flags"] = quality.get("quality_flags", [])
        extracted["requires_review"] = quality.get("requires_review", False)
        extracted["document_type_confidence"] = state.get("document_type_confidence", 0.0)
        extracted["classification_score_breakdown"] = state.get("classification_score_breakdown", {})
        extracted["normalized_text_preview"] = (normalized_text or "")[:3000]
        extracted["parser_metadata"] = state.get("parser_metadata", {})

        return {**state, "extraction_result": extracted}
    except Exception as exc:  # noqa: BLE001
        return {
            **state,
            "extraction_result": {
                "document_type": state.get("document_type", "generic"),
                "fields": {},
                "raw_key_values": {},
                "tables": {},
                "source_text_preview": "",
                "field_confidence": {},
                "extraction_evidence": {},
                "quality_flags": [f"extraction_error:{exc}"],
                "requires_review": True,
                "document_type_confidence": state.get("document_type_confidence", 0.0),
                "classification_score_breakdown": state.get("classification_score_breakdown", {}),
                "normalized_text_preview": (state.get("normalized_text", "") or "")[:3000],
                "parser_metadata": state.get("parser_metadata", {}),
                "extraction_error": str(exc),
            },
            "error": f"{state.get('error', '')} extraction_error: {exc}".strip(),
        }


def _needs_extract(_: WorkflowState) -> str:
    return "extract"


def build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node("classify", _classify_node)
    graph.add_node("extract", _extract_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _needs_extract, {"extract": "extract"})
    graph.add_edge("extract", END)
    return graph.compile()


_WORKFLOW = build_workflow()


def run_document_workflow(
    pdf_path: str,
    parser_mode: str | None = None,
    force_all_ocr_engines: bool | None = None,
    enable_docling: bool | None = None,
    ocr_dpi: int | None = None,
    ocr_preprocess: bool = True,
) -> dict:
    invoke_state: WorkflowState = {"pdf_path": pdf_path}
    if parser_mode is not None:
        invoke_state["parser_mode"] = parser_mode
    if force_all_ocr_engines is not None:
        invoke_state["force_all_ocr_engines"] = force_all_ocr_engines
    if enable_docling is not None:
        invoke_state["enable_docling"] = enable_docling
    if ocr_dpi is not None:
        invoke_state["ocr_dpi"] = int(ocr_dpi)
    invoke_state["ocr_preprocess"] = bool(ocr_preprocess)

    result = _WORKFLOW.invoke(invoke_state)
    return {
        "document_type": result.get("document_type", "generic"),
        "document_type_confidence": result.get("document_type_confidence", 0.0),
        "classification_score_breakdown": result.get("classification_score_breakdown", {}),
        "extraction_result": result.get("extraction_result", {}),
        "parser_metadata": result.get("parser_metadata", {}),
        "workflow_error": result.get("error"),
    }
