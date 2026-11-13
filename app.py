import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.settings import AppSettings, get_settings
from src.document_processing.ai_ocr_service import AIOCRService
from src.document_processing.ocr_service import OCRService
from src.document_processing.pdf_inspector import inspect_pdf
from src.document_processing.pdf_validator import PDFValidationError, validate_pdf_file
from src.document_processing.text_extractor import extract_native_text_with_quality
from src.document_processing.multi_format_extractor import extract_any_document
from src.exporters.format_exporters import build_export_package, build_preview_tables
from intelligence_plugin.publishers import ResumeTemplatePublisher
from src.llm.document_assistant import DocumentAssistant
from src.workflows.document_workflow import run_document_workflow
from src.storage.sqlite_store import SQLiteStore


def _init_page() -> None:
    st.set_page_config(
        page_title="INTELLIGENCE PLUGIN",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _sidebar_settings(settings: AppSettings) -> dict:
    st.sidebar.header("Processing Settings")
    mode = st.sidebar.selectbox("Mode", ["Local", "AI Enhanced"], index=0)

    parser_mode = "smart_hybrid"
    enable_docling = bool(getattr(settings, "enable_docling", True))
    force_all_ocr_engines = False

    if mode == "Local":
        parser_mode_options = ["simple_native", "forced_hybrid"]
        parser_mode_ui = st.sidebar.selectbox("Processing Strategy", parser_mode_options, index=0)

        # Internal routing keeps smart_hybrid as the simple/native lane.
        parser_mode = "smart_hybrid" if parser_mode_ui == "simple_native" else "forced_hybrid"

        if parser_mode == "forced_hybrid":
            enable_docling = st.sidebar.toggle("Enable Docling Parser", value=enable_docling)
            force_all_ocr_engines = st.sidebar.toggle(
                "Force EasyOCR on all pages",
                value=True,
            )

    ocr_enabled = True
    preprocess = True

    st.sidebar.markdown("---")
    st.sidebar.caption("Simple Native mode uses PyMuPDF/pdfplumber native extraction only.")
    st.sidebar.caption("Forced Hybrid mode uses Docling + EasyOCR (all pages) for maximum extraction coverage.")
    st.sidebar.caption("AI Enhanced OCR fallback (Gemini) is used only in AI Enhanced mode when local OCR is unavailable.")

    if parser_mode != "forced_hybrid":
        force_all_ocr_engines = False

    return {
        "mode": mode,
        "parser_mode": parser_mode,
        "enable_docling": enable_docling,
        "force_all_ocr_engines": force_all_ocr_engines,
        "ocr_enabled": ocr_enabled,
        "ocr_language": "eng",
        "ocr_dpi": 400,
        "preprocess": preprocess,
        "tesseract_cmd": settings.tesseract_cmd or "",
    }


def _render_header() -> None:
    st.title("INTELLIGENCE PLUGIN")
    st.caption("Upload a PDF. Understand it. Structure it. Export it.")
    st.markdown("---")


def _render_upload_limits(settings: AppSettings) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Upload Size (MB)", settings.max_upload_mb)
    c2.metric("Max PDF Pages", settings.max_pdf_pages)
    c3.metric("Batch Upload Limit", settings.max_batch_files)

    # st.caption(
    #     "Allowed formats: PDF, DOCX, XLSX, XLS, JPG, JPEG, PNG · "
    #     f"Max file size: {settings.max_upload_mb} MB per file · "
    #     f"Max PDFs per run: {settings.max_batch_files} files · "
    #     f"Max pages per PDF: {settings.max_pdf_pages}"
    # )


def _save_bytes_to_temp(content: bytes, original_name: str) -> Path:
    suffix = Path(original_name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)


def _show_native_text_preview(native_text_by_page: dict, max_chars: int = 6000) -> None:
    preview = "\n\n".join([f"[Page {p}]\n{text}" for p, text in native_text_by_page.items() if text.strip()])
    preview = preview[:max_chars] if preview else ""
    if preview:
        st.text_area("Native Text Preview", value=preview, height=300)
    else:
        st.info("No native text extracted from this PDF.")


def _process_document(temp_pdf_path: Path, uploaded_name: str, settings: AppSettings, ui_cfg: dict) -> dict:
    start = time.perf_counter()

    with st.status("Processing document...", expanded=True) as status:
        ext = temp_pdf_path.suffix.lower()

        if ext == ".pdf":
            st.write("Validating PDF...")
            metadata = validate_pdf_file(temp_pdf_path, settings=settings)

            st.write("Inspecting pages...")
            inspection = inspect_pdf(temp_pdf_path)

            st.write("Extracting native text...")
            native_text_by_page, native_text_quality = extract_native_text_with_quality(temp_pdf_path)
        else:
            st.write("Processing non-PDF document...")
            metadata = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "file_size_bytes": temp_pdf_path.stat().st_size if temp_pdf_path.exists() else 0,
                "page_count": 1,
                "model_dump": lambda: {
                    "is_valid": True,
                    "errors": [],
                    "warnings": [],
                    "file_size_bytes": temp_pdf_path.stat().st_size if temp_pdf_path.exists() else 0,
                    "page_count": 1,
                },
            }
            extracted_any = extract_any_document(temp_pdf_path)
            native_text_by_page = extracted_any.get("native_text_by_page", {1: ""})
            native_text_quality = {
                p: {
                    "selected_strategy": f"native_{extracted_any.get('pdf_type', 'generic')}",
                    "selected_score": 1.0 if (txt or "").strip() else 0.0,
                    "candidate_scores": {},
                    "char_count": len(txt or ""),
                    "low_quality": not bool((txt or "").strip()),
                }
                for p, txt in native_text_by_page.items()
            }
            inspection = type(
                "InspectionObj",
                (),
                {
                    "pdf_type": extracted_any.get("pdf_type", "generic"),
                    "total_pages": extracted_any.get("total_pages", 1),
                    "native_text_pages": sum(1 for t in native_text_by_page.values() if (t or "").strip()),
                    "pages_requiring_ocr": [p for p, t in native_text_by_page.items() if not (t or "").strip()],
                    "extracted_characters": sum(len(t or "") for t in native_text_by_page.values()),
                    "model_dump": lambda self: {
                        "pdf_type": self.pdf_type,
                        "total_pages": self.total_pages,
                        "native_text_pages": self.native_text_pages,
                        "pages_requiring_ocr": self.pages_requiring_ocr,
                        "extracted_characters": self.extracted_characters,
                    },
                },
            )()

        # Build OCR service lazily only if this stage truly needs OCR.
        ocr_service = None

        ocr_text_by_page = {}
        ocr_warning = None
        ocr_engine_used = "none"
        ocr_diagnostics = {}
        if ui_cfg["ocr_enabled"]:
            force_all_ui = bool(ui_cfg.get("force_all_ocr_engines", False) or ui_cfg.get("parser_mode") == "forced_hybrid")
            if ui_cfg.get("parser_mode") == "forced_hybrid":
                st.write("Checking local OCR availability...")
            pages_for_ocr = list(range(1, int(inspection.total_pages) + 1)) if (temp_pdf_path.suffix.lower() == ".pdf" and force_all_ui) else inspection.pages_requiring_ocr

            needs_local_ocr = False
            if temp_pdf_path.suffix.lower() == ".pdf" and pages_for_ocr:
                needs_local_ocr = True
            if temp_pdf_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                needs_local_ocr = True

            ocr_available = False
            ocr_msg = "OCR not initialized"
            if needs_local_ocr:
                ocr_service = OCRService(
                    tesseract_cmd=settings.tesseract_cmd,
                    language=ui_cfg["ocr_language"],
                    dpi=ui_cfg["ocr_dpi"],
                    preprocess=ui_cfg["preprocess"],
                )
                ocr_available, ocr_msg = ocr_service.is_available()
            else:
                ocr_msg = "Simple Native mode skips OCR"

            if not ocr_available and not needs_local_ocr:
                ocr_engine_used = "none (simple native mode)"

            if ocr_available and temp_pdf_path.suffix.lower() == ".pdf":
                if pages_for_ocr:
                    if force_all_ui:
                        st.write("Running EasyOCR on ALL pages (forced mode)...")
                    else:
                        st.write("Running EasyOCR on OCR-required pages...")
                    if ocr_service is not None:
                        ocr_text_by_page = ocr_service.run_on_pdf_pages(temp_pdf_path, pages_for_ocr)
                        ocr_diagnostics = ocr_service.get_last_diagnostics()
                        ocr_engine_used = ocr_service.summarize_last_run()
                else:
                    ocr_engine_used = "none (no pages selected for OCR)"
            elif ocr_available and temp_pdf_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.write("Running EasyOCR on image...")
                if ocr_service is not None:
                    ocr_text_by_page = ocr_service.run_on_image(temp_pdf_path)
                    ocr_diagnostics = ocr_service.get_last_diagnostics()
                    ocr_engine_used = ocr_service.summarize_last_run()
            elif ui_cfg["mode"] == "AI Enhanced" and pages_for_ocr and ui_cfg.get("parser_mode") == "forced_hybrid":
                llm_provider = (settings.llm_provider or "").strip().lower()
                if llm_provider == "openrouter":
                    ocr_warning = (
                        "Local OCR unavailable and AI OCR fallback is disabled for OpenRouter-only testing. "
                        "Native extraction will continue."
                    )
                else:
                    st.write("Local OCR unavailable. Trying AI Enhanced OCR fallback (Gemini)...")
                    ai_ocr_service = AIOCRService(settings=settings)
                    ai_available, ai_msg = ai_ocr_service.is_available()
                    if ai_available:
                        ocr_text_by_page = ai_ocr_service.run_on_pdf_pages(temp_pdf_path, pages_for_ocr)
                        ocr_engine_used = "gemini_vision"
                    else:
                        ocr_warning = (
                            f"Local OCR unavailable ({ocr_msg}). AI Enhanced OCR also unavailable: {ai_msg}. "
                            "Native extraction will continue."
                        )
            else:
                ocr_warning = f"OCR unavailable: {ocr_msg}. Native extraction will continue."

        if temp_pdf_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            native_has_text = any((v or "").strip() for v in native_text_by_page.values())
            ocr_has_text = any((v or "").strip() for v in ocr_text_by_page.values())
            if not native_has_text and not ocr_has_text:
                ocr_warning = (
                    "No text could be extracted from this image. Try a clearer/higher-resolution image, "
                    "increase OCR DPI, or disable preprocessing."
                )

        parser_metadata = {}
        if temp_pdf_path.suffix.lower() == ".pdf":
            st.write("Running extraction workflow (LangGraph)...")
            workflow_result = run_document_workflow(
                str(temp_pdf_path),
                parser_mode=ui_cfg.get("parser_mode"),
                force_all_ocr_engines=ui_cfg.get("force_all_ocr_engines"),
                enable_docling=ui_cfg.get("enable_docling"),
                ocr_dpi=ui_cfg.get("ocr_dpi"),
                ocr_preprocess=ui_cfg.get("preprocess", True),
            )
            document_type = workflow_result.get("document_type", "generic")
            extraction_result = workflow_result.get("extraction_result", {})
            parser_metadata = workflow_result.get("parser_metadata", {}) or {}
        else:
            extracted_any = extract_any_document(temp_pdf_path)
            document_type = extracted_any.get("document_type", "generic")
            extraction_result = {
                "fields": extracted_any.get("fields", {}),
                "raw_key_values": extracted_any.get("raw_key_values", {}),
                "tables": extracted_any.get("tables", {}),
            }

        duration = round(time.perf_counter() - start, 3)
        status.update(label="Processing complete", state="complete", expanded=False)

    low_quality_pages = [p for p, q in native_text_quality.items() if q.get("low_quality")]
    structured_data = {
        "document_type": document_type,
        "file_name": uploaded_name,
        "pdf_type": inspection.pdf_type,
        "fields": extraction_result.get("fields", {}),
        "raw_key_values": extraction_result.get("raw_key_values", {}),
        "tables": extraction_result.get("tables", {}),
        "metrics": inspection.model_dump(),
        "native_text_quality": native_text_quality,
        "low_quality_native_pages": low_quality_pages,
        "native_text_excerpt": {k: v[:800] for k, v in native_text_by_page.items()},
        "ocr_text_excerpt": {k: v[:800] for k, v in ocr_text_by_page.items()},
        "parser_metadata": parser_metadata,
    }

    api_payload = {
        "source_file": uploaded_name,
        "processing_mode": ui_cfg["mode"],
        "document_type": structured_data["document_type"],
        "data": structured_data,
    }

    package = build_export_package(
        base_name=uploaded_name,
        structured_data=structured_data,
        native_text_by_page=native_text_by_page,
        api_payload=api_payload,
    )

    metadata_dump = metadata.model_dump() if hasattr(metadata, "model_dump") else {
        "is_valid": metadata.get("is_valid", True),
        "errors": metadata.get("errors", []),
        "warnings": metadata.get("warnings", []),
        "file_size_bytes": metadata.get("file_size_bytes", 0),
        "page_count": metadata.get("page_count", 1),
    }

    return {
        "uploaded_name": uploaded_name,
        "metadata": metadata_dump,
        "inspection": inspection.model_dump(),
        "native_text_by_page": native_text_by_page,
        "ocr_text_by_page": ocr_text_by_page,
        "ocr_warning": ocr_warning,
        "ocr_engine_used": ocr_engine_used,
        "ocr_diagnostics": ocr_diagnostics,
        "duration": duration,
        "structured_data": structured_data,
        "api_payload": api_payload,
        "package": package,
        "mode": ui_cfg["mode"],
        "parser_metadata": parser_metadata,
    }


def _table_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(lambda x: "" if x is None else str(x))
    return safe


def _send_via_outlook_with_attachment(file_name: str, file_bytes: bytes, selected_format: str) -> tuple[bool, str]:
    try:
        import pythoncom  # type: ignore
        import win32com.client  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return False, f"pywin32/Outlook COM unavailable: {exc}"

    try:
        pythoncom.CoInitialize()

        mail_tmp_dir = Path(tempfile.gettempdir()) / "intelligence_plugin_mail"
        mail_tmp_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attach_path = mail_tmp_dir / f"{stamp}_{file_name}"
        attach_path.write_bytes(file_bytes)

        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)
        mail.Subject = f"Document Output - {selected_format}"
        mail.Body = "Please find the Attached document."
        mail.Attachments.Add(str(attach_path))
        mail.Display()
        return True, f"Outlook draft opened with attachment: {attach_path.name}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to open Outlook draft with attachment: {exc}"
    finally:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass


def _render_output_actions(file_name: str, file_bytes: bytes, mime: str, key_prefix: str, dl_key_suffix: str, selected_format: str) -> None:
    col_dl, col_outlook = st.columns(2)

    with col_dl:
        st.download_button(
            "Download",
            data=file_bytes,
            file_name=file_name,
            mime=mime,
            width="stretch",
            key=f"{key_prefix}_{dl_key_suffix}",
        )

    with col_outlook:
        if st.button("Mail via Outlook (auto-attach)", key=f"{key_prefix}_{dl_key_suffix}_outlook", width="stretch"):
            ok, msg = _send_via_outlook_with_attachment(file_name, file_bytes, selected_format)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)


def _render_results(results: dict, key_prefix: str = "single") -> None:
    uploaded_name = results["uploaded_name"]
    inspection = results["inspection"]
    native_text_by_page = results["native_text_by_page"]
    ocr_text_by_page = results["ocr_text_by_page"]
    ocr_warning = results["ocr_warning"]
    duration = results["duration"]
    structured_data = results["structured_data"]
    api_payload = results["api_payload"]
    package = results["package"]

    st.success("Document processed successfully.")
    st.subheader("Processing Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF Type", inspection["pdf_type"])
    c2.metric("Total Pages", inspection["total_pages"])
    c3.metric("Native Text Pages", inspection["native_text_pages"])
    c4.metric("OCR Pages", len(inspection["pages_requiring_ocr"]))

    parser_meta = results.get("parser_metadata", {}) or {}
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Extracted Characters", inspection["extracted_characters"])
    c6.metric("OCR Engine", results.get("ocr_engine_used", "none"))
    c7.metric("Parser Engine", parser_meta.get("parser_engine", "legacy"))
    c8.metric("Duration (s)", duration)

    if ocr_warning:
        st.warning(ocr_warning)

    with st.expander("Validation Metadata", expanded=False):
        st.json(results["metadata"])

    with st.expander("Inspection Metadata", expanded=True):
        st.json(inspection)

    with st.expander("Native Extraction", expanded=True):
        _show_native_text_preview(native_text_by_page)

    with st.expander("OCR Extraction", expanded=False):
        if ocr_text_by_page:
            preview = "\n\n".join([f"[Page {p}]\n{text}" for p, text in ocr_text_by_page.items() if text.strip()])
            st.text_area("OCR Text Preview", value=preview[:6000], height=300)
        else:
            st.info("No OCR output generated.")

    with st.expander("OCR Engine Diagnostics", expanded=False):
        diag = results.get("ocr_diagnostics", {})
        if diag:
            st.json(diag)
        else:
            st.caption("No OCR diagnostics available for this run.")

    with st.expander("Parser Diagnostics", expanded=False):
        if parser_meta:
            st.json(parser_meta)
        else:
            st.caption("No parser diagnostics available for this run.")

    _render_ai_assistant_section(results, key_prefix=key_prefix)

    st.subheader("On-Screen Structured Output + Download")

    base = Path(uploaded_name).stem
    json_name = f"{base}.structured.json"
    csv_name = f"{base}.structured.csv"
    xlsx_name = f"{base}.structured.xlsx"
    docx_name = f"{base}.structured.docx"
    api_name = f"{base}.api_payload.json"
    zip_name = f"{base}.bundle.zip"

    selected_format = st.selectbox(
        "Select output format to preview",
        ["JSON", "API Payload JSON", "CSV", "Excel", "Word", "ZIP (All)"],
        index=0,
        key=f"{key_prefix}_output_format_selector",
    )

    if selected_format == "JSON":
        st.json(structured_data)
        _render_output_actions(
            file_name=json_name,
            file_bytes=package[json_name],
            mime="application/json",
            key_prefix=key_prefix,
            dl_key_suffix="dl_json",
            selected_format=selected_format,
        )
    elif selected_format == "API Payload JSON":
        st.json(api_payload)
        _render_output_actions(
            file_name=api_name,
            file_bytes=package[api_name],
            mime="application/json",
            key_prefix=key_prefix,
            dl_key_suffix="dl_api",
            selected_format=selected_format,
        )
    elif selected_format == "CSV":
        st.info("CSV structured preview")
        preview_tables = build_preview_tables(structured_data)
        preview_sheet = st.selectbox(
            "CSV Table Section",
            list(preview_tables.keys()),
            key=f"{key_prefix}_csv_preview_sheet",
        )
        st.dataframe(_table_for_streamlit(preview_tables[preview_sheet]), width="stretch")
        _render_output_actions(
            file_name=csv_name,
            file_bytes=package[csv_name],
            mime="text/csv",
            key_prefix=key_prefix,
            dl_key_suffix="dl_csv",
            selected_format=selected_format,
        )
    elif selected_format == "Excel":
        st.info("Excel structured preview")
        preview_tables = build_preview_tables(structured_data)
        preview_sheet = st.selectbox(
            "Excel Sheet Preview",
            list(preview_tables.keys()),
            key=f"{key_prefix}_excel_preview_sheet",
        )
        st.dataframe(_table_for_streamlit(preview_tables[preview_sheet]), width="stretch")
        _render_output_actions(
            file_name=xlsx_name,
            file_bytes=package[xlsx_name],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key_prefix=key_prefix,
            dl_key_suffix="dl_xlsx",
            selected_format=selected_format,
        )
    elif selected_format == "Word":
        st.info("Word output prepared. Use download button to save `.docx`.")
        _render_output_actions(
            file_name=docx_name,
            file_bytes=package[docx_name],
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key_prefix=key_prefix,
            dl_key_suffix="dl_docx",
            selected_format=selected_format,
        )
    elif selected_format == "ZIP (All)":
        st.info("ZIP contains JSON, CSV, Excel, Word, raw text, and API payload.")
        _render_output_actions(
            file_name=zip_name,
            file_bytes=package[zip_name],
            mime="application/zip",
            key_prefix=key_prefix,
            dl_key_suffix="dl_zip",
            selected_format=selected_format,
        )

    _render_publish_section(results, key_prefix=key_prefix)

    with st.expander("Quick metrics table", expanded=False):
        st.dataframe(
            [
                {"field": "file_name", "value": str(uploaded_name)},
                {"field": "pdf_type", "value": str(inspection["pdf_type"])},
                {"field": "total_pages", "value": str(inspection["total_pages"])},
                {"field": "native_text_pages", "value": str(inspection["native_text_pages"])},
                {"field": "ocr_pages", "value": str(len(inspection["pages_requiring_ocr"]))},
                {"field": "extracted_characters", "value": str(inspection["extracted_characters"])},
            ],
            width="stretch",
        )


def _render_publish_section(results: dict, key_prefix: str = "single") -> None:
    structured_data = results["structured_data"]
    document_type = structured_data.get("document_type")
    source_pdf = Path(results["uploaded_name"])
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if document_type == "resume":
        st.subheader("Resume Template")
        if st.button("Build Resume Template (mock)", key=f"{key_prefix}_btn_resume", width="stretch"):
            path = ResumeTemplatePublisher().publish(structured_data, output_dir, source_pdf)
            st.session_state[f"{key_prefix}_resume_payload_path"] = str(path)

        resume_key = f"{key_prefix}_resume_payload_path"
        if resume_key in st.session_state:
            payload_bytes = Path(st.session_state[resume_key]).read_bytes()
            st.success("Resume template payload generated.")
            st.json(json.loads(payload_bytes))
            st.download_button(
                "Download Resume Template",
                data=payload_bytes,
                file_name=Path(st.session_state[resume_key]).name,
                mime="application/json",
                width="stretch",
                key=f"{key_prefix}_dl_resume",
            )


def _render_ai_assistant_section(results: dict, key_prefix: str = "single") -> None:
    if results.get("mode") != "AI Enhanced":
        st.subheader("Chat/Summary with PDF")
        st.caption("Switch Mode to 'AI Enhanced' in the sidebar to use Chat/Summary.")
        return

    settings = get_settings()

    provider_options = ["Gemini", "OpenRouter"]
    default_idx = 0 if (settings.llm_provider or "").strip().lower() != "openrouter" else 1
    provider_label = st.selectbox(
        "Select AI provider for Chat/Summary",
        provider_options,
        index=default_idx,
        key=f"{key_prefix}_ai_provider_selector",
    )
    selected_provider = "openrouter" if provider_label == "OpenRouter" else "gemini"

    provider_title = "OpenRouter" if selected_provider == "openrouter" else "Google Gemini"
    model_label = settings.openrouter_model if selected_provider == "openrouter" else settings.gemini_model

    st.subheader(f"Chat/Summary with PDF ({provider_title})")
    assistant = DocumentAssistant(settings=settings, provider=selected_provider)
    available, msg = assistant.is_available()

    if not available:
        key_name = "OPENROUTER_API_KEY" if selected_provider == "openrouter" else "GEMINI_API_KEY"
        st.warning(f"Chat/Summary unavailable: {msg} Set {key_name} in your .env file.")
        return

    st.caption(f"Model: {model_label}")

    summary_key = f"{key_prefix}_{selected_provider}_ai_summary_text"
    chat_key = f"{key_prefix}_{selected_provider}_chat_history"

    if st.button("Generate AI Summary", key=f"{key_prefix}_{selected_provider}_btn_ai_summary"):
        with st.spinner(f"{provider_title} is generating summary..."):
            summary = assistant.summarize_document(
                native_text_by_page=results["native_text_by_page"],
                ocr_text_by_page=results["ocr_text_by_page"],
                structured_data=results["structured_data"],
            )
        if summary.get("error"):
            st.error(f"Summary failed: {summary['error']}")
        else:
            st.session_state[summary_key] = summary.get("answer") or "No summary returned."

    if summary_key in st.session_state:
        st.text_area(
            "AI Summary",
            value=st.session_state[summary_key],
            height=180,
            key=f"{key_prefix}_{selected_provider}_ai_summary_box",
        )

    st.markdown("**Chat with this document**")

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for entry in st.session_state[chat_key]:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])

    user_query = st.chat_input(
        f"Ask anything about this document ({provider_title})...",
        key=f"{key_prefix}_{selected_provider}_chat_input",
    )
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner(f"{provider_title} is answering..."):
                answer = assistant.answer_question(
                    question=user_query,
                    native_text_by_page=results["native_text_by_page"],
                    ocr_text_by_page=results["ocr_text_by_page"],
                    structured_data=results["structured_data"],
                )
            if answer.get("error"):
                answer_text = f"Error: {answer['error']}"
                st.error(answer_text)
            else:
                answer_text = answer.get("answer") or "No answer returned."
                st.write(answer_text)

        st.session_state[chat_key].append({"question": user_query, "answer": answer_text})


def _render_stored_uploads(store: SQLiteStore) -> str | None:
    st.sidebar.subheader("Stored Uploads (SQLite)")
    rows = store.list_uploads(limit=500)
    if not rows:
        st.sidebar.caption("No stored uploads yet.")
        return None

    # Show only records whose stored file is currently present on disk.
    rows = [r for r in rows if Path(str(r.get("stored_path", ""))).exists()]
    if not rows:
        st.sidebar.caption("No stored uploads found on disk.")
        return None

    # Deduplicate same file names in dropdown (keep most recent row first).
    deduped_by_name = {}
    for r in rows:
        name_key = str(r.get("original_name", "")).strip().lower()
        if name_key and name_key not in deduped_by_name:
            deduped_by_name[name_key] = r
    rows = list(deduped_by_name.values())

    search_query = st.sidebar.text_input("Search stored files", value="", key="stored_file_search").strip().lower()
    if search_query:
        rows = [
            r
            for r in rows
            if search_query in str(r.get("original_name", "")).lower()
            or search_query in str(r.get("status", "")).lower()
            or search_query in str(r.get("document_type", "")).lower()
            or search_query in str(r.get("id", "")).lower()
        ]

    if not rows:
        st.sidebar.caption("No files match your search.")
        return None

    id_to_name = {r["id"]: r["original_name"] for r in rows}
    selected_id = st.sidebar.selectbox(
        "Select stored file",
        options=list(id_to_name.keys()),
        format_func=lambda x: id_to_name.get(x, x),
        key="stored_file_selector",
    )

    row, data = store.get_file_bytes(selected_id)
    if not row:
        st.warning("Selected record not found.")
        return None

    st.sidebar.json(
        {
            "id": row["id"],
            "original_name": row["original_name"],
            "stored_path": row["stored_path"],
            "size_bytes": row["size_bytes"],
            "mime_type": row.get("mime_type", ""),
            "status": row.get("status", ""),
            "document_type": row.get("document_type"),
            "uploaded_at": row.get("uploaded_at", ""),
        }
    )

    if data is None:
        st.sidebar.error("Stored file path is missing on disk.")
        return None

    st.sidebar.download_button(
        "Download stored original file",
        data=data,
        file_name=row["original_name"],
        mime=row.get("mime_type") or "application/octet-stream",
        key="dl_stored_original",
        width="stretch",
    )

    return selected_id


def main() -> None:
    _init_page()
    _render_header()

    settings = get_settings()
    ui_cfg = _sidebar_settings(settings)
    _render_upload_limits(settings)
    store = SQLiteStore()
    selected_stored_id = _render_stored_uploads(store)

    uploaded_files = st.file_uploader(
        f"Upload files (PDF, DOCX, XLSX, XLS, JPG, JPEG, PNG) · Max {settings.max_upload_mb} MB per file",
        type=["pdf", "docx", "xlsx", "xls", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    left, right = st.columns([1, 1])
    process_clicked = left.button("Process Files", type="primary", width="stretch")
    reset_clicked = right.button("Reset Session", width="stretch")

    if reset_clicked:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if process_clicked:
        files_from_upload = uploaded_files or []

        # If user selected a stored file and also uploaded a file with same name,
        # avoid processing same logical document twice.
        if selected_stored_id and files_from_upload:
            stored_row_for_compare, _ = store.get_file_bytes(selected_stored_id)
            if stored_row_for_compare:
                stored_name_l = str(stored_row_for_compare.get("original_name", "")).strip().lower()
                files_from_upload = [f for f in files_from_upload if str(getattr(f, "name", "")).strip().lower() != stored_name_l]

        if not files_from_upload and not selected_stored_id:
            st.error("Upload files or select one from Stored Uploads in the sidebar before processing.")
            return

        if len(files_from_upload) > settings.max_batch_files:
            st.error(f"Maximum {settings.max_batch_files} uploaded files are allowed per run.")
            return

        processed_results = []
        for uploaded_file in files_from_upload:
            file_size = getattr(uploaded_file, "size", 0) or 0
            if file_size > settings.max_upload_bytes:
                st.error(
                    f"File '{uploaded_file.name}' exceeds max size of {settings.max_upload_mb} MB."
                )
                continue

            content_bytes = uploaded_file.getbuffer().tobytes()
            existing_row = store.find_existing_upload(
                content=content_bytes,
                original_name=uploaded_file.name,
            )

            if existing_row:
                stored_row = existing_row
                stored_row_path = Path(stored_row["stored_path"])
                if not stored_row_path.exists():
                    st.warning(f"[{uploaded_file.name}] Existing DB record found but file missing on disk. Re-saving file.")
                    stored_row = store.save_upload(
                        original_name=uploaded_file.name,
                        content=content_bytes,
                        mime_type=getattr(uploaded_file, "type", None),
                    )
                else:
                    st.info(f"[{uploaded_file.name}] Duplicate detected in DB. Reusing stored file.")
            else:
                stored_row = store.save_upload(
                    original_name=uploaded_file.name,
                    content=content_bytes,
                    mime_type=getattr(uploaded_file, "type", None),
                )

            temp_pdf_path = _save_bytes_to_temp(content_bytes, uploaded_file.name)
            try:
                run_cfg = dict(ui_cfg)
                run_cfg["parser_mode"] = "forced_hybrid" if ui_cfg.get("parser_mode") == "forced_hybrid" else "smart_hybrid"
                run_cfg["force_all_ocr_engines"] = bool(
                    run_cfg["parser_mode"] == "forced_hybrid" and ui_cfg.get("force_all_ocr_engines", False)
                )

                results = _process_document(
                    temp_pdf_path=temp_pdf_path,
                    uploaded_name=uploaded_file.name,
                    settings=settings,
                    ui_cfg=run_cfg,
                )
                processed_results.append(results)
                store.mark_processed(
                    file_id=stored_row["id"],
                    document_type=results.get("structured_data", {}).get("document_type"),
                    status="processed",
                )
            except PDFValidationError as exc:
                store.mark_processed(file_id=stored_row["id"], document_type=None, status="failed")
                st.error(f"[{uploaded_file.name}] Validation failed: {exc.message}")
                with st.expander(f"Technical details - {uploaded_file.name}"):
                    st.code(str(exc))
            except Exception as exc:  # noqa: BLE001
                store.mark_processed(file_id=stored_row["id"], document_type=None, status="failed")
                st.error(f"[{uploaded_file.name}] Unexpected processing failure.")
                if settings.app_debug:
                    with st.expander(f"Technical details - {uploaded_file.name}"):
                        st.exception(exc)
            finally:
                try:
                    if temp_pdf_path.exists():
                        temp_pdf_path.unlink(missing_ok=True)
                except Exception:
                    pass

        if selected_stored_id:
            row, data = store.get_file_bytes(selected_stored_id)
            if not row or data is None:
                st.error("Selected stored file is not available on disk.")
            else:
                temp_pdf_path = _save_bytes_to_temp(data, row["original_name"])
                try:
                    run_cfg = dict(ui_cfg)
                    run_cfg["parser_mode"] = "forced_hybrid" if ui_cfg.get("parser_mode") == "forced_hybrid" else "smart_hybrid"
                    run_cfg["force_all_ocr_engines"] = bool(
                        run_cfg["parser_mode"] == "forced_hybrid" and ui_cfg.get("force_all_ocr_engines", False)
                    )

                    results = _process_document(
                        temp_pdf_path=temp_pdf_path,
                        uploaded_name=row["original_name"],
                        settings=settings,
                        ui_cfg=run_cfg,
                    )
                    processed_results.append(results)
                    store.mark_processed(
                        file_id=row["id"],
                        document_type=results.get("structured_data", {}).get("document_type"),
                        status="processed",
                    )
                except PDFValidationError as exc:
                    store.mark_processed(file_id=row["id"], document_type=None, status="failed")
                    st.error(f"[{row['original_name']}] Validation failed: {exc.message}")
                    with st.expander(f"Technical details - {row['original_name']}"):
                        st.code(str(exc))
                except Exception as exc:  # noqa: BLE001
                    store.mark_processed(file_id=row["id"], document_type=None, status="failed")
                    st.error(f"[{row['original_name']}] Unexpected processing failure.")
                    if settings.app_debug:
                        with st.expander(f"Technical details - {row['original_name']}"):
                            st.exception(exc)
                finally:
                    try:
                        if temp_pdf_path.exists():
                            temp_pdf_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        if processed_results:
            st.session_state["batch_results"] = processed_results
            st.session_state["last_results"] = processed_results[0]
        else:
            st.session_state.pop("batch_results", None)
            st.session_state.pop("last_results", None)

    if "batch_results" in st.session_state:
        labels = [r["uploaded_name"] for r in st.session_state["batch_results"]]
        selected_name = st.selectbox("Select processed file to view", labels, key="batch_result_selector")
        selected_result = next(
            (r for r in st.session_state["batch_results"] if r["uploaded_name"] == selected_name),
            st.session_state["batch_results"][0],
        )
        safe_prefix = f"batch_{Path(selected_name).stem}".replace(" ", "_")
        _render_results(selected_result, key_prefix=safe_prefix)
    elif "last_results" in st.session_state:
        _render_results(st.session_state["last_results"], key_prefix="single")


if __name__ == "__main__":
    main()
