from __future__ import annotations

from src.config.settings import AppSettings
from src.llm.gemini_client import GeminiClient, GeminiError
from src.llm.openrouter_client import OpenRouterClient, OpenRouterError


MAX_CONTEXT_CHARS = 12000


def _build_context(native_text_by_page: dict, ocr_text_by_page: dict, structured_data: dict) -> str:
    pages_text = []
    for page in sorted(set(list(native_text_by_page.keys()) + list(ocr_text_by_page.keys()))):
        text = (native_text_by_page.get(page) or "") + "\n" + (ocr_text_by_page.get(page) or "")
        text = text.strip()
        if text:
            pages_text.append(f"[Page {page}]\n{text}")

    combined = "\n\n".join(pages_text)

    fields_summary = ""
    fields = structured_data.get("fields") or {}
    if fields:
        fields_summary = "\n\nExtracted fields:\n" + "\n".join(f"- {k}: {v}" for k, v in fields.items())

    context = f"{combined}{fields_summary}".strip()
    return context[:MAX_CONTEXT_CHARS]


class DocumentAssistant:
    """Uses Google Gemini API to provide:
    - An AI-generated document summary
    - Question answering grounded in the extracted document text

    This is purely an AI Enhanced Mode feature. It is never called in
    Local Mode and requires GEMINI_API_KEY to be configured.
    """

    def __init__(self, settings: AppSettings, provider: str | None = None) -> None:
        self.settings = settings
        self._provider_override = (provider or "").strip().lower() if provider else None
        self.gemini_client = GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
        self.openrouter_client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
        )

    def _active_provider(self) -> str:
        if self._provider_override in {"openrouter", "gemini"}:
            return self._provider_override

        provider = (self.settings.llm_provider or "").strip().lower()
        if provider in {"openrouter", "gemini"}:
            return provider
        return "gemini"

    def is_available(self) -> tuple[bool, str]:
        provider = self._active_provider()
        if provider == "openrouter":
            if not self.openrouter_client.is_configured():
                return (
                    False,
                    "OpenRouter API key/model not configured "
                    "(OPENROUTER_API_KEY / OPENROUTER_MODEL).",
                )
            return True, "OpenRouter model is configured."

        if not self.gemini_client.is_configured():
            return False, "Gemini API key is not configured (GEMINI_API_KEY)."
        return True, "Gemini model is configured."

    def summarize_document(
        self,
        native_text_by_page: dict,
        ocr_text_by_page: dict,
        structured_data: dict,
    ) -> dict:
        context = _build_context(native_text_by_page, ocr_text_by_page, structured_data)
        if not context:
            return {"answer": None, "error": "No extracted text available to summarize."}

        question = (
            "Summarize the following document in 4-6 concise bullet points. "
            "Mention the document type, key parties, key dates, and key amounts "
            "if present. Do not invent information that is not in the text.\n\n"
            f"DOCUMENT TEXT:\n{context}"
        )

        provider = self._active_provider()
        try:
            if provider == "openrouter":
                answer = self.openrouter_client.complete_with_reasoning(question)
            else:
                answer = self.gemini_client.complete(
                    prompt=question,
                    model=self.settings.gemini_model,
                )
            return {"answer": answer, "error": None}
        except (GeminiError, OpenRouterError) as exc:
            return {"answer": None, "error": str(exc)}

    def answer_question(
        self,
        question: str,
        native_text_by_page: dict,
        ocr_text_by_page: dict,
        structured_data: dict,
    ) -> dict:
        context = _build_context(native_text_by_page, ocr_text_by_page, structured_data)
        if not context:
            return {"answer": None, "error": "No extracted text available to answer from."}

        prompt = (
            "Answer the question using ONLY the document text below. "
            "If the answer is not present in the text, say so clearly instead "
            "of guessing.\n\n"
            f"DOCUMENT TEXT:\n{context}\n\n"
            f"QUESTION: {question}"
        )

        provider = self._active_provider()
        try:
            if provider == "openrouter":
                answer = self.openrouter_client.complete_with_reasoning(prompt)
            else:
                answer = self.gemini_client.complete(
                    prompt=prompt,
                    model=self.settings.gemini_model,
                )
            return {"answer": answer, "error": None}
        except (GeminiError, OpenRouterError) as exc:
            return {"answer": None, "error": str(exc)}
