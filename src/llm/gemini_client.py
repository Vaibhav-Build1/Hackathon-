from __future__ import annotations

import google.generativeai as genai


class GeminiError(Exception):
    pass


class GeminiClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self._configured = False
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self._configured = True

    def is_configured(self) -> bool:
        return self._configured

    def complete(self, prompt: str, model: str | None = None) -> str:
        if not self._configured:
            raise GeminiError("Gemini API key is not configured (GEMINI_API_KEY).")

        try:
            m = genai.GenerativeModel(model or self.model)
            response = m.generate_content(prompt)
            return (getattr(response, "text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            raise GeminiError(f"Gemini request failed: {exc}") from exc
