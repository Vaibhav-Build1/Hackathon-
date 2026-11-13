from __future__ import annotations

import json

import requests


class OpenRouterError(Exception):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model.strip()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict) -> dict:
        try:
            resp = requests.post(
                url=self.base_url,
                headers=self._headers(),
                data=json.dumps(payload),
                timeout=60,
            )
        except requests.RequestException as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise OpenRouterError(f"OpenRouter HTTP {resp.status_code}: {resp.text}")

        try:
            return resp.json()
        except ValueError as exc:
            raise OpenRouterError("OpenRouter returned non-JSON response.") from exc

    def complete_with_reasoning(self, prompt: str) -> str:
        if not self.is_configured():
            raise OpenRouterError("OpenRouter client is not configured.")

        first_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"enabled": True},
        }
        first = self._post(first_payload)
        message1 = ((first.get("choices") or [{}])[0]).get("message") or {}
        content1 = message1.get("content", "")
        reasoning_details = message1.get("reasoning_details")

        second_messages = [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": content1,
                "reasoning_details": reasoning_details,
            },
            {"role": "user", "content": "Please provide the final concise answer."},
        ]

        second_payload = {
            "model": self.model,
            "messages": second_messages,
            "reasoning": {"enabled": True},
        }
        second = self._post(second_payload)
        message2 = ((second.get("choices") or [{}])[0]).get("message") or {}
        content2 = (message2.get("content") or "").strip()

        if not content2 and content1:
            return str(content1).strip()
        return content2
