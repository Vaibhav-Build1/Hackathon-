from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_SETTINGS_FILE = CONFIG_DIR / "settings.yaml"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    max_upload_mb: int = Field(default=20, alias="MAX_UPLOAD_MB")
    max_pdf_pages: int = Field(default=100, alias="MAX_PDF_PAGES")
    max_batch_files: int = Field(default=10, alias="MAX_BATCH_FILES")
    data_retention_mode: str = Field(default="session", alias="DATA_RETENTION_MODE")

    tesseract_cmd: str | None = Field(default=None, alias="TESSERACT_CMD")
    ocr_dpi: int = Field(default=300, alias="OCR_DPI")
    ocr_language: str = Field(default="eng", alias="OCR_LANGUAGE")

    llm_provider: str = Field(default="disabled", alias="LLM_PROVIDER")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")

    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str | None = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str | None = Field(default=None, alias="AZURE_OPENAI_API_VERSION")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str | None = Field(default=None, alias="OLLAMA_MODEL")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    gemini_vision_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_VISION_MODEL")

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b:free",
        alias="OPENROUTER_MODEL",
    )

    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    vector_store: str = Field(default="faiss", alias="VECTOR_STORE")
    rag_chunk_size: int = Field(default=1000, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=150, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")

    classification_auto_accept_threshold: float = Field(default=0.80, alias="CLASSIFICATION_AUTO_ACCEPT_THRESHOLD")
    classification_review_threshold: float = Field(default=0.55, alias="CLASSIFICATION_REVIEW_THRESHOLD")

    enable_docling: bool = Field(default=True, alias="ENABLE_DOCLING")
    parser_mode: str = Field(default="smart_hybrid", alias="PARSER_MODE")
    force_all_ocr_engines: bool = Field(default=False, alias="FORCE_ALL_OCR_ENGINES")

    connector_dry_run: bool = Field(default=True, alias="CONNECTOR_DRY_RUN")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _safe_yaml_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    load_dotenv(override=False)

    yaml_config = _safe_yaml_load(DEFAULT_SETTINGS_FILE)
    env_values = yaml_config.get("env", {}) if isinstance(yaml_config.get("env"), dict) else {}

    return AppSettings(**env_values)
