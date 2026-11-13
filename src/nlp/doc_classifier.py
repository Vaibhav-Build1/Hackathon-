from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from intelligence_plugin.classifiers import RULES as LEGACY_RULES
from src.nlp.text_normalizer import normalize_text, sample_for_semantics, tokenize_lines

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None  # type: ignore


DOC_TYPES = ["invoice_receipt", "resume", "generic"]


PROTOTYPES = {
    "invoice_receipt": (
        "Invoice receipt billing document with vendor, invoice number, date, subtotal, tax, total amount due, bill to details."
    ),
    "resume": (
        "Resume curriculum vitae profile with candidate name, experience, education, skills, projects, contact details, linkedin."
    ),
    "generic": "General document with mixed content and no strongly identifiable invoice or resume pattern.",
}


@dataclass
class ClassificationResult:
    document_type: str
    confidence: float
    score_breakdown: dict[str, dict[str, float]]


def _rule_scores(text: str) -> dict[str, float]:
    lower = (text or "").lower()
    out = {k: 0.0 for k in DOC_TYPES}
    for doc_type, pats in LEGACY_RULES.items():
        score = 0.0
        for p in pats:
            score += len(re.findall(p, lower, flags=re.IGNORECASE))
        out[doc_type] = score
    out["generic"] = max(0.0, 1.0 if out["invoice_receipt"] == 0 and out["resume"] == 0 else 0.0)
    return out


def _entity_hint_scores(text: str) -> dict[str, float]:
    lower = (text or "").lower()
    invoice_hints = [
        r"invoice\s*(?:no|number|#)",
        r"total\s*(?:due|amount)?",
        r"tax",
        r"bill\s*to",
        r"subtotal",
        r"\$[0-9]",
    ]
    resume_hints = [
        r"experience",
        r"education",
        r"skills",
        r"linkedin\.com",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"\+?\d[\d\s().\-]{7,}\d",
    ]
    inv = sum(len(re.findall(p, lower, flags=re.IGNORECASE)) for p in invoice_hints)
    res = sum(len(re.findall(p, lower, flags=re.IGNORECASE)) for p in resume_hints)
    return {"invoice_receipt": float(inv), "resume": float(res), "generic": 0.5 if (inv + res) == 0 else 0.0}


@lru_cache(maxsize=1)
def _get_embedder():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _semantic_scores(text: str) -> dict[str, float]:
    model = _get_embedder()
    if model is None:
        return {k: 0.0 for k in DOC_TYPES}

    sample = sample_for_semantics(tokenize_lines(text), max_chars=3500)
    if not sample:
        return {k: 0.0 for k in DOC_TYPES}

    try:
        emb_doc = model.encode([sample], normalize_embeddings=True)[0]
        proto_texts = [PROTOTYPES[k] for k in DOC_TYPES]
        emb_proto = model.encode(proto_texts, normalize_embeddings=True)
        sims = [float((emb_doc * emb_proto[i]).sum()) for i in range(len(DOC_TYPES))]
        return {DOC_TYPES[i]: max(0.0, sims[i]) for i in range(len(DOC_TYPES))}
    except Exception:
        return {k: 0.0 for k in DOC_TYPES}


def _softmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    mx = max(values.values())
    exps = {k: math.exp(v - mx) for k, v in values.items()}
    total = sum(exps.values()) or 1.0
    return {k: exps[k] / total for k in values}


def classify_document_ensemble(text: str) -> ClassificationResult:
    norm = normalize_text(text)

    rules = _rule_scores(norm)
    entity = _entity_hint_scores(norm)
    semantic = _semantic_scores(norm)

    # Weighted blend: rules (precision), semantic (recall), entity hints (stability).
    raw = {}
    for k in DOC_TYPES:
        raw[k] = 0.45 * rules.get(k, 0.0) + 0.35 * semantic.get(k, 0.0) + 0.20 * entity.get(k, 0.0)

    probs = _softmax(raw)
    winner = max(probs, key=probs.get) if probs else "generic"
    confidence = float(probs.get(winner, 0.0))

    # Conservative fallback to generic on weak confidence.
    if confidence < 0.42:
        winner = "generic"

    return ClassificationResult(
        document_type=winner,
        confidence=confidence,
        score_breakdown={
            "rules": rules,
            "semantic": semantic,
            "entity_hints": entity,
            "combined_raw": raw,
            "combined_prob": probs,
        },
    )
