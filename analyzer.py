from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .aspects import aspect_context, find_aspects
from .preprocess import normalize_text, split_sentences
from .sentiment_engine import SentimentEngine


def analyze_review(review: str, engine: SentimentEngine) -> Dict[str, Any]:
    clean = normalize_text(review)
    overall = engine.predict(clean)
    aspect_rows: List[Dict[str, Any]] = []

    for clause in split_sentences(clean):
        for aspect, keyword in find_aspects(clause):
            context = aspect_context(clause, keyword)
            result = engine.predict(context)
            aspect_rows.append({
                "aspect": aspect,
                "keyword": keyword,
                "context": context,
                "sentiment": result.label,
                "confidence": result.score,
                "backend": result.backend,
                "model": result.model,
            })

    # Avoid duplicate aspect rows caused by repeated keywords in one review.
    unique = {}
    for row in aspect_rows:
        key = (row["aspect"], row["context"].lower())
        unique[key] = row
    aspect_rows = list(unique.values())

    return {
        "review": clean,
        "overall_sentiment": overall.label,
        "overall_confidence": overall.score,
        "backend": overall.backend,
        "model": overall.model,
        "aspects": aspect_rows,
    }


def flatten_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = result["aspects"]
    if not rows:
        return [{
            "aspect": "No detected aspect",
            "sentiment": result["overall_sentiment"],
            "confidence": result["overall_confidence"],
            "context": result["review"],
        }]
    return [
        {
            "aspect": row["aspect"],
            "sentiment": row["sentiment"],
            "confidence": row["confidence"],
            "context": row["context"],
        }
        for row in rows
    ]
