from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

try:
    from transformers import pipeline
except Exception:  # optional until first model-backed call
    pipeline = None


PRIMARY_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
ALTERNATE_MODEL = os.getenv("ALTERNATE_SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")

POSITIVE_WORDS = {
    "excellent", "great", "good", "smooth", "comfortable", "safe", "amazing", "fast",
    "reliable", "quiet", "premium", "love", "lovely", "affordable", "efficient", "impressive",
    "responsive", "strong", "easy", "satisfied", "perfect", "best", "fantastic"
}
NEGATIVE_WORDS = {
    "poor", "bad", "slow", "uncomfortable", "unsafe", "expensive", "noisy", "weak", "worst",
    "disappointing", "disappointed", "problem", "problems", "issue", "issues", "long", "hard",
    "difficult", "rough", "high", "low", "failure", "terrible", "awful", "overpriced"
}


@dataclass
class SentimentResult:
    label: str
    score: float
    backend: str
    model: str


class SentimentEngine:
    """Model-backed sentiment with a deterministic offline fallback."""

    def __init__(self, model_name: Optional[str] = None, allow_fallback: bool = True, force_offline: bool = False):
        self.model_name = PRIMARY_MODEL if model_name is None else model_name
        self.allow_fallback = allow_fallback
        self.force_offline = force_offline
        self._pipe = None
        self.backend = "offline-fallback"

    def _load(self) -> None:
        if self.force_offline or self._pipe is not None or pipeline is None:
            return
        try:
            self._pipe = pipeline("sentiment-analysis", model=self.model_name, tokenizer=self.model_name)
            self.backend = "transformers"
        except Exception:
            if self.model_name != ALTERNATE_MODEL:
                try:
                    self._pipe = pipeline("text-classification", model=ALTERNATE_MODEL, tokenizer=ALTERNATE_MODEL)
                    self.model_name = ALTERNATE_MODEL
                    self.backend = "transformers"
                    return
                except Exception:
                    pass
            if not self.allow_fallback:
                raise

    @staticmethod
    def _fallback(text: str) -> SentimentResult:
        words = [w.lower().strip(".,!?;:()[]{}\"'") for w in text.split()]
        pos = sum(w in POSITIVE_WORDS for w in words)
        neg = sum(w in NEGATIVE_WORDS for w in words)
        total = max(1, pos + neg)
        margin = (pos - neg) / total
        if margin > 0.18:
            label = "Positive"
        elif margin < -0.18:
            label = "Negative"
        else:
            label = "Neutral"
        score = min(0.99, 0.50 + abs(margin) * 0.45)
        return SentimentResult(label, round(float(score), 4), "offline-fallback", "lexicon")

    @staticmethod
    def _map_label(raw_label: str, raw_score: float) -> str:
        label = str(raw_label).lower()
        if "positive" in label:
            return "Positive"
        if "negative" in label:
            return "Negative"
        if "neutral" in label:
            return "Neutral"
        # BERT product-review model returns 1-5 stars.
        digits = "".join(ch for ch in label if ch.isdigit())
        if digits:
            stars = int(digits[0])
            if stars <= 2:
                return "Negative"
            if stars == 3:
                return "Neutral"
            return "Positive"
        # Safe fallback when a third-party model uses LABEL_n.
        return "Positive" if raw_score >= 0.66 else "Neutral"

    def predict(self, text: str) -> SentimentResult:
        clean = str(text or "").strip()
        if not clean:
            return SentimentResult("Neutral", 0.0, self.backend, self.model_name)
        self._load()
        if self._pipe is None:
            return self._fallback(clean)
        try:
            result = self._pipe(clean, truncation=True, max_length=256)[0]
            raw_label = result.get("label", "Neutral")
            score = float(result.get("score", 0.0))
            return SentimentResult(self._map_label(raw_label, score), round(score, 4), self.backend, self.model_name)
        except Exception:
            if self.allow_fallback:
                return self._fallback(clean)
            raise

    def predict_many(self, texts: Iterable[str]) -> List[SentimentResult]:
        return [self.predict(t) for t in texts]
