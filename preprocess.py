import re
from typing import List

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lightweight preprocessing that preserves sentiment-bearing words."""
    text = str(text or "").strip()
    text = URL_RE.sub(" URL ", text)
    text = USER_RE.sub(" USER ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split review text into reasonably sized clauses for aspect attribution."""
    clean = normalize_text(text)
    if not clean:
        return []
    # First split on sentence punctuation, then on contrast/conjunction boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    clauses: List[str] = []
    for sentence in sentences:
        pieces = re.split(r"\s+(?:but|however|although|though|while|yet|and)\s+", sentence, flags=re.I)
        clauses.extend(p.strip(" ,;:") for p in pieces if p.strip(" ,;:"))
    return clauses
