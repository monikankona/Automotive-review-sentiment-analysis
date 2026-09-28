from src.aspects import find_aspects, aspect_context
from src.preprocess import normalize_text, split_sentences
from src.sentiment_engine import SentimentEngine
from src.analyzer import analyze_review


def test_preprocess():
    assert normalize_text("  Great car!   https://example.com ") == "Great car! URL"
    clauses = split_sentences("Battery range is excellent, but charging time is long.")
    assert len(clauses) == 2


def test_aspect_detection():
    found = [a for a, _ in find_aspects("The battery range is excellent and the seats are comfortable.")]
    assert "Battery" in found
    assert "Comfort" in found


def test_offline_analysis():
    engine = SentimentEngine(model_name="", allow_fallback=True)
    result = analyze_review("The battery range is excellent but charging is slow.", engine)
    aspects = {r["aspect"] for r in result["aspects"]}
    assert "Battery" in aspects

if __name__ == "__main__":
    test_preprocess(); test_aspect_detection(); test_offline_analysis(); print("All smoke tests passed.")
