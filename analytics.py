from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from .analyzer import analyze_review
from .sentiment_engine import SentimentEngine


def analyze_dataframe(df: pd.DataFrame, engine: SentimentEngine, review_column: str = "review") -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for idx, review in enumerate(df[review_column].fillna("")):
        result = analyze_review(str(review), engine)
        base = {"review_id": idx + 1, "review": str(review), "overall_sentiment": result["overall_sentiment"], "overall_confidence": result["overall_confidence"]}
        if result["aspects"]:
            for aspect in result["aspects"]:
                records.append({**base, "aspect": aspect["aspect"], "aspect_sentiment": aspect["sentiment"], "aspect_confidence": aspect["confidence"], "context": aspect["context"]})
        else:
            records.append({**base, "aspect": "No detected aspect", "aspect_sentiment": result["overall_sentiment"], "aspect_confidence": result["overall_confidence"], "context": str(review)})
    return pd.DataFrame(records)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["aspect", "aspect_sentiment"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
        .sort_values(["aspect", "review_count"], ascending=[True, False])
    )
