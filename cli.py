import argparse
from pathlib import Path

import pandas as pd

from .analytics import analyze_dataframe
from .sentiment_engine import SentimentEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze automotive review sentiment.")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--review-column", default="review", help="Column containing the review text")
    parser.add_argument("--offline", action="store_true", help="Use deterministic fallback sentiment")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.review_column not in df.columns:
        raise SystemExit(f"Review column '{args.review_column}' not found. Available: {list(df.columns)}")

    engine = SentimentEngine(model_name=None, allow_fallback=True, force_offline=args.offline)
    results = analyze_dataframe(df, engine, args.review_column)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Wrote {len(results)} result rows to {args.output}")


if __name__ == "__main__":
    main()
