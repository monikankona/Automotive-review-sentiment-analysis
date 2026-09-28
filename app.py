from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzer import analyze_review
from src.analytics import analyze_dataframe, summary_table
from src.sentiment_engine import SentimentEngine, PRIMARY_MODEL, ALTERNATE_MODEL

st.set_page_config(page_title="Automotive Review Analytics", page_icon="A", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: Arial, sans-serif; }
.block-container { max-width: 1200px; padding-top: 2rem; }
.metric-card { border: 1px solid #cfcfcf; padding: 16px; background: #fafafa; }
.small { color: #666; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("AI-Based Automotive Review and Customer Sentiment Analytics")
st.caption("Module 3 - Connectivity | BERT / XLM-R / NLP")

with st.sidebar:
    st.subheader("Model")
    choice = st.selectbox("Sentiment model", ["XLM-R (primary)", "BERT multilingual (alternate)", "Offline demo mode"])
    if choice == "XLM-R (primary)":
        model_name = PRIMARY_MODEL
        allow_fallback = True
    elif choice == "BERT multilingual (alternate)":
        model_name = ALTERNATE_MODEL
        allow_fallback = True
    else:
        model_name = "offline"
        allow_fallback = True
    st.markdown("Model downloads occur on first model-backed run.")

engine = SentimentEngine(model_name=model_name if model_name != "offline" else None, allow_fallback=allow_fallback, force_offline=(choice == "Offline demo mode"))

st.markdown("### 1. Single review analysis")
default_text = "The battery range is excellent, but the charging time is too long. The seats are comfortable and the price feels reasonable."
review = st.text_area("Enter an automotive review", value=default_text, height=140)
if st.button("Analyze Review", type="primary"):
    result = analyze_review(review, engine)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall sentiment", result["overall_sentiment"])
    c2.metric("Confidence", f"{result['overall_confidence'] * 100:.1f}%")
    c3.metric("Aspects detected", len(result["aspects"]))
    if result["aspects"]:
        st.dataframe(pd.DataFrame(result["aspects"])[["aspect", "sentiment", "confidence", "context"]], use_container_width=True, hide_index=True)
    else:
        st.info("No supported automotive aspect was detected; overall sentiment is still available.")

st.divider()
st.markdown("### 2. Batch review analytics")
example_path = ROOT / "data" / "sample_reviews.csv"
upload = st.file_uploader("Upload a CSV containing a review column", type=["csv"])
if upload is not None:
    df = pd.read_csv(upload)
else:
    df = pd.read_csv(example_path)

review_column = st.selectbox("Review column", list(df.columns), index=list(df.columns).index("review") if "review" in df.columns else 0)
if st.button("Analyze Dataset"):
    results = analyze_dataframe(df, engine, review_column)
    st.success(f"Analyzed {len(df)} reviews and generated {len(results)} aspect rows.")
    st.dataframe(results, use_container_width=True, hide_index=True)
    summary = summary_table(results)
    st.markdown("#### Aspect-level summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download detailed analysis CSV", csv, file_name="automotive_sentiment_results.csv", mime="text/csv")
