# AI-Based Automotive Review and Customer Sentiment Analytics

Module 3 (Connectivity) - Project option 3.

## What the project does
- Accepts individual automotive reviews or a CSV dataset.
- Performs lightweight text normalization.
- Detects the automotive aspects named in the Module 3 brief: Vehicle, Engine, Battery, Mileage, Safety, Comfort, Service, Infotainment and Price.
- Runs sentiment classification using a Hugging Face Transformer model. The default is `cardiffnlp/twitter-xlm-roberta-base-sentiment`; an alternate multilingual BERT product-review model is also supported.
- Produces overall and aspect-level sentiment, confidence, context snippets and CSV results.
- Provides a Streamlit dashboard for interactive analysis.
- Includes an offline lexicon fallback so the application can still demonstrate the end-to-end pipeline when model weights are unavailable.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

The first model-backed analysis may download model weights from Hugging Face. For a deterministic offline demonstration, select **Offline demo mode** in the sidebar.

## Batch CLI

```bash
python -m src.cli --input data/sample_reviews.csv --output outputs/automotive_sentiment_results.csv
```

## Design notes
The aspect-level stage uses local clause/context attribution. It is intentionally transparent and lightweight: aspect extraction is vocabulary-driven, while the sentiment decision is Transformer-based when available. This makes the project suitable for an academic prototype while leaving a clear path to a trained ABSA model in future work.

## Evidence and limitations
Do not interpret the included sample CSV as a public benchmark dataset. It is a curated demonstration dataset for validating the software pipeline. No accuracy number is claimed for this local sample because it does not contain independently verified labels.
