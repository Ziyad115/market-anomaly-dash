# Market Anomaly & Crisis Detector (Dash Version)

A live financial dashboard that detects statistical anomalies across five major asset classes (S&P 500, Gold, Oil, USD Index, VIX) using a causal, leakage-free rolling z-score model, with an Isolation Forest model included for comparison.

## Features
- Composite anomaly score (RMS of cross-asset z-scores) with per-asset contribution breakdown
- Causal expanding-window threshold (no future data leakage)
- Historical crisis backtest/validation (2008, 2020, 2022, etc.)
- Live news headlines pulled per flagged anomaly day via Google News RSS
- Isolation Forest model comparison

## Tech Stack
- Python, Dash, Plotly, Pandas, scikit-learn, yfinance

## Live Demo
[https://market-anomaly-dash.onrender.com]

## Running Locally
\`\`\`
pip install -r requirements.txt
python app.py
\`\`\`

## Data Sources
- Market data: Yahoo Finance (via yfinance)
- News: Google News RSS (free, no API key required)
