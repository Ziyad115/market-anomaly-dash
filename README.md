# Market Anomaly & Crisis Detector (Dash Version)

A live financial dashboard that detects statistical anomalies across a basket of
major assets using a causal, leakage-free rolling z-score model, with an
Isolation Forest model included for comparison. It ships with two market modes
you can switch between live:

- **🌐 Global** — S&P 500, Gold, Oil (WTI), USD Index, VIX
- **🇸🇦 Saudi** — TASI (Tadawul All Share Index), Brent, Gold

## Features
- **Dual market modes** with a live toggle in the sidebar — switch between the
  Global and Saudi universes without a reload; an active-market badge shows
  which one you're viewing
- Composite anomaly score (RMS of cross-asset z-scores) with per-asset
  contribution breakdown
- Causal expanding-window threshold (no future data leakage)
- Per-asset z-scores computed on each market's **own native trading calendar**
  (e.g. Saudi trades Sun–Thu vs Brent/Gold Mon–Fri), so a closed market never
  distorts another's score; the composite is anchored to the primary asset's
  calendar
- Historical crisis backtest/validation (2008, 2020, 2022, etc.)
- Live news headlines pulled per flagged anomaly day via Google News RSS
- Isolation Forest model comparison
- Correlation-aware rarity via Mahalanobis distance
- Bilingual UI (English / العربية) with instant switching

## Tech Stack
- Python, Dash, Plotly, Pandas, scikit-learn, yfinance

## Live Demo
[https://market-anomaly-dash.onrender.com]

## Running Locally
```
pip install -r requirements.txt
python app.py
```

## Data Sources
- Market data: Yahoo Finance (via yfinance), with automatic fallbacks:
  - FRED for VIX, WTI and Brent when Yahoo is unavailable
  - `KSA` (iShares MSCI Saudi Arabia ETF) as a fallback proxy for TASI, and
    `BNO` for Brent
  - a 24-hour on-disk cache of raw prices
- News: Google News RSS (free, no API key required)

## Health / Diagnostics
`GET /health` returns JSON describing the current load state — which modes
loaded, the per-ticker data source (yfinance / FRED / cache) or failure, the
active data source, and any load/compute error. Useful for debugging a
deployment.
