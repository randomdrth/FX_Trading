# FX Trade Signal Modeling (GBP/USD)

Group project for **Finance & Structuring for Data Science (IEOR 4571)** focused on translating ML predictions into an actionable FX trading decision rule.

## Project Goal
Build an end-to-end pipeline that **predicts next-day, high-confidence GBP/USD trade opportunities** and evaluates them in a way that resembles a real trading workflow (i.e., **only trade when the model is confident**).

Concretely, we define a profit-taking threshold (Δ = 0.5%) and predict whether the **next day’s intraday high** reaches that target. We then compare:
- Standard classification performance (threshold = 0.5)
- A trading-style policy: **trade only when predicted probability > p** (e.g., p = 0.6)

## Data
- Daily **GBP/USD OHLC** pulled from **Yahoo Finance** via `yfinance`
- Time-based splits to avoid leakage (train/validation/test separated chronologically)

## Methodology (What we built)
### 1) Labeling aligned to a trading rule
- Define a “profitable move” label using **next-day** price action (profit-taking event)
- Shift labels so features at time *t* map to outcome at *t+1* (leak-free supervision)

### 2) Feature engineering (technical + volatility)
We engineered a compact feature set designed to represent short-horizon regime and momentum:
- Returns / log-returns
- Price ranges and normalized OHLC relationships
- Rolling mean/std/volatility measures
- Indicators such as **ATR**, **RSI**, and momentum signals

### 3) Deep sequence modeling (why multiple architectures)
FX is time-dependent and non-stationary, so we tested several sequence models to learn short-term temporal structure and compare bias/variance tradeoffs:
- Baseline **LSTM**
- **CNN-LSTM** (local pattern extraction + temporal memory)
- **CNN with Attention**
- **GRU with Attention** (lighter recurrence + attention-based focus)

We used standardized preprocessing, rolling-window sequence construction, and careful train/validation monitoring to avoid overfitting.

## Evaluation
We evaluate performance in two ways:
1. **Classification metrics** (AUC, accuracy, precision, recall)
2. **Trading-style decision policy**: trade only when the model is confident (threshold gating),
   which directly reflects practical usage: fewer trades, potentially higher “trade quality.”

## Key Findings (High-level)
- Baseline sequence models (LSTM / CNN-LSTM) achieved **modest signal** (AUC ≈ 0.63), reflecting the difficulty of forecasting short-horizon FX moves.
- Adding **attention** improved performance and interpretability:
  - **CNN + Attention** improved over baseline.
  - **GRU + Attention** was the strongest overall performer (**best test AUC ≈ 0.668**).
- Confidence-thresholding provides a practical control knob:
  - Higher thresholds reduce trading frequency but can improve the precision of executed trades.

## Repository Contents
- `FinDS_Final_Project_Part_2_Final.ipynb` — end-to-end pipeline:
  - data pull (yfinance) → labeling → feature engineering → scaling
  - sequence construction
  - model training/evaluation across multiple deep architectures
- `IEOR 4571 Final Project_ Team 8 (GBP_USD).pdf` — full write-up, methodology, and results

## How to Run
1. Create an environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows
