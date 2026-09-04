"""
LSTM Trading Strategy — Phase 1
Data pull + 3-class labeling (up / neutral / down).

This reuses the same data-pulling logic as the previous project, but the
label is different: instead of binary up/down, small moves get their own
"neutral" class, since forcing a tiny +0.05% day into "up" is misleading --
it's not a meaningful move worth trading on.

Run locally: pip install yfinance pandas
cd /Users/yinying/Documents/Coding/stock_deeplearning
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd

TICKER = "NVDA"
START = "2019-01-01"
END = "2026-07-01"

NEUTRAL_BAND = 0.005  # +/- 0.5% next-day return counts as "neutral" -- adjust if needed


def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_label(df, neutral_band=NEUTRAL_BAND):
    df = df.copy()
    next_return = df["Close"].pct_change().shift(-1)  # tomorrow's return, known only in hindsight

    # 0 = down, 1 = neutral, 2 = up -- numeric labels, needed for the LSTM's output layer later
    df["label"] = 1  # default: neutral
    df.loc[next_return > neutral_band, "label"] = 2   # up
    df.loc[next_return < -neutral_band, "label"] = 0  # down

    df["next_return"] = next_return  # kept for later use in the backtest phase
    df = df.dropna(subset=["next_return"])  # last row has no "tomorrow" -- drop it
    return df


if __name__ == "__main__":
    raw = load_data(TICKER, START, END)
    labeled = add_label(raw)

    print(f"Pulled {len(labeled)} labeled rows for {TICKER}")
    print(f"\nLabel meaning: 0=down, 1=neutral, 2=up (neutral band = +/-{NEUTRAL_BAND*100:.1f}%)")
    print(f"\nClass balance:\n{labeled['label'].value_counts(normalize=True).sort_index()}")
    print(f"\n{labeled[['Close', 'next_return', 'label']].tail(10)}")
