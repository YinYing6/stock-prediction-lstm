> **Part 2 of 3** in a series testing whether ML can predict stock direction. [← Part 1: Classical ML](../stock-prediction-ml/README.md) | [Part 3: Real Backtest →](../quant-trading-backtest/README.md)

# Can Deep Learning (LSTM) Predict Stock Direction? (Also No)

Follow-up to my [stock_prediction](https://github.com/YinYing6/stock-prediction-ml) project. That one
tested whether simple ML (Logistic Regression, Random Forest) could predict
next-day stock direction using technical indicators — it couldn't. This
project asks: does a more powerful model (LSTM, a type of neural network
built for sequences) do any better?

## TL;DR

**No.** Tested on NVDA with proper walk-forward validation (6 time windows,
same rigor as the last project), the LSTM:
- Beat a naive "always guess the most common outcome" baseline in only
  **1 out of 6** windows
- Averaged **worse** accuracy than the naive baseline overall
- Repeatedly **collapsed** — a common deep learning failure where the model
  gives up trying to learn and just predicts one class over and over,
  because it can't find real signal to learn from

This actually strengthens the conclusion from the first project: it's not
that simple models were "too weak" — a more complex model doesn't fare
better on this task either, and comes with its own new failure modes on
top.

## What I tried

- Predicted **3 classes** (up / neutral / down) instead of binary up/down,
  using 30 days of price/volume history as input to an LSTM
- Fixed a real bug along the way: feeding the model raw price levels caused
  it to collapse completely (it learned "what year is it" instead of any
  real pattern, since price drifts a lot over 6 years). Switching to
  percentage-based features (returns, volume changes) fixed some of this,
  but not the core problem
- Validated properly across 6 rolling time windows — not just one
  train/test split — to make sure the result wasn't a lucky/unlucky fluke

## Why this is still a useful project

Same reason as the last one: the goal was never "predict the stock market,"
it was proving I can build and rigorously test an ML pipeline — including
neural networks — and report what actually happened, including debugging
real training failures (mode collapse) along the way, instead of tuning
until a number looks good.

## Files

```
lstm_phase1_data_labels.py     # data pull + 3-class labeling
lstm_phase2_sequences.py       # builds 30-day sequences + stationary features
lstm_phase3_train.py           # trains one LSTM, single train/test split
lstm_phase4_walkforward.py     # the real test — 6 rolling time windows
```

## Running it

```bash
pip install yfinance pandas numpy scikit-learn tensorflow
python lstm_phase1_data_labels.py
python lstm_phase4_walkforward.py
```
