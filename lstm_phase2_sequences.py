"""
LSTM Trading Strategy — Phase 2
Turn daily rows into sliding-window sequences: each training example is
"the past N days" of raw market data, used to predict the label on the day
right after the window ends.

Deliberately using raw price/volume features here (not hand-engineered
indicators like RSI/MACD from the previous project) -- one point of using
an LSTM is letting the model learn temporal patterns directly from the
sequence, rather than relying on features a human pre-computed. Simpler to
reason about and debug, matching "Level 1" scope.

Run locally: pip install numpy scikit-learn (plus everything from phase1)
"""

import numpy as np
import numpy as np
from sklearn.preprocessing import StandardScaler

from lstm_phase1_data_labels import load_data, add_label, TICKER, START, END

SEQUENCE_LENGTH = 30  # "past 30 days" per your original diagram

# NOT raw OHLCV -- raw price levels drift a lot over 6+ years (e.g. NVDA
# went from ~$50 to $200+), so a model fed raw price can partly just learn
# "what year is this" instead of a meaningful, repeatable pattern. Using
# relative/percentage features instead keeps the input roughly stationary
# (comparable scale regardless of what era of the stock's price it's from).
FEATURE_COLS = ["ret_close", "ret_open_gap", "hl_range", "co_range", "vol_change"]


def add_stationary_features(df):
    """
    Converts raw OHLCV into relative features. Every calculation only looks
    at the current day and earlier -- no lookahead into the future.
    """
    df = df.copy()
    df["ret_close"] = df["Close"].pct_change()                          # today's return
    df["ret_open_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)  # overnight gap
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]             # today's range, as % of price
    df["co_range"] = (df["Close"] - df["Open"]) / df["Open"]            # today's close vs open, as %
    df["vol_change"] = df["Volume"].pct_change()                        # volume change, not raw volume level
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    return df


def build_sequences(df, seq_len, feature_cols):
    """
    Returns X of shape (num_samples, seq_len, num_features) and y of shape (num_samples,).
    X[i] = days [i, i+seq_len) as raw features. y[i] = label on day i+seq_len
    (i.e. the day immediately after the window -- this is "tomorrow" relative
    to the window's last day, matching the label built in Phase 1).
    """
    values = df[feature_cols].values
    labels = df["label"].values

    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X), np.array(y)


def chronological_split(X, y, train_frac=0.8):
    split_idx = int(len(X) * train_frac)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def scale_sequences(X_train, X_test):
    """
    Scale each feature independently, fit ONLY on train data (same leakage
    rule as the previous project). Sequences are 3D (samples, timesteps,
    features), but StandardScaler only works on 2D -- so we flatten to 2D,
    scale, then reshape back.
    """
    n_train, seq_len, n_features = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features))

    X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    raw = load_data(TICKER, START, END)
    labeled = add_label(raw)
    featured = add_stationary_features(labeled)

    X, y = build_sequences(featured, SEQUENCE_LENGTH, FEATURE_COLS)
    print(f"Built {len(X)} sequences, each shaped {X.shape[1:]} (days, features)")

    X_train, X_test, y_train, y_test = chronological_split(X, y)
    print(f"Train: {X_train.shape[0]} sequences | Test: {X_test.shape[0]} sequences")

    X_train_scaled, X_test_scaled, scaler = scale_sequences(X_train, X_test)
    print(f"\nScaled train sample stats -- mean~0, std~1 expected:")
    print(f"  mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")

    print(f"\nLabel distribution in train: {np.bincount(y_train) / len(y_train)}")
    print(f"Label distribution in test:  {np.bincount(y_test) / len(y_test)}")
