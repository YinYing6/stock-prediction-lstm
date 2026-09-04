"""
LSTM Trading Strategy — Phase 4
Rolling walk-forward validation for the LSTM -- same principle as
stock_prediction_phase5.py from the previous project: train fresh on each
window's past data, test on unseen future data, repeat across 6 windows.

Key detail specific to sequence models: to build a test sequence for the
FIRST day of a test window, the model needs the prior 30 days of history --
which mostly falls inside the training window. That's fine and not leakage:
it's the same as a real deployed model always having access to all past
data up to "today." We include that trailing context explicitly below.

Run locally: needs everything from phase1/phase2/phase3
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from lstm_phase1_data_labels import load_data, add_label, TICKER, START, END
from lstm_phase2_sequences import add_stationary_features, build_sequences, scale_sequences, SEQUENCE_LENGTH, FEATURE_COLS
from lstm_phase3_train import build_model, NUM_CLASSES

from tensorflow import keras

N_SPLITS = 6
TEST_SIZE = 120
MIN_TRAIN_SIZE = 500


def rolling_windows(df, test_size, min_train_size, n_splits, context_len):
    """
    Same idea as the previous project's rolling_windows, but each test slice
    also includes `context_len` rows BEFORE the test start, so the first
    test-window sequences have the history they need. Those context rows
    are only ever used as input features, never as prediction targets.
    """
    total = len(df)
    windows = []
    last_test_end = total
    for i in range(n_splits):
        test_end = last_test_end - i * test_size
        test_start = test_end - test_size
        train_end = test_start
        if train_end < min_train_size:
            break
        train_df = df.iloc[:train_end]
        test_df_with_context = df.iloc[max(0, test_start - context_len):test_end]
        windows.append((train_df, test_df_with_context, test_start))
    return list(reversed(windows))


if __name__ == "__main__":
    raw = load_data(TICKER, START, END)
    labeled = add_label(raw)
    featured = add_stationary_features(labeled)

    windows = rolling_windows(featured, TEST_SIZE, MIN_TRAIN_SIZE, N_SPLITS, SEQUENCE_LENGTH)
    print(f"Built {len(windows)} rolling windows\n")

    results = []
    for i, (train_df, test_df_ctx, test_start) in enumerate(windows, 1):
        X_train, y_train = build_sequences(train_df, SEQUENCE_LENGTH, FEATURE_COLS)
        X_test, y_test = build_sequences(test_df_ctx, SEQUENCE_LENGTH, FEATURE_COLS)

        X_train_scaled, X_test_scaled, _ = scale_sequences(X_train, X_test)

        naive_class = np.bincount(y_train).argmax()
        naive_acc = accuracy_score(y_test, np.full_like(y_test, naive_class))

        raw_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(np.sqrt(raw_weights)))

        model = build_model(SEQUENCE_LENGTH, len(FEATURE_COLS), NUM_CLASSES)

        val_split_idx = int(len(X_train_scaled) * 0.85)
        X_fit, X_val = X_train_scaled[:val_split_idx], X_train_scaled[val_split_idx:]
        y_fit, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True
        )

        model.fit(
            X_fit, y_fit,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stop],
            verbose=0,  # quiet -- 6 windows of epoch-by-epoch logs would be unreadable
        )

        test_preds = model.predict(X_test_scaled, verbose=0).argmax(axis=1)
        lstm_acc = accuracy_score(y_test, test_preds)
        n_classes_predicted = len(set(test_preds))

        results.append({"window": i, "naive": naive_acc, "lstm": lstm_acc, "classes_predicted": n_classes_predicted})
        print(f"Window {i}: naive={naive_acc:.4f}  lstm={lstm_acc:.4f}  "
              f"(predicted {n_classes_predicted}/3 classes){'  <- collapsed!' if n_classes_predicted < 3 else ''}")

    naive_vals = [r["naive"] for r in results]
    lstm_vals = [r["lstm"] for r in results]
    beat_count = sum(1 for r in results if r["lstm"] > r["naive"])

    print(f"\n--- Summary across {len(results)} windows ---")
    print(f"Naive mean: {np.mean(naive_vals):.4f}  std: {np.std(naive_vals):.4f}")
    print(f"LSTM mean:  {np.mean(lstm_vals):.4f}  std: {np.std(lstm_vals):.4f}")
    print(f"LSTM beat naive baseline in {beat_count} / {len(results)} windows")
    collapsed_count = sum(1 for r in results if r["classes_predicted"] < 3)
    if collapsed_count:
        print(f"NOTE: model collapsed to <3 classes in {collapsed_count} / {len(results)} windows")
