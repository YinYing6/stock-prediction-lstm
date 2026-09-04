"""
LSTM Trading Strategy — Phase 3
Build and train an LSTM for 3-class (down/neutral/up) prediction.

Run locally: pip install tensorflow (plus everything from phase1/phase2)
Note: tensorflow install is large (~500MB+) and can take a few minutes.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from lstm_phase1_data_labels import load_data, add_label, TICKER, START, END
from lstm_phase2_sequences import (
    build_sequences, chronological_split, scale_sequences, add_stationary_features,
    SEQUENCE_LENGTH, FEATURE_COLS,
)

NUM_CLASSES = 3


def build_model(seq_len, n_features, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(32, return_sequences=False),  # 32 units -- deliberately small given ~1500 training sequences
        layers.Dropout(0.3),                       # dropout to fight overfitting -- small dataset, easy to memorize
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),  # lower LR + gradient clipping -- prevents the optimizer from taking a step so large it gets stuck
        loss="sparse_categorical_crossentropy",  # use this (not categorical_crossentropy) since y is integer labels, not one-hot
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    raw = load_data(TICKER, START, END)
    labeled = add_label(raw)
    featured = add_stationary_features(labeled)

    X, y = build_sequences(featured, SEQUENCE_LENGTH, FEATURE_COLS)
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_sequences(X_train, X_test)

    # Naive baseline -- same principle as the previous project: majority class from TRAIN only
    majority_class = np.bincount(y_train).argmax()
    naive_preds = np.full_like(y_test, majority_class)
    naive_acc = accuracy_score(y_test, naive_preds)
    print(f"Naive baseline (always predict class {majority_class}): {naive_acc:.4f}")

    # Class weights -- "neutral" is underrepresented (~15%), without this the
    # model can just learn to never predict it and still look decent on accuracy.
    # Using sqrt-dampened weights instead of raw "balanced" -- full balanced
    # weighting (2.26x on neutral) was too aggressive and destabilized training
    # (model collapsed to predicting one class and got stuck).
    raw_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    dampened_weights = np.sqrt(raw_weights)
    class_weight_dict = dict(enumerate(dampened_weights))
    print(f"Class weights (sqrt-dampened): {class_weight_dict}")

    model = build_model(SEQUENCE_LENGTH, len(FEATURE_COLS), NUM_CLASSES)
    model.summary()

    # Held-out validation split from train (chronological -- last 15% of TRAIN, not random)
    val_split_idx = int(len(X_train_scaled) * 0.85)
    X_fit, X_val = X_train_scaled[:val_split_idx], X_train_scaled[val_split_idx:]
    y_fit, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True
    )  # stop once validation accuracy stops improving -- with class weights active,
       # val_loss can be a misleading signal, so tracking accuracy directly is safer here

    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1,
    )

    # ── Evaluate on the untouched test set ──
    test_probs = model.predict(X_test_scaled)
    test_preds = test_probs.argmax(axis=1)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\n--- Results ---")
    print(f"Naive baseline: {naive_acc:.4f}")
    print(f"LSTM accuracy:  {test_acc:.4f}")

    # Guardrail: catch mode collapse automatically (model only ever predicting 1-2 of the 3 classes)
    predicted_classes = set(np.unique(test_preds))
    if len(predicted_classes) < NUM_CLASSES:
        missing = set(range(NUM_CLASSES)) - predicted_classes
        print(f"\n*** WARNING: model never predicted class(es) {missing} on the test set. ***")
        print("*** This usually means training collapsed rather than genuinely learning. ***")

    print(f"\nConfusion matrix (rows=actual, cols=predicted, order=down/neutral/up):")
    print(confusion_matrix(y_test, test_preds))
    print(f"\n{classification_report(y_test, test_preds, target_names=['down', 'neutral', 'up'])}")
