"""
train_models.py
---------------
Trains SimpleRNN, LSTM, and GRU models on the IMDB dataset and saves them
to the models/ directory along with training history.

Run once before launching the Streamlit app:
    python train_models.py
"""

import os
import json
import time
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Config ────────────────────────────────────────────────────────────────────
NUM_WORDS  = 10000
MAXLEN     = 200
EMBED_DIM  = 32
UNITS      = 32
EPOCHS     = 5
BATCH_SIZE = 64
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading IMDB dataset …")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
X_train = pad_sequences(X_train, maxlen=MAXLEN)
X_test  = pad_sequences(X_test,  maxlen=MAXLEN)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── Helper ────────────────────────────────────────────────────────────────────
def build_model(layer):
    model = Sequential([
        Embedding(input_dim=NUM_WORDS, output_dim=EMBED_DIM, input_length=MAXLEN),
        layer,
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_save(name, layer):
    print(f"\n{'='*50}")
    print(f"Training {name} …")
    model = build_model(layer)

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )
    elapsed = round(time.time() - t0, 2)

    # Evaluate
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "accuracy":          float(accuracy_score(y_test, y_pred)),
        "precision":         float(precision_score(y_test, y_pred)),
        "recall":            float(recall_score(y_test, y_pred)),
        "f1":                float(f1_score(y_test, y_pred)),
        "val_accuracy":      float(history.history["val_accuracy"][-1]),
        "val_loss":          float(history.history["val_loss"][-1]),
        "training_time":     elapsed,
        "num_params":        int(model.count_params()),
        "train_acc_history": [float(v) for v in history.history["accuracy"]],
        "val_acc_history":   [float(v) for v in history.history["val_accuracy"]],
        "train_loss_history":[float(v) for v in history.history["loss"]],
        "val_loss_history":  [float(v) for v in history.history["val_loss"]],
    }

    # Save model + metrics
    model_path   = os.path.join(MODELS_DIR, f"{name.lower()}_model.keras")
    metrics_path = os.path.join(MODELS_DIR, f"{name.lower()}_metrics.json")
    model.save(model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  Saved to : {model_path}")
    return metrics


# ── Train all three ───────────────────────────────────────────────────────────
results = {}
results["SimpleRNN"] = train_and_save("SimpleRNN", SimpleRNN(UNITS))
results["LSTM"]      = train_and_save("LSTM",      LSTM(UNITS))
results["GRU"]       = train_and_save("GRU",       GRU(UNITS))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Model Comparison Summary")
print("="*50)
header = f"{'Metric':<22} {'SimpleRNN':>12} {'LSTM':>12} {'GRU':>12}"
print(header)
print("-" * len(header))
for key, label in [
    ("accuracy",      "Accuracy"),
    ("precision",     "Precision"),
    ("recall",        "Recall"),
    ("f1",            "F1 Score"),
    ("val_accuracy",  "Val Accuracy"),
    ("val_loss",      "Val Loss"),
    ("training_time", "Train Time (s)"),
    ("num_params",    "Parameters"),
]:
    row = f"{label:<22}"
    for m in ["SimpleRNN", "LSTM", "GRU"]:
        v = results[m][key]
        row += f" {v:>12.4f}" if isinstance(v, float) else f" {v:>12}"
    print(row)

best = max(results, key=lambda m: results[m]["accuracy"])
print(f"\nBest Model (by accuracy): {best}")
