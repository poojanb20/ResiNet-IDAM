
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Project paths
DATA_PATH = Path("data/processed/hai-train-scaled.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ot_autoencoder.keras"
THRESHOLD_PATH = MODEL_DIR / "autoencoder_threshold.json"

# Keep training manageable on an 8 GB RAM machine
EPOCHS = 20
BATCH_SIZE = 512
THRESHOLD_PERCENTILE = 99


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    print("Loading HAI training data...")
    df = pd.read_csv(DATA_PATH)

    # Timestamp is metadata, not a model input.
    feature_columns = [
        column for column in df.columns if column != "timestamp"
    ]

    # float32 reduces memory usage.
    X = df[feature_columns].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite feature values.")

    # Chronological split: first 80% for training, final 20% for validation.
    split_index = int(len(X) * 0.80)
    X_train = X[:split_index]
    X_val = X[split_index:]

    print(f"Total rows: {len(X):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features: {X.shape[1]}")

    # Small neural network that learns to reconstruct its input.
    inputs = keras.Input(shape=(X.shape[1],))
    encoded = layers.Dense(32, activation="relu")(inputs)
    encoded = layers.Dense(16, activation="relu")(encoded)
    decoded = layers.Dense(32, activation="relu")(encoded)
    outputs = layers.Dense(X.shape[1], activation="linear")(decoded)

    model = keras.Model(inputs, outputs, name="ot_autoencoder")
    model.compile(optimizer="adam", loss="mse")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("\nTraining Autoencoder...")
    model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1
    )

    # Reconstruction error: mean squared difference per row.
    print("\nCalculating validation reconstruction errors...")
    reconstructed = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    errors = np.mean(np.square(X_val - reconstructed), axis=1)

    threshold = float(np.percentile(errors, THRESHOLD_PERCENTILE))
    flagged = int(np.sum(errors > threshold))

    model.save(MODEL_PATH)

    metadata = {
        "threshold": threshold,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "feature_columns": feature_columns,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "features": int(X.shape[1])
    }

    THRESHOLD_PATH.write_text(json.dumps(metadata, indent=2))

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")
    print(f"Validation threshold: {threshold:.8f}")
    print(
        f"Validation rows flagged: {flagged:,}/{len(X_val):,} "
        f"({flagged / len(X_val) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
