
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = Path("data/processed/hai-test-scaled.csv")
MODEL_PATH = Path("models/ot_autoencoder.keras")
THRESHOLD_PATH = Path("models/autoencoder_threshold.json")


def main():
    print("Loading test data and saved model...")

    df = pd.read_csv(DATA_PATH)
    metadata = json.loads(THRESHOLD_PATH.read_text())

    feature_columns = metadata["feature_columns"]
    threshold = metadata["threshold"]

    if "label" not in df.columns:
        raise ValueError("The test dataset must contain a 'label' column.")

    if not set(feature_columns).issubset(df.columns):
        raise ValueError("Test features do not match the training features.")

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y_true = df["label"].to_numpy(dtype=int)

    model = keras.models.load_model(MODEL_PATH)

    print("Calculating reconstruction errors...")
    reconstructed = model.predict(X, batch_size=512, verbose=0)
    errors = np.mean(np.square(X - reconstructed), axis=1)

    # Higher reconstruction error means a potential anomaly.
    y_pred = (errors > threshold).astype(int)

    print("\n--- Autoencoder Test Results ---")
    print(f"Threshold: {threshold:.8f}")
    print(f"Test records: {len(y_true):,}")
    print(f"Actual attacks: {int(y_true.sum()):,}")
    print(f"Predicted anomalies: {int(y_pred.sum()):,}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        )
    )

    print("\nAttack-focused metrics:")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")


if __name__ == "__main__":
    main()
