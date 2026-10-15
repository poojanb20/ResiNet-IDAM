
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

SEED = 42
BATCH_SIZE = 256
TARGET_FPR = 0.01  # Target approximately 1% false positives on normal validation data

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

TRAIN_PATH = os.path.join(
    BASE_DIR, "data/it/processed/ids2025_train_scaled.csv"
)
TEST_PATH = os.path.join(
    BASE_DIR, "data/it/processed/ids2025_test_scaled.csv"
)
MODEL_PATH = os.path.join(
    BASE_DIR, "models/ids2025_autoencoder.keras"
)
REPORT_PATH = os.path.join(
    BASE_DIR, "reports/it/ids2025_autoencoder_validation_threshold_report.json"
)
PREDICTIONS_PATH = os.path.join(
    BASE_DIR, "reports/it/ids2025_autoencoder_validation_threshold_predictions.csv"
)

LABEL_COLUMN = "newLabel"
NORMAL_LABEL = "Normal"

print("Loading datasets and saved Autoencoder...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

features = [c for c in train_df.columns if c != LABEL_COLUMN]

if features != [c for c in test_df.columns if c != LABEL_COLUMN]:
    raise ValueError("Train and test features do not match.")

# Recreate the same normal-only fitting/validation split used during training.
normal_df = train_df[
    train_df[LABEL_COLUMN].astype(str).str.strip() == NORMAL_LABEL
]
X_normal = normal_df[features].to_numpy(dtype=np.float32)

X_fit, X_val = train_test_split(
    X_normal,
    test_size=0.15,
    random_state=SEED,
    shuffle=True,
)

# The original model was trained on X_fit. Evaluate validation normals.
val_reconstructed = model.predict(
    X_val, batch_size=BATCH_SIZE, verbose=0
)
val_errors = np.mean(
    np.square(X_val - val_reconstructed), axis=1
)

# Choose an empirical threshold that targets a 1% validation-normal FPR.
# "higher" selects an observed error value instead of interpolating.
threshold = float(
    np.quantile(val_errors, 1 - TARGET_FPR, method="higher")
)

print(f"Normal validation records: {len(X_val):,}")
print(f"Target validation FPR: {TARGET_FPR:.2%}")
print(f"Selected threshold: {threshold:.8f}")

# Evaluate the fixed threshold on the held-out test set only once.
X_test = test_df[features].to_numpy(dtype=np.float32)
y_test = (
    test_df[LABEL_COLUMN].astype(str).str.strip() != NORMAL_LABEL
).astype(int).to_numpy()

test_reconstructed = model.predict(
    X_test, batch_size=BATCH_SIZE, verbose=0
)
test_errors = np.mean(
    np.square(X_test - test_reconstructed), axis=1
)
y_pred = (test_errors > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

normal_test_count = int((y_test == 0).sum())

report = {
    "model": "Dense Autoencoder",
    "dataset": "IDS2025",
    "threshold_selection": "Normal validation reconstruction errors only",
    "target_validation_false_positive_rate": TARGET_FPR,
    "selected_threshold": threshold,
    "validation_normal_records": int(len(X_val)),
    "test_records": int(len(y_test)),
    "test_normal_records": normal_test_count,
    "test_attack_records": int(y_test.sum()),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    "attack_precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "attack_recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "attack_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "test_false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "roc_auc": float(roc_auc_score(y_test, test_errors)),
    "average_precision": float(average_precision_score(y_test, test_errors)),
    "limitations": [
        "The target FPR is an empirical validation target, not a guarantee of the same test FPR.",
        "IDS2025 is derived from CIC-IDS2017.",
        "A random split does not establish zero-day detection.",
    ],
}

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

predictions = test_df[[LABEL_COLUMN]].copy()
predictions["is_attack"] = y_test
predictions["reconstruction_error"] = test_errors
predictions["predicted_anomaly"] = y_pred
predictions.to_csv(PREDICTIONS_PATH, index=False)

print("\n===== VALIDATION-THRESHOLD TEST RESULTS =====")
for key in [
    "accuracy",
    "balanced_accuracy",
    "attack_precision",
    "attack_recall",
    "attack_f1",
    "test_false_positive_rate",
    "roc_auc",
    "average_precision",
]:
    print(f"{key}: {report[key]:.4f}")

print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print("\nSaved report:", REPORT_PATH)
print("Saved predictions:", PREDICTIONS_PATH)
