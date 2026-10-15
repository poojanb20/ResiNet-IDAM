
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------
# 2. Paths and configuration
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

TRAIN_PATH = os.path.join(
    BASE_DIR, "data/it/processed/ids2025_train_scaled.csv"
)
TEST_PATH = os.path.join(
    BASE_DIR, "data/it/processed/ids2025_test_scaled.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports/it")

MODEL_PATH = os.path.join(
    MODEL_DIR, "ids2025_autoencoder.keras"
)
REPORT_PATH = os.path.join(
    REPORT_DIR, "ids2025_autoencoder_report.json"
)
PREDICTIONS_PATH = os.path.join(
    REPORT_DIR, "ids2025_autoencoder_predictions.csv"
)

LABEL_COLUMN = "newLabel"
NORMAL_LABEL = "Normal"

# This percentile is chosen using validation NORMAL data only.
THRESHOLD_PERCENTILE = 99.0
EPOCHS = 50
BATCH_SIZE = 256

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------
# 3. Load prepared datasets
# --------------------------------------------------

print("Loading preprocessed datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

if LABEL_COLUMN not in train_df or LABEL_COLUMN not in test_df:
    raise ValueError(f"Expected label column '{LABEL_COLUMN}'.")

feature_columns = [
    column for column in train_df.columns
    if column != LABEL_COLUMN
]

if feature_columns != [
    column for column in test_df.columns
    if column != LABEL_COLUMN
]:
    raise ValueError("Train/test feature columns or order do not match.")

# Train the autoencoder on normal traffic only.
normal_df = train_df[
    train_df[LABEL_COLUMN].astype(str).str.strip() == NORMAL_LABEL
]

if len(normal_df) < 10:
    raise ValueError("Not enough normal records to train the autoencoder.")

X_normal = normal_df[feature_columns].to_numpy(dtype=np.float32)

# Split normal training data into fitting and validation subsets.
X_fit, X_val = train_test_split(
    X_normal,
    test_size=0.15,
    random_state=SEED,
    shuffle=True,
)

X_test = test_df[feature_columns].to_numpy(dtype=np.float32)

y_test = (
    test_df[LABEL_COLUMN].astype(str).str.strip() != NORMAL_LABEL
).astype(int).to_numpy()

print(f"Features: {len(feature_columns)}")
print(f"Normal training records: {len(X_normal):,}")
print(f"Autoencoder fitting records: {len(X_fit):,}")
print(f"Normal validation records: {len(X_val):,}")
print(f"Test records: {len(X_test):,}")
print(f"Test attack records: {int(y_test.sum()):,}")

# --------------------------------------------------
# 4. Build the dense autoencoder
# --------------------------------------------------

input_dim = X_fit.shape[1]

inputs = tf.keras.Input(shape=(input_dim,), name="network_features")

# Encoder
x = tf.keras.layers.Dense(
    48, activation="relu", name="encoder_1"
)(inputs)
x = tf.keras.layers.Dense(
    24, activation="relu", name="encoder_2"
)(x)
latent = tf.keras.layers.Dense(
    12, activation="relu", name="latent_representation"
)(x)

# Decoder
x = tf.keras.layers.Dense(
    24, activation="relu", name="decoder_1"
)(latent)
x = tf.keras.layers.Dense(
    48, activation="relu", name="decoder_2"
)(x)
outputs = tf.keras.layers.Dense(
    input_dim, activation="linear", name="reconstruction"
)(x)

autoencoder = tf.keras.Model(inputs, outputs, name="ids2025_autoencoder")

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
)

autoencoder.summary()

# --------------------------------------------------
# 5. Train using normal traffic only
# --------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1,
)

print("\nTraining autoencoder on NORMAL traffic only...")

history = autoencoder.fit(
    X_fit,
    X_fit,
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1,
)

autoencoder.save(MODEL_PATH)
print(f"\nSaved model: {MODEL_PATH}")

# --------------------------------------------------
# 6. Select anomaly threshold from normal validation data
# --------------------------------------------------

val_reconstructed = autoencoder.predict(X_val, batch_size=BATCH_SIZE, verbose=0)

val_errors = np.mean(
    np.square(X_val - val_reconstructed), axis=1
)

threshold = float(
    np.percentile(val_errors, THRESHOLD_PERCENTILE)
)

print(f"\nThreshold ({THRESHOLD_PERCENTILE}th normal-validation percentile): {threshold:.8f}")

# --------------------------------------------------
# 7. Evaluate once on the held-out test set
# --------------------------------------------------

test_reconstructed = autoencoder.predict(
    X_test, batch_size=BATCH_SIZE, verbose=0
)

test_errors = np.mean(
    np.square(X_test - test_reconstructed), axis=1
)

# Higher reconstruction error means more anomalous.
y_pred = (test_errors > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

normal_count = int((y_test == 0).sum())
attack_count = int((y_test == 1).sum())

metrics = {
    "model": "Dense Autoencoder",
    "dataset": "IDS2025",
    "training_method": "Trained only on normal training records",
    "threshold_method": (
        f"{THRESHOLD_PERCENTILE}th percentile of normal validation reconstruction errors"
    ),
    "threshold": threshold,
    "epochs_completed": len(history.history["loss"]),
    "features": len(feature_columns),
    "train_normal_records": int(len(X_normal)),
    "validation_normal_records": int(len(X_val)),
    "test_records": int(len(y_test)),
    "test_normal_records": normal_count,
    "test_attack_records": attack_count,
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    "precision_attack": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall_attack": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_attack": float(f1_score(y_test, y_pred, zero_division=0)),
    "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
    "roc_auc": (
        float(roc_auc_score(y_test, test_errors))
        if len(np.unique(y_test)) == 2 else None
    ),
    "average_precision": (
        float(average_precision_score(y_test, test_errors))
        if len(np.unique(y_test)) == 2 else None
    ),
    "limitations": [
        "Threshold was selected from normal validation records, not attack-labelled test records.",
        "Anomaly detection is not proof of zero-day detection.",
        "Random train/test splitting may leave related network flows across both sets.",
        "IDS2025 is derived from CIC-IDS2017; results are not independent-dataset validation.",
    ],
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

predictions_df = test_df[[LABEL_COLUMN]].copy()
predictions_df["is_attack"] = y_test
predictions_df["reconstruction_error"] = test_errors
predictions_df["predicted_anomaly"] = y_pred

predictions_df.to_csv(PREDICTIONS_PATH, index=False)

print("\n========== AUTOENCODER TEST RESULTS ==========")
print(f"Accuracy:           {metrics['accuracy']:.4f}")
print(f"Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")
print(f"Attack precision:   {metrics['precision_attack']:.4f}")
print(f"Attack recall:      {metrics['recall_attack']:.4f}")
print(f"Attack F1:          {metrics['f1_attack']:.4f}")
print(f"False positive rate:{metrics['false_positive_rate']:.4f}")
print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
print(f"Average precision:  {metrics['average_precision']:.4f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\nSaved report:", REPORT_PATH)
print("Saved predictions:", PREDICTIONS_PATH)
