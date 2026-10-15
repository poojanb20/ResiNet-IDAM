
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Project paths and imports
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.zero_trust.risk_engine import RiskInput, calculate_risk

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "reports/it/ids2025_autoencoder_validation_threshold_predictions.csv",
)
TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data/it/processed/ids2025_train_scaled.csv",
)
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models/ids2025_autoencoder.keras",
)
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "reports/it/ids2025_zero_trust_results.csv",
)

SEED = 42
BATCH_SIZE = 256
LABEL_COLUMN = "newLabel"

# Simulated context values for this prototype.
# Replace these with real asset/access metadata later.
ASSET_CRITICALITY = 0.8
TRUST_SCORE = 0.6

# The Autoencoder does not produce a classification confidence.
# Keep it at zero rather than pretending reconstruction error
# is a classification probability.
CLASSIFICATION_CONFIDENCE = 0.0

# --------------------------------------------------
# 2. Load saved predictions and model
# --------------------------------------------------

print("Loading Autoencoder predictions...")

predictions = pd.read_csv(PREDICTIONS_PATH)
train_df = pd.read_csv(TRAIN_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

if "reconstruction_error" not in predictions.columns:
    raise ValueError("Prediction file has no reconstruction_error column.")

features = [
    column for column in train_df.columns
    if column != LABEL_COLUMN
]

normal_df = train_df[
    train_df[LABEL_COLUMN].astype(str).str.strip() == "Normal"
]

X_normal = normal_df[features].to_numpy(dtype=np.float32)

# Recreate the validation split used in model training.
_, X_val = train_test_split(
    X_normal,
    test_size=0.15,
    random_state=SEED,
    shuffle=True,
)

# --------------------------------------------------
# 3. Calculate normal validation reconstruction errors
# --------------------------------------------------

print("Calculating normal validation error distribution...")

val_reconstructed = model.predict(
    X_val,
    batch_size=BATCH_SIZE,
    verbose=0,
)

val_errors = np.mean(
    np.square(X_val - val_reconstructed),
    axis=1,
)

sorted_val_errors = np.sort(val_errors)

# --------------------------------------------------
# 4. Convert each error into an anomaly score
# --------------------------------------------------

def calculate_anomaly_score(error):
    """
    Empirical percentile relative to normal validation errors.

    A score near 1 means the error is unusually high compared
    with normal validation traffic. It is NOT attack probability.
    """
    rank = np.searchsorted(
        sorted_val_errors,
        error,
        side="right",
    )

    return float(rank / len(sorted_val_errors))


print("Applying Zero Trust risk scoring...")

risk_scores = []
risk_levels = []
recommended_actions = []
anomaly_scores = []

for error in predictions["reconstruction_error"].to_numpy():
    anomaly_score = calculate_anomaly_score(float(error))

    result = calculate_risk(
        RiskInput(
            anomaly_score=anomaly_score,
            classification_confidence=CLASSIFICATION_CONFIDENCE,
            asset_criticality=ASSET_CRITICALITY,
            trust_score=TRUST_SCORE,
        )
    )

    anomaly_scores.append(anomaly_score)
    risk_scores.append(result["risk_score"])
    risk_levels.append(result["risk_level"])
    recommended_actions.append(result["recommended_action"])

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

output = predictions.copy()

output["anomaly_score"] = anomaly_scores
output["asset_criticality"] = ASSET_CRITICALITY
output["trust_score"] = TRUST_SCORE
output["risk_score"] = risk_scores
output["risk_level"] = risk_levels
output["recommended_action"] = recommended_actions

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

print("\n===== DETECTION + ZERO TRUST RESULTS =====")
print(f"Records processed: {len(output):,}")
print(f"Low risk: {(output['risk_level'] == 'LOW').sum():,}")
print(f"Medium risk: {(output['risk_level'] == 'MEDIUM').sum():,}")
print(f"High risk: {(output['risk_level'] == 'HIGH').sum():,}")

print("\nRecommended actions:")
print(output["recommended_action"].value_counts().to_string())

print("\nSample results:")
columns_to_show = [
    LABEL_COLUMN,
    "reconstruction_error",
    "anomaly_score",
    "risk_score",
    "risk_level",
    "recommended_action",
]

print(output[columns_to_show].head(10).to_string(index=False))

print("\nSaved results:", OUTPUT_PATH)
print(
    "\nNote: Asset criticality and trust are simulated. "
    "Risk scores are prototype decisions, not production access controls."
)
