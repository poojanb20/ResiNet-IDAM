import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE = Path(".")
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports" / "it"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the prepared datasets
train = pd.read_csv(
    BASE / "data/it/processed/ids2025_train_scaled.csv"
)
test = pd.read_csv(
    BASE / "data/it/processed/ids2025_test_scaled.csv"
)

X_train = train.drop(columns=["newLabel"])
y_train = train["newLabel"].astype(str).str.strip()

X_test = test.drop(columns=["newLabel"])
y_test = test["newLabel"].astype(str).str.strip()

# Train ONLY on normal training records
X_normal = X_train[y_train == "Normal"]

if len(X_normal) == 0:
    raise ValueError("No Normal records found in the training set.")

print("Normal training records:", len(X_normal))
print("Test records:", len(X_test))

# Build and train the anomaly detector
model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_normal)

# Isolation Forest returns 1 for normal and -1 for anomaly
predictions = model.predict(X_test)

# Convert to binary labels: 0 = normal, 1 = attack/anomaly
y_actual = (y_test != "Normal").astype(int)
y_predicted = (predictions == -1).astype(int)

# Calculate evaluation metrics
tn, fp, fn, tp = confusion_matrix(
    y_actual, y_predicted, labels=[0, 1]
).ravel()

results = {
    "model": "IsolationForest",
    "training_records": int(len(X_normal)),
    "test_records": int(len(X_test)),
    "contamination": 0.01,
    "accuracy": float(accuracy_score(y_actual, y_predicted)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_actual, y_predicted)
    ),
    "attack_precision": float(
        precision_score(y_actual, y_predicted, zero_division=0)
    ),
    "attack_recall": float(
        recall_score(y_actual, y_predicted, zero_division=0)
    ),
    "attack_f1": float(
        f1_score(y_actual, y_predicted, zero_division=0)
    ),
    "false_positive_rate": float(
        fp / (fp + tn) if (fp + tn) else 0
    ),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "per_class_attack_detection_rate": {},
}

# Detection rate for each attack category
for label in sorted(y_test.unique()):
    if label == "Normal":
        continue

    mask = y_test == label
    results["per_class_attack_detection_rate"][label] = {
        "records": int(mask.sum()),
        "detected": int(y_predicted[mask].sum()),
        "detection_rate": float(y_predicted[mask].mean()),
    }

# Save model and results
joblib.dump(
    model,
    MODEL_DIR / "ids2025_isolation_forest.joblib",
)

with open(
    REPORT_DIR / "ids2025_isolation_forest_report.json",
    "w",
) as file:
    json.dump(results, file, indent=2)

output = test[["newLabel"]].copy()
output["predicted_anomaly"] = y_predicted
output["anomaly_score"] = -model.decision_function(X_test)
output.to_csv(
    REPORT_DIR / "ids2025_isolation_forest_predictions.csv",
    index=False,
)

# Display results
print("\n=== ISOLATION FOREST RESULTS ===")
for key, value in results.items():
    if key != "per_class_attack_detection_rate":
        print(f"{key}: {value}")

print("\n=== PER-CLASS ATTACK DETECTION ===")
for label, details in results[
    "per_class_attack_detection_rate"
].items():
    print(
        f"{label}: {details['detected']}/{details['records']} "
        f"detected ({details['detection_rate']:.2%})"
    )

print("\n=== BINARY CONFUSION MATRIX ===")
print("Rows: actual [Normal, Attack]")
print("Columns: predicted [Normal, Attack]")
print(confusion_matrix(y_actual, y_predicted, labels=[0, 1]))

print("\nModel saved: models/ids2025_isolation_forest.joblib")
print("Report saved: reports/it/ids2025_isolation_forest_report.json")
