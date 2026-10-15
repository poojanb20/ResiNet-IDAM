from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path.cwd()
DATA = ROOT / "data/it/processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports/it"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

LABEL = "newLabel"

# Load the already-preprocessed datasets
train = pd.read_csv(DATA / "ids2025_train_scaled.csv")
test = pd.read_csv(DATA / "ids2025_test_scaled.csv")

# Normalize label text
train[LABEL] = train[LABEL].astype(str).str.strip()
test[LABEL] = test[LABEL].astype(str).str.strip()

X_train = train.drop(columns=[LABEL])
y_train = train[LABEL]

X_test = test.drop(columns=[LABEL])
y_test = test[LABEL]

# Ensure train and test feature columns match
if X_train.columns.tolist() != X_test.columns.tolist():
    raise ValueError("Training and testing feature columns do not match.")

# Train a baseline classifier
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# Evaluate on the held-out test set
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, predictions, average="weighted", zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=model.classes_,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    y_test, predictions, labels=model.classes_
)

# Save model
model_path = MODELS / "ids2025_random_forest.joblib"
joblib.dump(model, model_path)

# Save classification report
report_path = REPORTS / "ids2025_random_forest_report.json"
report_path.write_text(
    json.dumps({
        "model": "RandomForestClassifier",
        "test_rows": int(len(y_test)),
        "feature_count": int(X_train.shape[1]),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "test_label_counts": {
            str(k): int(v) for k, v in y_test.value_counts().items()
        },
    }, indent=2),
    encoding="utf-8",
)

# Save confusion matrix and feature importance
pd.DataFrame(
    matrix,
    index=[f"actual_{c}" for c in model.classes_],
    columns=[f"predicted_{c}" for c in model.classes_],
).to_csv(REPORTS / "ids2025_confusion_matrix.csv")

pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values(
    "importance", ascending=False
).to_csv(REPORTS / "ids2025_feature_importance.csv", index=False)

# Print results
print("\n=== RANDOM FOREST BASELINE RESULTS ===")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")

print("\n=== PER-CLASS RESULTS ===")
print(classification_report(
    y_test, predictions, labels=model.classes_, zero_division=0
))

print("\nSaved model:", model_path)
print("Saved report:", report_path)
print("Saved confusion matrix:", REPORTS / "ids2025_confusion_matrix.csv")
print("Saved feature importance:", REPORTS / "ids2025_feature_importance.csv")
