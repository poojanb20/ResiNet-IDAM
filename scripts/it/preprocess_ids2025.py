from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
ROOT = Path.cwd()
RAW = ROOT / "data/it/raw/IDS2025.xlsx"
OUT = ROOT / "data/it/processed"
REPORT = ROOT / "reports/it"

OUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

LABEL = "newLabel"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# 1. Load original data
if not RAW.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW}")

df = pd.read_excel(RAW, sheet_name=0)
original_rows = len(df)

# Clean label whitespace and validate labels
df[LABEL] = df[LABEL].astype("string").str.strip()
df = df.dropna(subset=[LABEL])
df = df[df[LABEL] != ""]

# 2. Convert feature columns to numeric values
feature_cols = [c for c in df.columns if c != LABEL]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert positive/negative infinity to missing values
df[feature_cols] = df[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)

# 3. Remove exact duplicate records
before_dedup = len(df)
df = df.drop_duplicates().reset_index(drop=True)
duplicates_removed = before_dedup - len(df)

# Keep only classes with enough examples for a stratified split
class_counts = df[LABEL].value_counts()
if (class_counts < 2).any():
    raise ValueError(
        "At least one class has fewer than 2 rows. "
        "Inspect class counts before splitting."
    )

# 4. Split before fitting preprocessing steps
X = df[feature_cols]
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# 5. Impute missing/infinite values using training medians only
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=feature_cols,
    index=X_train.index,
)
X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_cols,
    index=X_test.index,
)

# 6. Remove features constant in the training set
constant_cols = [
    col for col in X_train_imp.columns
    if X_train_imp[col].nunique(dropna=False) <= 1
]

X_train_imp = X_train_imp.drop(columns=constant_cols)
X_test_imp = X_test_imp.drop(columns=constant_cols)

# 7. Scale using training data only
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imp),
    columns=X_train_imp.columns,
    index=X_train_imp.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imp),
    columns=X_test_imp.columns,
    index=X_test_imp.index,
)

# 8. Save scaled datasets with labels
train_out = X_train_scaled.copy()
train_out[LABEL] = y_train

test_out = X_test_scaled.copy()
test_out[LABEL] = y_test

train_path = OUT / "ids2025_train_scaled.csv"
test_path = OUT / "ids2025_test_scaled.csv"

train_out.to_csv(train_path, index=False)
test_out.to_csv(test_path, index=False)

# Save preprocessing objects for future inference
import joblib

(ROOT / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(imputer, ROOT / "models/ids2025_imputer.joblib")
joblib.dump(scaler, ROOT / "models/ids2025_scaler.joblib")

# Save feature list
feature_path = OUT / "ids2025_feature_columns.json"
feature_path.write_text(
    json.dumps(list(X_train_scaled.columns), indent=2),
    encoding="utf-8",
)

# 9. Create preprocessing report
report = {
    "source": str(RAW.relative_to(ROOT)),
    "original_rows": int(original_rows),
    "rows_after_label_validation": int(before_dedup),
    "exact_duplicate_rows_removed": int(duplicates_removed),
    "rows_after_deduplication": int(len(df)),
    "train_rows": int(len(train_out)),
    "test_rows": int(len(test_out)),
    "original_columns_including_label": int(len(feature_cols) + 1),
    "features_removed_as_constant": constant_cols,
    "final_feature_count": int(X_train_scaled.shape[1]),
    "label_column": LABEL,
    "label_counts_after_deduplication": {
        str(k): int(v) for k, v in df[LABEL].value_counts().items()
    },
    "train_label_counts": {
        str(k): int(v) for k, v in y_train.value_counts().items()
    },
    "test_label_counts": {
        str(k): int(v) for k, v in y_test.value_counts().items()
    },
    "missing_values_imputed": True,
    "imputation_strategy": "Training-set median",
    "scaler": "StandardScaler fitted on training data only",
    "split": "80/20 stratified random split",
    "random_state": RANDOM_STATE,
    "warning": (
        "Random stratification does not guarantee independence "
        "between related network flows. Evaluate possible flow or "
        "time-based leakage before making research claims."
    ),
}

report_path = REPORT / "ids2025_preprocessing_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== IDS2025 PREPROCESSING COMPLETE ===")
print(f"Original rows: {original_rows:,}")
print(f"Exact duplicate rows removed: {duplicates_removed:,}")
print(f"Rows retained: {len(df):,}")
print(f"Training rows: {len(train_out):,}")
print(f"Testing rows: {len(test_out):,}")
print(f"Final feature count: {X_train_scaled.shape[1]}")
print(f"Constant columns removed: {constant_cols}")
print("\nTraining label counts:")
print(y_train.value_counts().to_string())
print("\nTesting label counts:")
print(y_test.value_counts().to_string())
print("\nSaved files:")
print(train_path)
print(test_path)
print(feature_path)
print(report_path)
print(ROOT / "models/ids2025_imputer.joblib")
print(ROOT / "models/ids2025_scaler.joblib")
