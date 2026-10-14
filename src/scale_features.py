
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent.parent

train_path = root / "data" / "processed" / "hai-train-features.csv"
test_path = root / "data" / "processed" / "hai-test-labeled.csv"

output_dir = root / "data" / "processed"
model_dir = root / "models"

print("Loading datasets...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

feature_cols = [
    col for col in train.columns if col != "timestamp"
]

# Select exactly the same features in the same order.
X_train = train[feature_cols].astype("float32")
X_test = test[feature_cols].astype("float32")

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nFitting scaler on training data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Scaling test data...")
X_test_scaled = scaler.transform(X_test)

output_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# Preserve timestamps and labels alongside scaled features.
train_scaled = pd.DataFrame(
    X_train_scaled, columns=feature_cols
)
train_scaled.insert(0, "timestamp", train["timestamp"])

test_scaled = pd.DataFrame(
    X_test_scaled, columns=feature_cols
)
test_scaled.insert(0, "timestamp", test["timestamp"])
test_scaled["label"] = test["label"].astype(int)

train_scaled.to_csv(
    output_dir / "hai-train-scaled.csv", index=False
)
test_scaled.to_csv(
    output_dir / "hai-test-scaled.csv", index=False
)

joblib.dump(scaler, model_dir / "standard_scaler.joblib")

print("\nScaling completed.")
print("Training output:", train_scaled.shape)
print("Test output:", test_scaled.shape)
print("Scaler saved:", model_dir / "standard_scaler.joblib")
