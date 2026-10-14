
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
input_path = root / "data" / "raw" / "hai-train1.txt"
output_path = root / "data" / "processed" / "hai-train-features.csv"

print("Loading training data...")
df = pd.read_csv(input_path)

# Keep timestamps separately from numerical model features.
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if df["timestamp"].isna().any():
    raise ValueError("Invalid timestamps found.")

if df.isna().any().any():
    raise ValueError("Missing values found. Inspect before proceeding.")

features = df.drop(columns=["timestamp"])

# Identify features with no variation in the training dataset.
constant_cols = [
    col for col in features.columns
    if features[col].nunique(dropna=False) <= 1
]

# Remove constant features from model input.
features = features.drop(columns=constant_cols)

# Save timestamps and remaining features together for traceability.
processed = pd.concat(
    [df[["timestamp"]], features],
    axis=1
)

output_path.parent.mkdir(parents=True, exist_ok=True)
processed.to_csv(output_path, index=False)

print("\nTraining features prepared.")
print("Saved to:", output_path)
print("Records:", len(processed))
print("Original process features:", len(df.columns) - 1)
print("Removed constant features:", len(constant_cols))
print("Remaining process features:", len(features.columns))
print("Output columns:", len(processed.columns))
print("Removed columns:", constant_cols)
