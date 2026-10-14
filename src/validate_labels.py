
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
data_path = root / "data" / "raw" / "hai-test1.txt"
label_path = root / "data" / "raw" / "label-test1.txt"

print("Loading test data and labels...")
data = pd.read_csv(data_path)
labels = pd.read_csv(label_path)

data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce")

print("\n--- Record counts ---")
print("Test records:", len(data))
print("Label records:", len(labels))

print("\n--- Label information ---")
print("Label values and counts:")
print(labels["label"].value_counts(dropna=False).sort_index().to_string())

print("\n--- Timestamp checks ---")
print("Invalid test timestamps:", data["timestamp"].isna().sum())
print("Invalid label timestamps:", labels["timestamp"].isna().sum())
print("Duplicate test timestamps:", data["timestamp"].duplicated().sum())
print("Duplicate label timestamps:", labels["timestamp"].duplicated().sum())

print("\n--- Alignment checks ---")
print("Same record count:", len(data) == len(labels))
print("Same timestamp sequence:",
      data["timestamp"].equals(labels["timestamp"]))

print("Every test timestamp has a label:",
      data["timestamp"].isin(labels["timestamp"]).all())

print("Every label timestamp exists in test data:",
      labels["timestamp"].isin(data["timestamp"]).all())
