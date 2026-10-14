
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent

data_path = root / "data" / "raw" / "hai-test1.txt"
label_path = root / "data" / "raw" / "label-test1.txt"
output_path = root / "data" / "processed" / "hai-test-labeled.csv"

data = pd.read_csv(data_path)
labels = pd.read_csv(label_path)

data["timestamp"] = pd.to_datetime(data["timestamp"])
labels["timestamp"] = pd.to_datetime(labels["timestamp"])

if data["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps in test data.")

if labels["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps in labels.")

merged = data.merge(
    labels,
    on="timestamp",
    how="left",
    validate="one_to_one"
)

if merged["label"].isna().any():
    raise ValueError("Some test records have no labels.")

merged["label"] = merged["label"].astype(int)

output_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(output_path, index=False)

print("Labeled test dataset created successfully.")
print("Output:", output_path)
print("Records:", len(merged))
print("Columns:", len(merged.columns))
print("\nLabel distribution:")
print(merged["label"].value_counts().sort_index())
