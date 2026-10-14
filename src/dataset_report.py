
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "raw" / "hai-train1.txt"
report_path = project_dir / "data" / "reports" / "dataset_report.txt"

print("Analyzing dataset...")
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

constant_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

numeric_cols = df.select_dtypes(include="number").columns

lines = [
    "RESINET-IDAM: DATASET INSPECTION REPORT",
    "=" * 45,
    "",
    "Dataset file: hai-train1.txt",
    "Dataset type: Industrial time-series telemetry",
    f"Total records: {len(df)}",
    f"Total columns: {len(df.columns)}",
    f"Process features (excluding timestamp): {len(df.columns) - 1}",
    f"File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB",
    "",
    "TIMESTAMP SUMMARY",
    "-" * 25,
    f"First timestamp: {df['timestamp'].min()}",
    f"Last timestamp: {df['timestamp'].max()}",
    f"Invalid timestamps: {df['timestamp'].isna().sum()}",
    "",
    "DATA QUALITY",
    "-" * 25,
    f"Missing values: {df.isna().sum().sum()}",
    f"Duplicate rows: {df.duplicated().sum()}",
    f"Constant columns: {len(constant_cols)}",
    "",
    "CONSTANT COLUMNS",
    "-" * 25,
    *constant_cols,
    "",
    "COLUMN NAMES AND DATA TYPES",
    "-" * 25,
    *[f"{col}: {df[col].dtype}" for col in df.columns],
    "",
    "NUMERIC FEATURE SUMMARY",
    "-" * 25,
    df[numeric_cols].describe().round(3).to_string(),
    "",
    "IMPORTANT NOTE",
    "-" * 25,
    "No obvious attack-label column was identified in the inspected header.",
    "Verify the dataset release and official label scheme before training",
    "or evaluating supervised attack detection models."
]

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines), encoding="utf-8")

print("Report generated successfully.")
print("Saved to:", report_path)
