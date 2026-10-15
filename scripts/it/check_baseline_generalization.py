import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

base = Path(".")

model = joblib.load("models/ids2025_random_forest.joblib")

train = pd.read_csv(
    "data/it/processed/ids2025_train_scaled.csv"
)
test = pd.read_csv(
    "data/it/processed/ids2025_test_scaled.csv"
)

X_train = train.drop(columns=["newLabel"])
y_train = train["newLabel"].astype(str).str.strip()

X_test = test.drop(columns=["newLabel"])
y_test = test["newLabel"].astype(str).str.strip()

for name, X, y in [
    ("TRAIN", X_train, y_train),
    ("TEST", X_test, y_test),
]:
    predictions = model.predict(X)

    print(f"\n=== {name} RESULTS ===")
    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y, predictions), 4)
    )
    print("Macro F1:", round(f1_score(
        y, predictions, average="macro", zero_division=0
    ), 4))
