
import os
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

TRAIN_PATH = "data/processed/hai-train-scaled.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")


def main():
    print("Loading OT training data...")

    df = pd.read_csv(TRAIN_PATH)
    X = df.drop(columns=["timestamp"], errors="ignore")

    if X.empty:
        raise ValueError("Training dataset is empty.")

    if X.isnull().any().any():
        raise ValueError("Training dataset contains missing values.")

    print(f"Training rows: {len(X):,}")
    print(f"Features: {X.shape[1]}")

    print("Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=2
    )
    model.fit(X)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
