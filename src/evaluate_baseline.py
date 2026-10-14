
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

TEST_PATH = "data/processed/hai-test-scaled.csv"
MODEL_PATH = "models/isolation_forest.joblib"

def main():
    print("Loading test data...")
    df = pd.read_csv(TEST_PATH)

    y_true = df["label"].astype(int)
    X_test = df.drop(columns=["timestamp", "label"], errors="ignore")

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Predicting anomalies...")
    # Isolation Forest: -1 = anomaly, 1 = normal
    predictions = model.predict(X_test)
    y_pred = (predictions == -1).astype(int)

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0
    ))

    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

if __name__ == "__main__":
    main()
