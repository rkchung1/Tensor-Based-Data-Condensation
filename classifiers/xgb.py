import argparse
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_dataset(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from a CSV file and separate features and labels.
    Assumes the label column is named 'label'.
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y

def min_max_scale(X: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """
    Scale features to [0,1] using min–max normalization.
    """
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    return (X - min_val) / range_val

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost classifier on a dataset."
    )
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to the training dataset CSV file.")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Path to the testing dataset CSV file.")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the ensemble (default: 100).")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Maximum tree depth (default: 3).")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Boosting learning rate (default: 0.1).")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Subsample ratio of the training instances (default: 1.0).")
    args = parser.parse_args()

    # Load data
    X_train, y_train = load_dataset(args.train_dataset)
    X_test, y_test   = load_dataset(args.test_dataset)

    # Feature scaling
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    X_train = min_max_scale(X_train, min_val, max_val)
    X_test  = min_max_scale(X_test,  min_val, max_val)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        # use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()