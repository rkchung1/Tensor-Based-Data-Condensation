import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
    Scale features to [0,1] using min-max normalization.
    """
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    return (X - min_val) / range_val

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a KNN classifier on a dataset.")
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to the training dataset CSV file.")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Path to the testing dataset CSV file.")
    parser.add_argument("--n_neighbors", type=int, default=5,
                        help="Number of neighbors for KNN (default: 5).")
    parser.add_argument("--weights", type=str, choices=["uniform", "distance"],
                        default="uniform",
                        help="Weight function used in prediction (default: uniform).")
    args = parser.parse_args()

    # Load data
    X_train, y_train = load_dataset(args.train_dataset)
    X_test, y_test   = load_dataset(args.test_dataset)

    # Feature scaling
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    X_train = min_max_scale(X_train, min_val, max_val)
    X_test  = min_max_scale(X_test,  min_val, max_val)

    # Train KNN
    model = KNeighborsClassifier(n_neighbors=args.n_neighbors,
                                 weights=args.weights)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()