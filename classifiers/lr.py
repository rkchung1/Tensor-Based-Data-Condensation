import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report  # type: ignore
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def load_dataset(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from a CSV file and separate features and labels.
    Assumes the label column is named 'label'.
    """
    data = pd.read_csv(file_path)
    X = data.drop(columns=["label"]).values
    y = data["label"].values
    return X, y

def min_max_scale(X: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """
    Scale features to [0,1] using min-max normalization.
    """
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    return (X - min_val) / range_val

def main(args):
    # Load datasets
    X_train, y_train = load_dataset(args.train_dataset)
    X_test, y_test   = load_dataset(args.test_dataset)

    # Compute min/max from training set
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)

    # Scale data
    X_train = min_max_scale(X_train, min_val, max_val)
    X_test  = min_max_scale(X_test,  min_val, max_val)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    # DataLoaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Model, loss, optimizer
    input_size  = X_train.shape[1]
    output_size = len(set(y_train))
    model = LogisticRegression(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("Training Logistic Regression...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.numpy())
            all_trues.extend(y_batch.numpy())

    acc = accuracy_score(all_trues, all_preds)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Logistic Regression classifier on a dataset.")
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to the training dataset CSV file.")
    parser.add_argument("--test_dataset",  type=str, required=True,
                        help="Path to the testing dataset CSV file.")
    parser.add_argument("--batch_size",    type=int,   default=64,
                        help="Batch size for training and testing.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--epochs",        type=int,   default=10,
                        help="Number of epochs to train the model.")
    args = parser.parse_args()

    main(args)