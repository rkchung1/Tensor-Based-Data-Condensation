import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_sizes[1], output_size)
        )

    def forward(self, x):
        return self.model(x)


# Data utils
def load_dataset(file_path: str):
    """
    Load a dataset from a CSV file. Separate labels and keep all features.
    Returns:
        X_df: DataFrame of features (numeric + categorical)
        y: 1D NumPy array of labels
    """
    df = pd.read_csv(file_path)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataset.")
    y = df["label"].values
    X_df = df.drop(columns=["label"])
    return X_df, y

def min_max_scale(X: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """
    Apply min-max scaling to scale features to the range [0, 1].
    """
    range_val = max_val - min_val
    range_val[range_val == 0] = 1  # Avoid division by zero for constant features
    return (X - min_val) / range_val


def main(args):
    # Load datasets
    X_train_df, y_train = load_dataset(args.train_dataset)
    X_test_df,  y_test  = load_dataset(args.test_dataset)

    # Align test dataset columns with train dataset columns (assumes same schema)
    X_test_df = X_test_df.reindex(columns=X_train_df.columns)

    # Identify numeric and (binary) categorical columns
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns
    cat_cols = [col for col in num_cols if set(pd.unique(X_train_df[col])) <= {0, 1}]
    num_cols = num_cols.difference(cat_cols)

    # Compute min/max on training numeric features
    min_val = X_train_df[num_cols].min(axis=0).values
    max_val = X_train_df[num_cols].max(axis=0).values

    # Scale numeric features
    X_train_num = min_max_scale(X_train_df[num_cols].values, min_val, max_val)
    X_test_num  = min_max_scale(X_test_df[num_cols].values,  min_val, max_val)

    # Keep categorical features unchanged
    X_train_cat = X_train_df[cat_cols].values if len(cat_cols) else np.empty((len(X_train_df), 0))
    X_test_cat  = X_test_df[cat_cols].values  if len(cat_cols) else np.empty((len(X_test_df), 0))

    # Concatenate numeric and categorical features
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test  = np.hstack([X_test_num,  X_test_cat])

    # Convert to CPU tensors here; move to GPU per-batch for best throughput
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # Initialize model, loss, optimizer
    input_size = X_train.shape[1]
    hidden_sizes = [128, 64]
    output_size = len(np.unique(y_train))
    model = MLP(input_size, hidden_sizes, output_size, dropout_prob=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    use_cuda = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # Training
    print("Training the MLP model...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Move each batch to GPU with non_blocking copies
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"Epoch [{epoch}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    print("\nEvaluating the model...")
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on a dataset, scaling only numeric features.")
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
    parser.add_argument("--num_workers",   type=int,   default=0,
                        help="DataLoader workers (try 2–4 in Colab).")
    args = parser.parse_args()
    main(args)