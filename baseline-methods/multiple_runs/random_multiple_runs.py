import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from mlp import load_dataset, min_max_scale, MLP
from random_sample_tdbench import random_sample

def main():
    parser = argparse.ArgumentParser(
        description="Sample via per-class random sampling and evaluate with the same MLP."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('test_csv',  help="Path to the test CSV file.")
    parser.add_argument('-N', '--sample_size',
                        type=float,
                        default=100,
                        help="Number of samples per class (if >1) or fraction (if <=1).")
    parser.add_argument('--match_balance',
                        action='store_true',
                        help="Preserve class distribution when sampling.")
    parser.add_argument('--batch_size',
                        type=int, default=64,
                        help="Batch size for MLP training.")
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help="Learning rate for MLP.")
    parser.add_argument('--epochs',
                        type=int, default=100,
                        help="Number of epochs for MLP training.")
    parser.add_argument('--n_runs',
                        type=int, default=1,
                        help="Number of independent runs to average.")
    parser.add_argument('--random_state',
                        type=int, default=0,
                        help="Random seed for sampling and MLP.")
    args = parser.parse_args()

    # Load & preprocess datasets exactly as in mlp.py
    X_train_df, y_train = load_dataset(args.train_csv)
    X_test_df,  y_test  = load_dataset(args.test_csv)
    # Align columns
    X_test_df = X_test_df[X_train_df.columns]
    # Identify numeric vs categorical (binary) features
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in num_cols
                if set(X_train_df[c].unique()).issubset({0, 1})]
    num_cols = list(num_cols.difference(cat_cols))
    # Compute min/max on train numeric features
    min_val = X_train_df[num_cols].min(axis=0).values
    max_val = X_train_df[num_cols].max(axis=0).values
    # Scale numeric and keep categorical
    X_train_num = min_max_scale(X_train_df[num_cols].values, min_val, max_val)
    X_test_num  = min_max_scale(X_test_df[num_cols].values,  min_val, max_val)
    X_train_cat = X_train_df[cat_cols].values
    X_test_cat  = X_test_df[cat_cols].values
    # Final numpy arrays
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test  = np.hstack([X_test_num,  X_test_cat])

    # Sample training set
    X_tr_c, y_tr_c = random_sample(
        X_train, y_train,
        N=args.sample_size,
        random_state=args.random_state,
        match_balance=args.match_balance
    )
    # Shuffle sampled training set
    X_tr_c, y_tr_c = shuffle(X_tr_c, y_tr_c, random_state=args.random_state)

    # Prepare test DataLoader
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=args.batch_size, shuffle=False
    )

    accs = []
    for run in range(args.n_runs):
        # Prepare train DataLoader
        X_tr_tensor = torch.tensor(X_tr_c, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_tr_c, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(X_tr_tensor, y_tr_tensor),
            batch_size=args.batch_size, shuffle=True
        )

        # Instantiate model, criterion, optimizer
        input_size  = X_tr_c.shape[1]
        output_size = len(np.unique(y_train))
        model = MLP(input_size, [128, 64], output_size, dropout_prob=0.5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4
        )

        # Train
        model.train()
        for epoch in range(1, args.epochs + 1):
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(yb.cpu().tolist())
        accs.append(accuracy_score(all_labels, all_preds))

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"Over {args.n_runs} runs — Mean Accuracy: {mean_acc*100:.2f}%, Std: {std_acc*100:.2f}%")

if __name__ == '__main__':
    main()