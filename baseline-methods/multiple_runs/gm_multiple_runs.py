import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from mlp import load_dataset, min_max_scale, MLP
from gm_tdbench import gradient_matching

def main():
    parser = argparse.ArgumentParser(
        description="Condense via Gradient Matching and evaluate with the same MLP."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('test_csv',  help="Path to the test CSV file.")
    parser.add_argument('-N', '--sample_size',
                        type=int, default=60,
                        help="Number of synthetic samples per class.")
    parser.add_argument('--gm_epochs',
                        type=int, default=500,
                        help="Number of epochs for gradient matching.")
    parser.add_argument('--mlp_dim',
                        type=int, default=1024,
                        help="Hidden layer dimension for GM MLP.")
    parser.add_argument('--lr_gm_mlp',
                        type=float, default=0.01,
                        help="Learning rate for GM internal MLP.")
    parser.add_argument('--lr_data',
                        type=float, default=0.1,
                        help="Learning rate for synthetic data.")
    parser.add_argument('--mom_data',
                        type=float, default=0.5,
                        help="Momentum for synthetic data optimizer.")
    parser.add_argument('--n_hidden_layers',
                        type=int, default=2,
                        help="Number of hidden layers in GM MLP.")
    parser.add_argument('--batch_size',
                        type=int, default=64,
                        help="Batch size for evaluation MLP.")
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help="Learning rate for evaluation MLP.")
    parser.add_argument('--epochs',
                        type=int, default=100,
                        help="Number of epochs for evaluation MLP.")
    parser.add_argument('--n_runs',
                        type=int, default=1,
                        help="Number of independent evaluation runs.")
    parser.add_argument('--random_state',
                        type=int, default=0,
                        help="Random seed for GM and MLP.")
    args = parser.parse_args()

    # Load & preprocess exactly as in mlp.py
    X_train_df, y_train = load_dataset(args.train_csv)
    X_test_df,  y_test  = load_dataset(args.test_csv)
    X_test_df = X_test_df[X_train_df.columns]
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in num_cols
                if set(X_train_df[c].unique()).issubset({0, 1})]
    num_cols = list(num_cols.difference(cat_cols))
    min_val = X_train_df[num_cols].min(axis=0).values
    max_val = X_train_df[num_cols].max(axis=0).values
    X_train_num = min_max_scale(X_train_df[num_cols].values, min_val, max_val)
    X_test_num  = min_max_scale(X_test_df[num_cols].values,  min_val, max_val)
    X_train_cat = X_train_df[cat_cols].values
    X_test_cat  = X_test_df[cat_cols].values
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test  = np.hstack([X_test_num,  X_test_cat])

    # Condense via gradient matching
    X_syn, y_syn = gradient_matching(
        X_train, y_train,
        N=args.sample_size,
        n_epochs=args.gm_epochs,
        mlp_dim=args.mlp_dim,
        lr_mlp=args.lr_gm_mlp,
        lr_data=args.lr_data,
        mom_data=args.mom_data,
        n_hidden_layers=args.n_hidden_layers,
        random_state=args.random_state
    )
    X_syn, y_syn = shuffle(X_syn, y_syn, random_state=args.random_state)

    # Prepare test DataLoader
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=args.batch_size, shuffle=False
    )

    # Evaluate with the same MLP architecture
    accs = []
    for run in range(args.n_runs):
        # Train on synthetic data
        X_tr_tensor = torch.tensor(X_syn, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_syn, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(X_tr_tensor, y_tr_tensor),
            batch_size=args.batch_size, shuffle=True
        )

        input_size  = X_syn.shape[1]
        output_size = len(np.unique(y_train))
        model = MLP(input_size, [128, 64], output_size, dropout_prob=0.5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4
        )

        model.train()
        for epoch in range(1, args.epochs + 1):
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

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