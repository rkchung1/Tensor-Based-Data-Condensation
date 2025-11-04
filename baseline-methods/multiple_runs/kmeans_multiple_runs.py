import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from mlp import load_dataset, min_max_scale, MLP

def get_closest_point(
    clustering: KMeans,
    points: np.ndarray,
    point_idxs: np.ndarray,
) -> np.ndarray:
    """
    For each cluster center, find the index of the closest original point.
    """
    return np.array([
        point_idxs[clustering.labels_ == l][
            ((c - points[clustering.labels_ == l])**2).sum(axis=1).argmin()
        ]
        for l, c in enumerate(clustering.cluster_centers_)
    ])

def kmeans_condense(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    get_closest: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Condense X, y by running per-class KMeans.
    If get_closest=False, use cluster centers.
    If get_closest=True, pick actual points closest to each center.
    """
    all_idxs = np.arange(len(X))
    X_out, y_out = [], []
    for cls in np.unique(y):
        Xc = X[y == cls]
        idxc = all_idxs[y == cls]
        km = KMeans(n_clusters=n_clusters,
                    random_state=random_state,
                    n_init="auto").fit(Xc)
        if get_closest:
            keep = get_closest_point(km, Xc, idxc)
            X_repr = X[keep]
        else:
            X_repr = km.cluster_centers_
        X_out.append(X_repr)
        y_out += [cls] * n_clusters

    X_condensed = np.vstack(X_out)
    y_condensed = np.array(y_out)
    return X_condensed, y_condensed

def main():
    parser = argparse.ArgumentParser(
        description="Condense via per-class KMeans and evaluate with the same MLP."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('test_csv',  help="Path to the test CSV file.")
    parser.add_argument('-K', '--n_clusters',
                        type=int,
                        default=60,
                        help="Number of clusters per class.")
    parser.add_argument('--get_closest',
                        action='store_true',
                        default=True,
                        help="Select closest points instead of centroids.")
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
                        type=int,
                        default=0,
                        help="Random seed for KMeans and MLP.")
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

    # Condense training set
    X_tr_c, y_tr_c = kmeans_condense(
        X_train, y_train,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        get_closest=args.get_closest
    )
    # Shuffle condensed training set
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