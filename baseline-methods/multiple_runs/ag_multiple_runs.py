import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from typing import Tuple, Union
from mlp import load_dataset, min_max_scale, MLP

def get_closest_point(
    clustering: AgglomerativeClustering,
    points: np.ndarray,
    point_idxs: np.ndarray,
) -> np.ndarray:
    centroids = NearestCentroid().fit(points, clustering.labels_)
    return np.array([
        point_idxs[clustering.labels_ == l][
            ((c - points[clustering.labels_ == l])**2).sum(axis=1).argmin()
        ]
        for l, c in enumerate(centroids.centroids_)
    ])

def agglomerative(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    random_state: Union[int, np.random.Generator] = 0,
    get_closest: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    clustered_X, clustered_y, clustered_idxs = [], [], []
    all_idxs = np.arange(len(X))
    for label in np.unique(y):
        pts = X[y == label]
        if len(pts) < N:
            print(f"Warning: Requested {N} clusters for class {label}, but only {len(pts)} points available.")
            n = len(pts)
        else:
            n = N
        if n == 1:
            clustered_X.append(pts.mean(axis=0).reshape(1, -1))
            clustered_y.append(label)
            continue
        clustering = AgglomerativeClustering(n_clusters=n).fit(pts)
        if get_closest:
            idxs = get_closest_point(clustering, pts, all_idxs[y == label])
            clustered_idxs.append(idxs)
            clustered_X.append(X[idxs])
        else:
            cent = NearestCentroid().fit(pts, clustering.labels_)
            clustered_X.append(cent.centroids_)
        clustered_y += [label] * n

    if get_closest:
        clustered_idxs = np.hstack(clustered_idxs)
    else:
        clustered_idxs = None

    clustered_X = np.vstack(clustered_X)
    clustered_y = np.array(clustered_y)
    return clustered_X, clustered_y, clustered_idxs

def main():
    parser = argparse.ArgumentParser(
        description="Condense via per-class Agglomerative Clustering and evaluate with the same MLP."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('test_csv',  help="Path to the test CSV file.")
    parser.add_argument('-K', '--n_clusters',
                        type=int,
                        default=60,
                        help="Number of clusters per class.")
    parser.add_argument('--get_closest',
                        action='store_true',
                        help="Select closest points to centroids instead of centroids.")
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
                        help="Random seed for clustering and MLP.")
    args = parser.parse_args()

    # Load & preprocess datasets exactly as in mlp.py
    X_train_df, y_train = load_dataset(args.train_csv)
    X_test_df,  y_test  = load_dataset(args.test_csv)
    X_test_df = X_test_df[X_train_df.columns]
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in num_cols if set(X_train_df[c].unique()).issubset({0, 1})]
    num_cols = list(num_cols.difference(cat_cols))
    min_val = X_train_df[num_cols].min(axis=0).values
    max_val = X_train_df[num_cols].max(axis=0).values
    X_train_num = min_max_scale(X_train_df[num_cols].values, min_val, max_val)
    X_test_num  = min_max_scale(X_test_df[num_cols].values,  min_val, max_val)
    X_train_cat = X_train_df[cat_cols].values
    X_test_cat  = X_test_df[cat_cols].values
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test  = np.hstack([X_test_num,  X_test_cat])

    # Condense training set
    X_tr_c, y_tr_c, _ = agglomerative(
        X_train, y_train,
        N=args.n_clusters,
        random_state=args.random_state,
        get_closest=args.get_closest
    )
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