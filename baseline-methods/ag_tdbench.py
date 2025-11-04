import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Union
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.utils import shuffle

def load_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if 'label' in df.columns:
        y = df['label'].values
        X = df.drop(columns=['label']).values
    else:
        y = None
        X = df.values
    return X, y

def save_condensed_dataset(
    X_condensed: np.ndarray,
    y_condensed: np.ndarray,
    output_path: str
):
    df = pd.DataFrame(X_condensed,
                      columns=[f"feature_{i}" for i in range(X_condensed.shape[1])])
    if y_condensed is not None:
        df.insert(0, 'label', y_condensed)
    df.to_csv(output_path, index=False)
    print(f"Condensed dataset saved to '{output_path}'")

def get_closest_point(
    clustering: AgglomerativeClustering,
    points: np.ndarray,
    point_idxs: np.ndarray,
) -> np.ndarray:
    centroids = NearestCentroid().fit(points, clustering.labels_)
    return [
        point_idxs[clustering.labels_ == l][
            ((c - points[clustering.labels_ == l]) ** 2).sum(1).argmin()
        ]
        for l, c in enumerate(centroids.centroids_)
    ]

def agglomerative(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    random_state: Union[int, np.random.Generator] = 0,
    match_balance: bool = False,
    get_closest: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    clustered_X, clustered_y, clustered_idxs = [], [], []
    all_idxs = np.arange(len(X))
    for label in np.unique(y):
        pts = X[y == label]
        if len(pts) < N:  # Handle case where N exceeds the number of points
            print(f"Warning: Requested {N} clusters for class {label}, but only {len(pts)} points available.")
            N = len(pts)
        if N == 1:  # Special case for 1 cluster
            clustered_X.append(pts.mean(axis=0).reshape(1, -1))  # Use mean as the centroid
            clustered_y.append(label)
            continue
        clustering = AgglomerativeClustering(n_clusters=N).fit(pts)
        if get_closest:
            idxs = get_closest_point(
                clustering=clustering,
                points=pts,
                point_idxs=all_idxs[y == label],
            )
            clustered_idxs.append(idxs)
            clustered_X.append(X[idxs])
        else:
            cent = NearestCentroid().fit(pts, clustering.labels_)
            clustered_X.append(cent.centroids_)
        clustered_y += [label] * N

    if get_closest:
        clustered_idxs = np.hstack(clustered_idxs)
    else:
        clustered_idxs = None

    clustered_X = np.vstack(clustered_X)
    clustered_y = np.array(clustered_y)
    return clustered_X, clustered_y, clustered_idxs

def main():
    parser = argparse.ArgumentParser(
        description="Condense train set via per-class Agglomerative Clustering."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('-N', '--n_clusters',
                        type=int,
                        default=100,
                        help="Number of clusters per class.")
    parser.add_argument('--get_closest',
                        action='store_true',
                        help="Select closest points to centroids instead of centroids.")
    parser.add_argument('--train_output',
                        default='train_agglomerated.csv',
                        help="Path to save condensed train set.")
    parser.add_argument('--random_state',
                        type=int,
                        default=0,
                        help="Random seed (unused by AgglomerativeClustering).")
    parser.add_argument('--match_balance',
                        action='store_true',
                        help="(Unused) preserve class distribution when clustering.")
    args = parser.parse_args()

    X_train, y_train = load_csv(args.train_csv)
    X_tr_a, y_tr_a, _ = agglomerative(
        X_train, y_train,
        N=args.n_clusters,
        random_state=args.random_state,
        match_balance=args.match_balance,
        get_closest=args.get_closest
    )

    X_tr_a, y_tr_a = shuffle(X_tr_a, y_tr_a, random_state=args.random_state)
    save_condensed_dataset(X_tr_a, y_tr_a, args.train_output)

if __name__ == '__main__':
    main()