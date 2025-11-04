import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def load_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from a CSV file.
    Returns features X and labels y (or None if no label column).
    """
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
    """
    Save condensed dataset to CSV, with 'label' column first if labels are provided.
    """
    df = pd.DataFrame(X_condensed,
                      columns=[f"feature_{i}" for i in range(X_condensed.shape[1])])
    if y_condensed is not None:
        df.insert(0, 'label', y_condensed)
    df.to_csv(output_path, index=False)
    print(f"Condensed dataset saved to '{output_path}'")

def get_closest_point(
    clustering: KMeans,
    points: np.ndarray,
    point_idxs: np.ndarray,
) -> np.ndarray:
    """
    For each cluster center, find the index of the closest original point.
    """
    return [
        point_idxs[clustering.labels_ == l][
            ((c - points[clustering.labels_ == l])**2).sum(1).argmin()
        ]
        for l, c in enumerate(clustering.cluster_centers_)
    ]

def kmeans_condense(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    get_closest: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Condense X, y by running per-class KMeans.  
    If get_closest=False, use cluster centers.  
    If get_closest=True, pick actual points closest to each center.
    Implemnentation based on tdbench's kmeans.py.
    """
    all_idxs = np.arange(len(X))
    X_out, y_out = [], []
    for cls in np.unique(y):
        Xc = X[y == cls]
        idxc = all_idxs[y == cls]
        km = KMeans(n_clusters=n_clusters,
                    random_state=random_state,
                    n_init="auto").fit(Xc)
        if not get_closest:
            X_repr = km.cluster_centers_
        else:
            keep = np.array(get_closest_point(km, Xc, idxc))
            X_repr = X[keep]
        X_out.append(X_repr)
        y_out += [cls] * n_clusters

    X_condensed = np.vstack(X_out)
    y_condensed = np.array(y_out)
    return X_condensed, y_condensed

def main():
    parser = argparse.ArgumentParser(
        description="Condense train and test sets via per-class KMeans."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('-K', '--n_clusters',
                        type=int,
                        default=100,
                        help="Number of clusters per class (default: 100).")
    parser.add_argument('--get_closest',
                        action='store_true',
                        help="If set, select closest points instead of centroids.")
    parser.add_argument('--train_output',
                        default='train_condensed.csv',
                        help="Path to save condensed train set.")
    parser.add_argument('--test_output',
                        default='test_condensed.csv',
                        help="Path to save condensed test set.")
    parser.add_argument('--random_state',
                        type=int,
                        default=0,
                        help="Random seed for KMeans.")
    args = parser.parse_args()

    # Load datasets
    X_train, y_train = load_csv(args.train_csv)

    # Condense
    X_tr_c, y_tr_c = kmeans_condense(
        X_train, y_train,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        get_closest=args.get_closest
    )
    """ X_te_c, y_te_c = kmeans_condense(
        X_test, y_test,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        get_closest=args.get_closest
    ) """

    # Shuffle condensed datasets to mix classes
    X_tr_c, y_tr_c = shuffle(X_tr_c, y_tr_c, random_state=args.random_state)
    # X_te_c, y_te_c = shuffle(X_te_c, y_te_c, random_state=args.random_state)


    # Save condensed datasets
    save_condensed_dataset(X_tr_c, y_tr_c, args.train_output)
    # save_condensed_dataset(X_te_c, y_te_c, args.test_output)

if __name__ == '__main__':
    main()