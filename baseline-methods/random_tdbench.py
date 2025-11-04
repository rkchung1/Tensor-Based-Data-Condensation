import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Union
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

def random_sample(
    X: np.ndarray,
    y: np.ndarray,
    N: Union[int, float],
    random_state: Union[int, np.random.Generator] = 0,
    match_balance: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample N points per class (if N>1) or fraction N (if N<=1).  
    If match_balance=True, preserve class distribution.
    """
    np.random.seed(random_state)
    sampled_X, sampled_y = [], []
    y_unique, y_counts = np.unique(y, return_counts=True)

    if N >= 1:
        if match_balance:
            sample_sizes = (y_counts / y_counts.sum() * N).astype(int)
        else:
            sample_sizes = [int(N)] * len(y_unique)
    else:
        if match_balance:
            sample_sizes = (y_counts * N).astype(int)
        else:
            sample_sizes = [int(len(y) * N)] * len(y_unique)

    for label, ss in zip(y_unique, sample_sizes):
        idxs = np.where(y == label)[0]
        chosen = np.random.choice(idxs, ss, replace=False)
        sampled_X.append(X[chosen])
        sampled_y += [label] * ss

    sampled_X = np.vstack(sampled_X)
    sampled_y = np.array(sampled_y)
    return sampled_X, sampled_y

def main():
    parser = argparse.ArgumentParser(
        description="Condense train set via random sampling per class."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('-N', '--sample_size',
                        type=float,
                        default=100,
                        help="Number of samples per class (if >1) or fraction (if <=1).")
    parser.add_argument('--match_balance',
                        action='store_true',
                        help="Preserve class distribution when sampling.")
    parser.add_argument('--train_output',
                        default='train_sampled.csv',
                        help="Path to save condensed train set.")
    parser.add_argument('--random_state',
                        type=int,
                        default=0,
                        help="Random seed for sampling.")
    args = parser.parse_args()

    # Load dataset
    X_train, y_train = load_csv(args.train_csv)

    # Sample
    X_tr_s, y_tr_s = random_sample(
        X_train, y_train,
        N=args.sample_size,
        random_state=args.random_state,
        match_balance=args.match_balance
    )

    # Shuffle sampled datasets to mix classes
    X_tr_s, y_tr_s = shuffle(X_tr_s, y_tr_s, random_state=args.random_state)

    # Save sampled datasets
    save_condensed_dataset(X_tr_s, y_tr_s, args.train_output)

if __name__ == '__main__':
    main()