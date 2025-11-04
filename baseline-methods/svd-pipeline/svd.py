import argparse
import numpy as np
import pandas as pd

def load_csv(csv_path: str) -> np.ndarray:
    """
    Load a dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        X: Features as a NumPy array.
    """
    data = pd.read_csv(csv_path)
    if 'label' in data.columns:
        data = data.drop(columns=['label'])  # Drop the label column if it exists
    return data.values

def perform_svd(X: np.ndarray, rank: int):
    """
    Perform Singular Value Decomposition (SVD) on the input matrix.

    Args:
        X: Input matrix, shape (n_samples x n_features).
        rank: Number of singular values/vectors to keep.

    Returns:
        U: Left singular vectors, shape (n_samples x rank).
        S: Singular values, shape (rank,).
        V: Right singular vectors, shape (n_features x rank).
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U_reduced = U[:, :rank]
    S_reduced = S[:rank]
    V_reduced = Vt[:rank, :].T  # Transpose Vt to get V
    return U_reduced, S_reduced, V_reduced

def save_svd_results(U: np.ndarray, S: np.ndarray, V: np.ndarray, output_path: str):
    """
    Save the SVD results (U, S, V) to an .npz file.

    Args:
        U: Left singular vectors, shape (n_samples x rank).
        S: Singular values, shape (rank,).
        V: Right singular vectors, shape (n_features x rank).
        output_path: Path to save the .npz file.
    """
    np.savez(output_path, U=U, S=S, V=V)
    print(f"SVD results saved to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Perform Singular Value Decomposition (SVD) on a dataset from a CSV file."
    )
    parser.add_argument(
        'input_csv',
        help="Path to the input CSV file."
    )
    parser.add_argument(
        '--rank',
        type=int,
        required=True,
        help="Number of singular values/vectors to keep."
    )
    parser.add_argument(
        '-o', '--output',
        default='svd_result.npz',
        help="Output .npz file to save U, S, and V (default: svd_result.npz)."
    )
    args = parser.parse_args()

    # Load dataset
    X = load_csv(args.input_csv)
    # first 10,000 samples for testing
    # X = X[:10000] if X.shape[0] > 10000 else X
    print(f"Loaded dataset with shape {X.shape}")

    # Perform SVD
    print(f"Performing SVD with rank {args.rank}...")
    U, S, V = perform_svd(X, args.rank)

    # Print results
    print(f"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
    print("Singular values:")
    print(S)

    # Save results
    save_svd_results(U, S, V, args.output)

if __name__ == '__main__':
    main()

# Example usage:
# python svd.py \
#     fashion-mnist_train.csv \
#     --rank 50 \
#     -o fashion-mnist_svd_50.npz