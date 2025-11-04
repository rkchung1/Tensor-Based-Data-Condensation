import argparse
import numpy as np
import pandas as pd

def load_csv(csv_path: str):
    """
    Load a dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        X: Features as a NumPy array.
        labels: Labels as a NumPy array (if available), otherwise None.
    """
    data = pd.read_csv(csv_path)
    if 'label' in data.columns:
        labels = data['label'].values
        data = data.drop(columns=['label'])
    else:
        labels = None
    return data.values, labels

def load_factor_matrix(npz_path: str) -> np.ndarray:
    """
    Load the factor matrix V from a Tucker decomposition .npz file.

    Args:
        npz_path: Path to the .npz file containing the factor matrices.

    Returns:
        V: Factor matrix for mode 2, shape (original_dim x reduced_dim).
    """
    data = np.load(npz_path)
    if 'factor_2' not in data and 'V' not in data:
        raise KeyError(f"Key 'factor_2' or 'V' not found in {npz_path}. Available keys: {data.files}")
    return data['factor_2'] if 'factor_2' in data else data['V']

def project_dataset(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Project a dataset onto the reduced feature space using the factor matrix V.

    Args:
        X: Dataset features, shape (n_samples x original_dim).
        V: Factor matrix for mode 2, shape (original_dim x reduced_dim).

    Returns:
        X_projected: Projected dataset, shape (n_samples x reduced_dim).
    """
    return np.dot(X, V)

def save_projected_dataset(X_projected: np.ndarray, labels: np.ndarray, output_path: str):
    """
    Save the projected dataset to a CSV file.

    Args:
        X_projected: Projected dataset, shape (n_samples x reduced_dim).
        labels: Original labels (if available), shape (n_samples,).
        output_path: Path to save the projected dataset CSV file.
    """
    df = pd.DataFrame(X_projected, columns=[f"feature_{i}" for i in range(X_projected.shape[1])])
    if labels is not None:
        df.insert(0, 'label', labels)
    df.to_csv(output_path, index=False)
    print(f"Projected dataset saved to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Project datasets onto reduced feature space using factor matrix V.")
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('test_csv', help="Path to the test dataset CSV file.")
    parser.add_argument('factor_npz', help="Path to the Tucker decomposition .npz file containing factor matrix V.")
    parser.add_argument('--train_output', default='train_projected.csv', help="Path to save the projected training dataset.")
    parser.add_argument('--test_output', default='test_projected.csv', help="Path to save the projected test dataset.")
    args = parser.parse_args()

    # Load datasets and factor matrix
    X_train, train_labels = load_csv(args.train_csv)
    X_test, test_labels = load_csv(args.test_csv)
    V = load_factor_matrix(args.factor_npz)

    # first 10,000 samples of training for testing
    # X_train = X_train[:10000] if X_train.shape[0] > 10000 else X_train
    # train_labels = train_labels[:10000] if train_labels is not None and len(train_labels) > 10000 else train_labels

    # Project datasets onto reduced feature space
    X_train_projected = project_dataset(X_train, V)
    X_test_projected = project_dataset(X_test, V)

    # Save projected datasets
    save_projected_dataset(X_train_projected, train_labels, args.train_output)
    save_projected_dataset(X_test_projected, test_labels, args.test_output)

if __name__ == '__main__':
    main()