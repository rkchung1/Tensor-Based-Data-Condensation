# scripts/4_project_test_features.py

import os
import argparse
import numpy as np

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    test_data = np.load(args.test_input)
    X_test = test_data["X"]
    y_test = test_data["y"]

    # Load U_features, from Tucker 
    tucker = np.load(args.decomposed_input)
    U_features = tucker["U_features"]  # shape: (original_dim, r2)

    # Project test set
    X_test_proj = X_test @ U_features  # shape: (num_test_samples, r2)

    # Save compressed test set
    out_path = os.path.join(args.output_dir, "test_compressed2.npz")
    np.savez(out_path, X=X_test_proj, y=y_test)

    print(f"Saved compressed test set to {out_path}")
    print(f"Shape: {X_test_proj.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_input", type=str, default="data/processed/test.npz", help="Path to test.npz")
    parser.add_argument("--decomposed_input", type=str, default="data/decomposed/tucker_train2.npz", help="Path to Tucker output .npz")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Where to save projected test set")
    args = parser.parse_args()

    main(args)
