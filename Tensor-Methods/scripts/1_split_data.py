# scripts/1_split_data.py
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

#split the dataset based on test_size specification, then save it
def split_and_save(X, y, out_dir, test_size, random_state):
    os.makedirs(out_dir, exist_ok=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if len(np.unique(y)) > 1 else None,
        random_state=random_state
    )
    train_path = os.path.join(out_dir, "train.npz") #these files will be used later for testing and training the mlp, and for tensor methods.
    test_path  = os.path.join(out_dir, "test.npz")
    np.savez(train_path, X=X_tr, y=y_tr)
    np.savez(test_path,  X=X_te, y=y_te)
    return train_path, test_path, {
        "n_train": int(X_tr.shape[0]),
        "n_test":  int(X_te.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts_train": {int(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))}, #i'm outputting the splits so I can make sure the classes are correct, and that I am using the expected dataset.
        "class_counts_test":  {int(k): int(v) for k, v in zip(*np.unique(y_te, return_counts=True))}
    }

def main():
    p = argparse.ArgumentParser(description="Split dataset into train/test.")
    p.add_argument("--input", type=str, required=True, help="Path to input .npz containing X and y")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save train/test .npz")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split proportion") #safety; split to 20% just in case.
    p.add_argument("--random_state", type=int, default=42, help="Random seed") #randomly select the seed for the split.
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] {args.input}")
    data = np.load(args.input)
    X, y = data["X"], data["y"]

    train_path, test_path, stats = split_and_save(X, y, args.output_dir, args.test_size, args.random_state)

    print(f"[OK] Saved splits to: {args.output_dir}")
    print(f"[OK] Train: {train_path}")
    print(f"[OK] Test : {test_path}")
    print(f"[info] Stats: {stats}")
    print(f"[TIP] Next: 2_tensorize_train.py --input {train_path} --output_dir data/tensors --num_slices 5 --noise_scale 0.1")

if __name__ == "__main__":
    main()
