                                                                                                                           # scripts/2_tensorize_train.py

import os
import argparse
import numpy as np

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # training data from split step
    data = np.load(args.input)
    X, y = data["X"], data["y"]

    N, F = X.shape
    S = args.num_slices

    # 1 clean + (S - 1) noisy slices
    if S < 1:
        raise ValueError("Number of slices must be at least 1")

    slices = [X]  # First slice: clean data

    # Generate one fixed noise matrix
    noise = np.random.normal(loc=0.0, scale=args.noise_scale, size=X.shape).astype(np.float32)

    for _ in range(S - 1):
        noisy = X + noise #add the noise according to the desired noise scale.
        noisy = np.clip(noisy, 0.0, 1.0)
        slices.append(noisy)
#stack slices to create full tensor
    tensor = np.stack(slices, axis=0)  # shape: (S, N, F)

    # Save
    out_path = os.path.join(args.output_dir, "train_tensor.npz")
    np.savez(out_path, tensor=tensor, labels=y)

    print(f"Saved tensor: shape={tensor.shape}  {out_path}") #output shape of the tensor, sanity check for tensor shape.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/train2.npz", help="Path to train.npz")
    parser.add_argument("--output_dir", type=str, default="data/tensors", help="Where to save 3D tensor")
    parser.add_argument("--num_slices", type=int, default=5, help="Total slices (1 clean + N-1 same-noise)")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Gaussian noise stddev (default: 0.1)")
    args = parser.parse_args()

    main(args)

