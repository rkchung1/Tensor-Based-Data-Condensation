import os
import argparse
import numpy as np
import torch
import tensorly as tl

#Randomized SVD to allow for GPU usage and larger tensors.
def randomized_svd_gpu(matrix, rank, n_iter=3):
    m, n = matrix.shape
    omega = torch.randn(n, rank + 10, device=matrix.device, dtype=matrix.dtype)
    Q = matrix @ omega
    for _ in range(n_iter):
        Q = matrix.T @ Q
        Q, _ = torch.linalg.qr(Q)
        Q = matrix @ Q
        Q, _ = torch.linalg.qr(Q)
    B = Q.T @ matrix
    U_tilde, S, V = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U[:, :rank], S[:rank], V[:rank] 

#implementation of 2 way tucker. Did not use Tucker function from tensorly because it does not support GPU usage
def two_way_tucker(tensor, rank_samples, rank_features, device):
    tensor = torch.from_numpy(tensor).float().to(device)

    # Mode 1: samples
    unfolded_1 = tensor.permute(1, 0, 2).reshape(tensor.shape[1], -1)
    U_samples, _, _ = randomized_svd_gpu(unfolded_1, rank_samples)
    del unfolded_1; torch.cuda.empty_cache()

    # Mode 2: features
    unfolded_2 = tensor.permute(2, 0, 1).reshape(tensor.shape[2], -1)
    U_features, _, _ = randomized_svd_gpu(unfolded_2, rank_features)
    del unfolded_2; torch.cuda.empty_cache()

    # Core tensor
    core_temp = torch.einsum('ijk,jr->irk', tensor, U_samples)
    core_tensor = torch.einsum('irk,km->irm', core_temp, U_features)

    return core_tensor, U_samples, U_features

def main(args):
    tl.set_backend('pytorch')
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu') #attempt to send to gpu, otherwise use cpu
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tensor
    data = np.load(args.input)
    tensor = data["tensor"]
    labels = data["labels"]
    print(f"Tensor shape: {tensor.shape}") #sanity check for tensor shape

    # Decompose
    print(f"Decomposing to ranks: samples={args.rank_samples}, features={args.rank_features}")
    core, U_samples, U_features = two_way_tucker(tensor, args.rank_samples, args.rank_features, device)

    # Save the output in file for next step
    output_path = os.path.join(args.output_dir, "tucker_train2.npz")
    np.savez(output_path,
             core=core.cpu().numpy(),
             U_samples=U_samples.cpu().numpy(),
             U_features=U_features.cpu().numpy(),
             labels=labels)
    print(f"Saved Tucker output to {output_path}")
    print(f"Core shape: {core.shape}")
    print(f"U_samples shape: {U_samples.shape}")
    print(f"U_features shape: {U_features.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/tensors/train_tensor2.npz", help="Path to input tensor .npz")
    parser.add_argument("--output_dir", type=str, default="data/decomposed", help="Directory to save results")
    parser.add_argument("--rank_samples", type=int, default=1000, help="Rank for sample mode")
    parser.add_argument("--rank_features", type=int, default=100, help="Rank for feature mode")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index to use (default: 0)")
    args = parser.parse_args()
    main(args)
