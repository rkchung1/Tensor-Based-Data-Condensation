import argparse
import functools
import numpy as np
import pandas as pd
from typing import Tuple, Union
from sklearn.utils import shuffle

# JAX / Neural Tangents imports
from jax.example_libraries import optimizers
import jax
# import jax.config
from jax import numpy as jnp
from jax import scipy as sp
from neural_tangents import stax

# Enable 64-bit in JAX for stability
jax.config.update("jax_enable_x64", True)

def load_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from a CSV file.
    Returns features X and labels y.
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
    df = pd.DataFrame(
        X_condensed,
        columns=[f"feature_{i}" for i in range(X_condensed.shape[1])]
    )
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

def FullyConnectedNetwork(
    hidden_dims,
    out_dim,
    W_std=np.sqrt(2),
    b_std=0.1,
    parameterization="ntk",
):
    """Returns neural_tangents.stax fully connected network."""
    activation_fn = stax.Relu()
    dense = functools.partial(
        stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization
    )
    layers = []
    for hidden in hidden_dims:
        layers += [dense(hidden), activation_fn]
    layers += [
        stax.Dense(out_dim, W_std=W_std, b_std=b_std, parameterization=parameterization)
    ]
    return stax.serial(*layers)

def get_kernel_fn(hidden_dims, out_dim):
    _, _, _kernel_fn = FullyConnectedNetwork(hidden_dims, out_dim)
    return jax.jit(functools.partial(_kernel_fn, get="ntk"))

def get_loss_fn(kernel_fn):
    @jax.jit
    def loss_fn(x_support, y_support, x_target, y_target, reg=1e-6):
        k_ss = kernel_fn(x_support, x_support)
        k_ts = kernel_fn(x_target, x_support)
        k_ss_reg = (
            k_ss
            + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0]
        )
        pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, assume_a="pos"))
        mse_loss = 0.5 * jnp.mean((pred - y_target) ** 2)
        acc = jnp.mean(jnp.argmax(pred, axis=1) == jnp.argmax(y_target, axis=1))
        return mse_loss, acc
    return loss_fn

def get_update_functions(init_params, kernel_fn, lr):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(init_params)
    loss_fn = get_loss_fn(kernel_fn)
    value_and_grad = jax.value_and_grad(
        lambda params, x_target, y_target: loss_fn(
            params["x"], params["y"], x_target, y_target
        ),
        has_aux=True,
    )
    @jax.jit
    def update_fn(step, opt_state, params, x_target, y_target):
        (loss, acc), dparams = value_and_grad(params, x_target, y_target)
        return opt_update(step, dparams, opt_state), (loss, acc)
    return opt_state, get_params, update_fn

def convert_onehot(labels: np.ndarray) -> np.ndarray:
    converted = np.zeros((len(labels), len(set(labels))))
    converted[np.arange(len(labels)), labels] = 1
    return converted

def kip(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    n_epochs: int,
    mlp_dim: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform KIP-based dataset condensation.
    Source: https://github.com/google-research/google-research/tree/master/kip
    """
    idxs = np.arange(len(X))
    X = X.astype(float)
    y_onehot = convert_onehot(y).astype(float)

    support_idxs, _ = random_sample(idxs, y, N, random_state=random_state)
    support_idxs = support_idxs.flatten()
    x_support = X[support_idxs].copy()
    y_support = y_onehot[support_idxs].copy()

    params_init = {"x": x_support, "y": y_support}
    kernel_fn = get_kernel_fn([mlp_dim], y_onehot.shape[1])
    opt_state, get_params, update_fn = get_update_functions(
        params_init, kernel_fn, lr=4e-2
    )
    params = get_params(opt_state)

    for i in range(n_epochs):
        target_idxs, _ = random_sample(
            idxs, y, N * 10, random_state=random_state
        )
        target_idxs = target_idxs.flatten()
        x_target = X[target_idxs]
        y_target = y_onehot[target_idxs]
        opt_state, _ = update_fn(i + 1, opt_state, params, x_target, y_target)
        params = get_params(opt_state)

    X_syn = np.array(params["x"])
    y_syn = params_init["y"].argmax(1)
    return X_syn, y_syn

def main():
    parser = argparse.ArgumentParser(
        description="Condense train set via Kernel Inducing Points (KIP)."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('-N', '--sample_size',
                        type=int,
                        default=100,
                        help="Number of synthetic samples per class.")
    parser.add_argument('--n_epochs',
                        type=int,
                        default=1000,
                        help="Number of optimization epochs for KIP.")
    parser.add_argument('--mlp_dim',
                        type=int,
                        default=1024,
                        help="Hidden dimension for kernel network.")
    parser.add_argument('--train_output',
                        default='train_kip.csv',
                        help="Path to save condensed train set.")
    parser.add_argument('--random_state',
                        type=int,
                        default=0,
                        help="Random seed for sampling.")
    args = parser.parse_args()

    X_train, y_train = load_csv(args.train_csv)
    X_tr_kip, y_tr_kip = kip(
        X_train, y_train,
        N=args.sample_size,
        n_epochs=args.n_epochs,
        mlp_dim=args.mlp_dim,
        random_state=args.random_state
    )

    X_tr_kip, y_tr_kip = shuffle(X_tr_kip, y_tr_kip, random_state=args.random_state)
    save_condensed_dataset(X_tr_kip, y_tr_kip, args.train_output)

if __name__ == '__main__':
    main()