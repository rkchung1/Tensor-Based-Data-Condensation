import argparse
import copy
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from typing import Tuple, Union
from sklearn.utils import shuffle

def load_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
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

# xavier uniform with controlled random generator
def xavier_uniform(tensor, gain=1.0, generator=None):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    a = np.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-a, a, generator=generator)

def init_weights(m, gain=1.0, generator=None):
    if isinstance(m, nn.Linear):
        xavier_uniform(m.weight, gain=gain, generator=generator)
        m.bias.data.fill_(0.01)

# A simple MLP class
class MLP(nn.Module):
    def __init__(
        self,
        input_shape: int,
        n_hidden_layers: int,
        hidden_dim: int,
        n_labels: int,
    ):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(n_hidden_layers)
            ],
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_labels),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.classifier(self.embedder(x))

def get_mlp(
    input_shape: int,
    n_hidden_layers: int,
    hidden_dim: int,
    n_labels: int,
    random_state: int,
):
    """
    Construct an MLP and apply weight initialization.
    """
    gen = torch.Generator("cpu")
    gen.manual_seed(random_state)
    initializer = partial(init_weights, gain=1.0, generator=gen)

    mlp = MLP(
        input_shape=input_shape,
        n_hidden_layers=n_hidden_layers,
        hidden_dim=hidden_dim,
        n_labels=n_labels,
    ).apply(initializer)
    return mlp

def gradient_matching(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    n_epochs: int,
    mlp_dim: int,
    lr_mlp: float,
    lr_data: float,
    mom_data: float,
    n_hidden_layers: int,
    random_state: int | np.random.Generator = 0,
):
    """
    Perform gradient matching-based dataset condensation.
    Original code from https://github.com/GeorgeCazenavette/mtt-distillation
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_labels = np.unique(y).shape[0]
    X_real = torch.tensor(X).float().to(device)
    y_real = torch.tensor(y).long().to(device)

    idxs = np.arange(len(X))
    support_idxs, _ = random_sample(idxs, y, N, random_state=random_state)
    support_idxs = support_idxs.flatten()
    X_syn = torch.tensor(X[support_idxs]).float().to(device)
    y_syn = torch.tensor(y[support_idxs]).long().to(device)

    model = get_mlp(
        input_shape=X.shape[1],
        n_hidden_layers=n_hidden_layers,
        hidden_dim=mlp_dim,
        n_labels=n_labels,
        random_state=random_state,
    ).to(device)

    opt_model = optim.SGD(model.parameters(), lr=lr_mlp)
    opt_data = optim.SGD([X_syn], lr=lr_data, momentum=mom_data)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        loss_gw = torch.tensor(0.0, device=device)
        for label in np.unique(y):
            o_syn = model(X_syn[y_syn == label])
            o_real = model(X_real[y_real == label])

            l_syn = criterion(o_syn, y_syn[y_syn == label])
            l_real = criterion(o_real, y_real[y_real == label])

            gw_syn = torch.autograd.grad(l_syn, model.parameters(), create_graph=True)
            gw_real = (
                g.detach().clone()
                for g in torch.autograd.grad(l_real, model.parameters())
            )

            loss_gw += sum(
                torch.sum(
                    1
                    - (
                        torch.sum(gwr * gws, dim=-1)
                        / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 1e-6)
                    )
                )
                for gwr, gws in zip(gw_real, gw_syn)
            )

        opt_data.zero_grad()
        loss_gw.backward()
        opt_data.step()

        # update model on frozen synthetic data
        X_syn_frozen = copy.deepcopy(X_syn)
        y_syn_frozen = copy.deepcopy(y_syn)
        out = model(X_syn_frozen)
        loss_model = criterion(out, y_syn_frozen)
        opt_model.zero_grad()
        loss_model.backward()
        opt_model.step()

    return X_syn_frozen.cpu().detach().numpy(), y_syn_frozen.cpu().detach().numpy()

def main():
    # Default parameters come from GM implementation by Zhao et al. (2021)
    parser = argparse.ArgumentParser(
        description="Condense train set via Gradient Matching."
    )
    parser.add_argument('train_csv', help="Path to the training CSV file.")
    parser.add_argument('-N', '--sample_size',
                        type=int,
                        default=100,
                        help="Number of synthetic samples per class.")
    parser.add_argument('--n_epochs',
                        type=int,
                        default=500,
                        help="Number of epochs for gradient matching.")
    parser.add_argument('--mlp_dim',
                        type=int,
                        default=1024,
                        help="Hidden layer dimension for MLP.")
    parser.add_argument('--lr_mlp',
                        type=float,
                        default=0.01,
                        help="Learning rate for MLP.")
    parser.add_argument('--lr_data',
                        type=float,
                        default=0.1,
                        help="Learning rate for synthetic data.")
    parser.add_argument('--mom_data',
                        type=float,
                        default=0.5,
                        help="Momentum for synthetic data optimizer.")
    parser.add_argument('--n_hidden_layers',
                        type=int,
                        default=2,
                        help="Number of hidden layers in MLP.")
    parser.add_argument('--train_output',
                        default='train_gradient_matched.csv',
                        help="Path to save condensed train set.")
    parser.add_argument('--random_state',
                        type=int,
                        default=0,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    X_train, y_train = load_csv(args.train_csv)
    X_tr_gm, y_tr_gm = gradient_matching(
        X_train, y_train,
        N=args.sample_size,
        n_epochs=args.n_epochs,
        mlp_dim=args.mlp_dim,
        lr_mlp=args.lr_mlp,
        lr_data=args.lr_data,
        mom_data=args.mom_data,
        n_hidden_layers=args.n_hidden_layers,
        random_state=args.random_state
    )

    X_tr_gm, y_tr_gm = shuffle(X_tr_gm, y_tr_gm, random_state=args.random_state)
    save_condensed_dataset(X_tr_gm, y_tr_gm, args.train_output)

if __name__ == '__main__':
    main()