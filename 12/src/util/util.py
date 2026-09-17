import ast
import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from tqdm import tqdm

JST = ZoneInfo("Asia/Tokyo")


def _initialize_weight(module, initializer):
    if isinstance(module, (nn.Embedding, nn.Linear)):
        initializer(module.weight.data)
        if isinstance(module, nn.Linear) and module.bias is not None:
            constant_(module.bias.data, 0)


def initialize_with_xavier_normal(module):
    _initialize_weight(module, xavier_normal_)


def initialize_with_xavier_uniform(module):
    _initialize_weight(module, xavier_uniform_)


def is_train_stop(val, best_val, current_epoch, early_stop_num):
    """Return early-stopping state after comparing the current score."""
    if val > best_val:
        return False, 0, val

    current_epoch += 1
    is_stop = current_epoch >= early_stop_num
    return is_stop, current_epoch, best_val


def convert_to_num(arg):
    if arg == "inf":
        return float(arg)

    try:
        return ast.literal_eval(arg)
    except ValueError, SyntaxError:
        return arg


def get_yyyymmddhhmmss():
    return datetime.now(tz=JST).strftime("%Y%m%d%H%M%S")


def create_dir(dir_name):
    Path(dir_name).mkdir(parents=True, exist_ok=True)


def get_tqdm(iterable, desc=None):
    return tqdm(iterable, total=len(iterable), ncols=100, desc=desc)


def _get_laplacian_eigendecomposition(laplacian, symmetry=True):
    if symmetry:
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(laplacian)
    return eigenvalues, eigenvectors


def get_l_p_of_not_normalized_A(A, device, symmetry=True):
    degree = torch.diag(torch.sum(A, dim=1)).to(torch.float)
    safe_degree = torch.where(degree == 0, torch.ones_like(degree), degree)
    degree_inv_sqrt = torch.pow(safe_degree, -1 / 2) - torch.where(
        degree == 0, torch.ones_like(degree), 0
    )

    A = A.to(device)
    degree_inv_sqrt = degree_inv_sqrt.to(device)

    normalized_adjacency = degree_inv_sqrt @ A.to(torch.float) @ degree_inv_sqrt
    normalized_laplacian = (
        torch.eye(normalized_adjacency.shape[0], device=device)
        - normalized_adjacency
    )
    eigenvalues, eigenvectors = _get_laplacian_eigendecomposition(
        normalized_laplacian, symmetry=symmetry
    )

    return normalized_adjacency, normalized_laplacian, eigenvalues, eigenvectors


def get_l_p_of_normalized_A(A, device, symmetry=True):
    laplacian = torch.eye(A.shape[0], device=device) - A
    eigenvalues, eigenvectors = _get_laplacian_eigendecomposition(
        laplacian, symmetry=symmetry
    )
    return A, laplacian, eigenvalues, eigenvectors


def save_adj_matrix(adj_matrix, result_dir, model_name, dataset_name, nnz):
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{get_yyyymmddhhmmss()}_{model_name}_{dataset_name}_nnz_{nnz}_adj_matrix.pkl"
    with open(result_path / file_name, mode="wb") as f:
        pickle.dump(adj_matrix, f)
