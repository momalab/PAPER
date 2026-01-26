import os
import pickle
import torch
import numpy as np
import random
from torchvision import datasets

"""
Constants and dataset mappings.

Attributes
    ARTIFACTS_DIR: Base directory for all results.
    LOG_DIR: Base directory for logs.
    MODEL_DIR: Base directory for models.
    DATASET_DIR: Base directory for cached datasets.
    RESULTS_LOG_DIR: Directory for result logs.
    ACCURACY_DIR: Directory for accuracy outputs.
    ENSEMBLE_DIR: Directory for ensemble outputs.
    FUSING_DIR: Directory for fusing logs.
    CLUSTER_DIR: Directory for clustering logs.
    JSON_MODEL_PATH: Directory for model metadata.
    DATASET_JSON_PATH: Directory for dataset metadata.
    COEFF_DIR: Directory for polynomial coefficients.
    num_classes_map: Dataset name to class count mapping.
    dataset_map: Dataset name to (torchvision dataset class, root path) mapping.
    CLUSTERING_SEEDS: Predefined random seeds for clustering experiments.
"""

ARTIFACTS_DIR = 'artifacts'

LOG_DIR = 'logs'
MODEL_DIR = 'models'
DATASET_DIR = 'datasets'

RESULTS_LOG_DIR = 'results_logs'
ACCURACY_DIR = 'accuracy_results'
ENSEMBLE_DIR = 'ensemble_results'
FUSING_DIR = 'fusing_logs'
CLUSTER_DIR = 'cluster_logs'

JSON_MODEL_PATH = 'model_jsons'
DATASET_JSON_PATH = 'dataset_jsons'

COEFF_DIR = 'poly_coeffs'

num_classes_map = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny': 200
}

dataset_map = {
    'cifar10': (datasets.CIFAR10, './data'),
    'cifar100': (datasets.CIFAR100, './data'),
    'tiny': (datasets.ImageFolder, './data/tiny-imagenet-200'),
}

CLUSTERING_SEEDS = [20168, 2907, 11621, 30149, 17797]

def set_seed(seed):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def save_dataset(loader, path):
    """
    Save dataset object from a DataLoader to disk.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing dataset to save.
        path (str): Path to save the serialized dataset.
    """
    with open(path, 'wb') as f:
        pickle.dump(loader.dataset, f)

def load_dataset(path):
    """
    Load a previously saved dataset from disk.

    Args:
        path (str): Path to the serialized dataset file.

    Returns:
        torch.utils.data.Dataset: Loaded dataset object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
