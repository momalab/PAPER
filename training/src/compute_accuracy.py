import logging
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from arguments import parse_arguments
from logger import init_logging, setup_paths, get_model_save_path
from models import get_model, replace_model
from poly_activation import PolyActivation
from trainer import test
from utils import ACCURACY_DIR, ARTIFACTS_DIR, DATASET_DIR, RESULTS_LOG_DIR, load_dataset


def build_model(args):
    """
    Build the model and optionally replace ReLU with a polynomial activation.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            model (str): Model architecture name.
            degree (int): Degree of the polynomial.
            clip (int or float): Activation clipping threshold.
            pbit (int): Precision bits used in coefficient generation.

    Returns:
        torch.nn.Module: Instantiated model ready for evaluation.
    """
    model = get_model(args).cuda()
    if "poly" in args.model:
        poly_activation = PolyActivation(args.degree, args.clip, args.pbit)
        replace_model(model, poly_activation)
    return model


def build_log_file(args):
    """
    Build the accuracy log file path based on configuration.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            dataset (str): Dataset name.
            model (str): Model architecture name.
            degree (int): Degree of the polynomial.
            clip (int or float): Activation clipping threshold.
            pbit (int): Precision bits used in coefficient generation.
            penalty (float): Penalty coefficient for polynomial activation.
            sigma (float): Standard deviation for Gaussian noise to the activations.

    Returns:
        str: Absolute path where the accuracy log will be written
    """
    name = (
        f"accuracy_summary"
        if "poly" not in args.model
        else (
            f"accuracy_summary_degree_{args.degree}"
            f"_clip_{args.clip}_pbit_{args.pbit}_zeta_{args.penalty}_sigma_{args.sigma}"
        )
    )
    path = os.path.join(ARTIFACTS_DIR, RESULTS_LOG_DIR, ACCURACY_DIR, f"{args.dataset}_{args.model}", f"{name}.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def evaluate_seeds(model, model_path, dataset_path, batch_size, seeds):
    """
    Evaluate checkpoints produced by a list of seeds.

    Args:
        model (torch.nn.Module): Model to reuse for all seeds
        model_path (str): Directory containing checkpoints
        dataset_path (str): Directory containing saved test splits per seed
        batch_size (int): Batch size for evaluation
        seeds (list[int]): List of seeds to evaluate

    Returns:
        tuple (list[float], str): accuracies for all seeds, best model path
    """
    accuracies = []
    best_acc = -1.0
    best_model_path = ""

    for seed in seeds:
        ckpt = get_model_save_path(model_path, seed)
        model.load_state_dict(torch.load(ckpt, weights_only=True))

        testdataset_path = os.path.join(dataset_path, f"testdataset_seed_{seed}.pkl")
        test_dataset = load_dataset(testdataset_path)
        testloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        test_acc = test(model, testloader)
        accuracies.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = ckpt

    return accuracies, best_model_path


def main():
    """
    Compute and log accuracy statistics over multiple seeds.
    """
    args = parse_arguments()
    _, model_path = setup_paths(args)

    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, args.dataset)
    model = build_model(args)

    log_file = build_log_file(args)
    init_logging(log_file)

    accuracies, best_model_path = evaluate_seeds(
        model=model,
        model_path=model_path,
        dataset_path=dataset_path,
        batch_size=args.batch_size_test,
        seeds=args.seed_list
    )
    
    accuracies = np.array(accuracies)
    logging.info("Accuracy statistics:")
    logging.info(f"Accuracies: {accuracies}")
    logging.info(
        f"Mean: {accuracies.mean():.4f}, Std: {accuracies.std():.4f}, "
        f"Min: {accuracies.min():.2f}, Max: {accuracies.max():.2f}"
    )
    logging.info(f"Accuracy (mean, std): ({accuracies.mean():.4f}, {accuracies.std():.4f})")
    logging.info(f"Best model path: {best_model_path}")

if __name__ == "__main__":
    main()
