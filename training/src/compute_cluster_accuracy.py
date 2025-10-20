import os
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from arguments import parse_arguments
from logger import get_model_save_path, init_logging, setup_paths
from models import get_model, replace_model
from poly_activation import PolyActivation
from trainer import test
from utils import ARTIFACTS_DIR, ACCURACY_DIR, DATASET_DIR, RESULTS_LOG_DIR, CLUSTERING_SEEDS, load_dataset


def build_model(args):
    """
    Construct the model and apply polynomial activation.

    Args:
        args (argparse.Namespace): Experiment configuration with
            degree (int): Degree of the polynomial activation.
            clip (int or float): Clipping threshold for the activation.
            pbit (int): Precision setting for polynomial coefficients.

    Returns:
        torch.nn.Module: Model with polynomial activations and explicit conv biases.
    """
    model = get_model(args).cuda()
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is None:
            m.bias = nn.Parameter(torch.zeros(m.out_channels, device=m.weight.device))
    poly_activation = PolyActivation(args.degree, args.clip, args.pbit)
    replace_model(model, poly_activation)
    return model


def build_log_file(args, mode):
    """
    Build the accuracy log file path for clustered checkpoints.

    Args:
        args (argparse.Namespace): Experiment configuration with
            dataset (str): Dataset name.
            model (str): Model architecture name.
            degree (int): Polynomial degree.
            clip (int or float): Activation clipping threshold.
            pbit (int): Coefficient precision.
            penalty (float): Penalty coefficient for training.
            sigma (float): Training noise standard deviation.
        mode (str): Clustering mode.

    Returns:
        str: Absolute log file path.
    """
    name = (
        f"accuracy_cluster_summary"
        f"_degree_{args.degree}_clip_{args.clip}_pbit_{args.pbit}"
        f"_zeta_{args.penalty}_sigma_{args.sigma}_mode_{mode}"
    )
    path = os.path.join(
        ARTIFACTS_DIR,
        RESULTS_LOG_DIR,
        ACCURACY_DIR,
        f"{args.dataset}_{args.model}",
        f"{name}.log",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def main():
    """
    Evaluation of clustered checkpoints.
    """
    args = parse_arguments()
    _, model_path = setup_paths(args)
    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, args.dataset)

    model = build_model(args)

    mode = "slice" if args.slice else "full"
    log_file = build_log_file(args, mode)
    init_logging(log_file)

    seeds = args.seed_list
    centroid_list = args.k_list
    for k in centroid_list:
        accuracies = []
        best_acc = -1
        best_model_path = ""

        for seed in seeds:
            testdataset_path = os.path.join(dataset_path, f'testdataset_seed_{seed}.pkl')
            test_dataset = load_dataset(testdataset_path)
            testloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

            for cseed in CLUSTERING_SEEDS:
                model_save_path = get_model_save_path(model_path, seed)
                model_save_path = model_save_path.replace(".pth", f"_fused_{mode}_k_{k}_cseed_{cseed}.pth")

                model.load_state_dict(torch.load(model_save_path, weights_only=True))
                test_acc = test(model, testloader)
                accuracies.append(test_acc)

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model_path = model_save_path
        
        accuracies = np.array(accuracies)
        logging.info(f"Accuracy statistics: (K={k})")
        logging.info(f"Accuracies: {accuracies}")
        logging.info(f"Mean: {accuracies.mean():.4f}, Std: {accuracies.std():.4f}, Min: {accuracies.min():.2f}, Max: {accuracies.max():.2f}")
        logging.info(f"Accuracy (mean, std): ({accuracies.mean():.4f}, {accuracies.std():.4f})")
        logging.info(f"Best model path: {best_model_path}\n")


if __name__ == "__main__":
    main()
