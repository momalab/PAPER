import logging
import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import re
import numpy as np
from collections import defaultdict

from arguments import parse_arguments
from logger import init_logging, setup_paths
from models import get_model, replace_model
from poly_activation import PolyActivation
from utils import ARTIFACTS_DIR, ACCURACY_DIR, CLUSTERING_SEEDS, DATASET_DIR, RESULTS_LOG_DIR, load_dataset


def build_poly_model(args):
    """
    Construct the model and replace ReLU with polynomial activations.

    Args:
        args (argparse.Namespace): Experiment configuration with:
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
    poly_activation = PolyActivation(degree=args.degree, clip=args.clip, pbit=args.pbit)
    replace_model(model, poly_activation)
    return model


def load_poly_model(model_path, path, args):
    """
    Load a fused polynomial checkpoint.

    Args:
        model_path (str): Directory containing checkpoints.
        path (str): Checkpoint filename.
        args (argparse.Namespace): Experiment configuration.

    Returns:
        torch.nn.Module: Model in evaluation mode with loaded weights.
    """
    model = build_poly_model(args)
    path = os.path.join(model_path, path)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def evaluate_ensemble(models, testloader):
    """
    Evaluate an ensemble by averaging logits.

    Args:
        models (list[torch.nn.Module]): Models set to eval.
        testloader (DataLoader): Evaluation loader.

    Returns:
        float: Accuracy in percent.
    """
    correct = 0
    total = 0
    for inputs, targets in testloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = [model(inputs) for model in models]

        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        _, predicted = ensemble_output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    ensemble_acc = 100.0 * correct / total
    return ensemble_acc


def build_log_file(args):
    """
    Build the accuracy log file path.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            dataset (str): Dataset name.
            model (str): Model architecture.
            sigma (float): Noise parameter.
            degree (int): Polynomial degree.
            clip (int or float): Activation clipping threshold.
            pbit (int): Coefficient precision.
            penalty (float): Penalty coefficient.
            k (int): Number of centroids per slice.

    Returns:
        str: Absolute log file path.
    """
    name = (
        f"accuracy_ensemble_cluster_summary"
        f"_degree_{args.degree}_clip_{args.clip}_pbit_{args.pbit}"
        f"_zeta_{args.penalty}_sigma_{args.sigma}_k_{args.k}"
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
    Evaluation of clustered ensembles.
    """
    args = parse_arguments()
    _, model_path = setup_paths(args)

    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, args.dataset)
    testdataset_path = os.path.join(dataset_path, 'ensemble_testdataset.pkl')
    test_dataset = load_dataset(testdataset_path)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    log_file = build_log_file(args)
    init_logging(log_file)

    ensemble_size = args.num_models_list
    for num_models in ensemble_size:
        files_by_cseed = defaultdict(list)
        for filename in os.listdir(model_path):
            if f'k_{args.k}' not in filename or f'ens_{num_models}' not in filename:
                continue
            
            match = re.search(r'cseed_(\d+)', filename)
            if match:
                cseed = int(match.group(1))
                if cseed in CLUSTERING_SEEDS:
                    files_by_cseed[cseed].append(filename)
                    
        ensemble_accuracies = []
        best_acc = -1
        best_model_path = None
        for files in files_by_cseed.values():
            models = []
            for f in files:
                model = load_poly_model(model_path, f, args)
                models.append(model)
            acc = evaluate_ensemble(models, testloader)
            if acc > best_acc:
                best_acc = acc
                best_model_path = files
            ensemble_accuracies.append(acc)
            
        accuracies = np.array(ensemble_accuracies)
        logging.info(f"Accuracy statistics: (Ensemble={num_models})")
        logging.info(f"Accuracies: {accuracies}")
        logging.info(f"Mean: {accuracies.mean():.4f}, Std: {accuracies.std():.4f}, Min: {accuracies.min():.2f}, Max: {accuracies.max():.2f}")
        logging.info(f"Accuracy (mean, std): ({accuracies.mean():.4f}, {accuracies.std():.4f})")
        logging.info(f"Best model path: {best_model_path}\n")

if __name__ == "__main__":
    main()
