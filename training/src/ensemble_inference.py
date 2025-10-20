import itertools
import logging
from math import comb
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from arguments import parse_arguments
from dataloader import get_datasets, get_transforms
from logger import init_logging, setup_paths
from models import get_model, replace_model
from poly_activation import PolyActivation
from trainer import test
from utils import ARTIFACTS_DIR, DATASET_DIR, ENSEMBLE_DIR, RESULTS_LOG_DIR, save_dataset


def build_poly_model(args):
    """
    Construct a model and replace ReLU with polynomial activations.

    Args:
        args (argparse.Namespace): Experiment configuration with
            degree (int): Degree of the polynomial activation.
            clip (int): Clipping threshold for polynomial activation.
            pbit (int): Precision setting for polynomial coefficients.

    Returns:
        torch.nn.Module: Model on GPU with polynomial activations.
    """
    model = get_model(args).cuda()
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is None:
            m.bias = nn.Parameter(torch.zeros(m.out_channels, device=m.weight.device))
    
    poly_activation = PolyActivation(degree=args.degree, clip=args.clip, pbit=args.pbit)
    replace_model(model, poly_activation)
    return model


def evaluate_single_models(args, model_path, testloader, model_seeds):
    """
    Evaluate all single models and return accuracies sorted by performance.

    Args:
        args (argparse.Namespace): Experiment configuration.
        model_path (str): Path to saved checkpoints.
        testloader (DataLoader): Data loader for evaluation.
        model_seeds (list[int]): Seeds identifying checkpoints.

    Returns:
        list[tuple[int, float]]: (seed, accuracy) pairs sorted descending by accuracy.
    """
    accuracies = []
    for mseed in model_seeds:
        model = load_poly_model(mseed, args, model_path)
        acc = test(model, testloader)
        logging.info(f"mseed={mseed}, accuracy={acc:.2f}")
        accuracies.append((mseed, acc))
    
    accuracies.sort(key=lambda x: x[1], reverse=True)
    return accuracies


def load_poly_model(mseed, args, model_path):
    """
    Load a fused polynomial checkpoint for a specific seed.

    Args:
        mseed (int): Seed index used to select the checkpoint.
        args (argparse.Namespace): Experiment configuration.
        model_path (str): Directory containing saved checkpoints.

    Returns:
        torch.nn.Module: Model loaded with checkpoint weights in eval mode.
    """
    model = build_poly_model(args)
    path = os.path.join(model_path, f'best_model_{mseed}_fused.pth')
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def evaluate_top_k_ensemble(top_k_models, testloader):
    """
    Evaluate an ensemble by averaging logits across models.

    Args:
        models (list[torch.nn.Module]): Models in evaluation mode.
        testloader (DataLoader): Loader for evaluation dataset.

    Returns:
        float: Ensemble classification accuracy.
    """
    correct = 0
    total = 0

    for inputs, targets in testloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = [model(inputs) for model in top_k_models]
        
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        _, predicted = ensemble_output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    ensemble_acc = 100.0 * correct / total
    return ensemble_acc


def build_log_file(args):
    """
    Build the ensemble log file path.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            dataset (str): Dataset name.
            model (str): Model architecture name.
            degree (int): Degree of the polynomial.
            clip (int or float): Activation clipping threshold.
            pbit (int): Precision bits used in coefficient generation.
            penalty (float): Penalty coefficient for polynomial activation.
            sigma (float): Standard deviation for Gaussian noise.
            num_models (int): Number of models in the ensemble.

    Returns:
        str: Absolute path where the ensemble log will be written.
    """
    name = (
        f"logs_degree_{args.degree}"
        f"_clip_{args.clip}_pbit_{args.pbit}_zeta_{args.penalty}_sigma_{args.sigma}_num_models_{args.num_models}"
    )
    path = os.path.join(
        ARTIFACTS_DIR,
        RESULTS_LOG_DIR,
        ENSEMBLE_DIR,
        f"{args.dataset}_{args.model}",
        f"{name}.log",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def main():
    """
    Evaluation of single models and ensembles.
    """
    args = parse_arguments()
    _, model_path = setup_paths(args)

    transform_train, transform_test = get_transforms(args.dataset)
    _, testset = get_datasets(args.dataset, transform_train, transform_test)

    test_size = int(args.validation_split * len(testset))
    local_random = random.Random(42)
    random_indices = local_random.sample(range(len(testset)), test_size)
    test_subset = Subset(testset, random_indices)

    testloader = DataLoader(
        test_subset, 
        batch_size=args.batch_size_test, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, args.dataset)
    testdataset_path = os.path.join(dataset_path, 'ensemble_testdataset.pkl')
    if not os.path.exists(testdataset_path):
        save_dataset(testloader, testdataset_path)
    
    log_file = build_log_file(args)
    init_logging(log_file)

    model_seeds = args.seed_list
    single_model_accuracies = evaluate_single_models(args, model_path, testloader, model_seeds)

    n = len(single_model_accuracies)
    k = args.num_models
    if n < k:
        logging.info(f"Found {n} models which is fewer than requested {k}. Skipping ensemble evaluation.")
        return

    top_k = single_model_accuracies[:args.num_models]
    top_k_models = [load_poly_model(mseed, args, model_path) for mseed, _ in top_k]

    top_k_acc = evaluate_top_k_ensemble(top_k_models, testloader)
    logging.info(f"\nTop-{args.num_models} ensemble accuracy: {top_k_acc:.2f}\n")

    total_combos = comb(n, k)
    logging.info(f"Total single models = {n}")
    logging.info(f"Total possible ensembles of size {k} = {total_combos}")

    model_cache = {}
    def get_model_by_seed(seed):
        if seed not in model_cache:
            model_cache[seed] = load_poly_model(seed, args, model_path)
        return model_cache[seed]

    ensemble_accuracies = []

    if total_combos <= 30:
        logging.info(f"Evaluating all {total_combos} ensembles of {k} models...\n")
        ordered_seeds = [mseed for mseed, _ in single_model_accuracies]
        for idx, combo in enumerate(itertools.combinations(ordered_seeds, k), start=1):
            models = [get_model_by_seed(s) for s in combo]
            acc = evaluate_top_k_ensemble(models, testloader)
            ensemble_accuracies.append(acc)
            seeds_str = ", ".join([f"(mseed={s})" for s in combo])
            logging.info(f"Ensemble {idx}: Accuracy = {acc:.2f}, Models = [{seeds_str}]")
    else:
        logging.info(f"Evaluating 30 random ensembles of {k} models...")
        rng = random.Random(2025)
        ordered_seeds = [mseed for mseed, _ in single_model_accuracies]
        for i in range(30):
            sampled = rng.sample(ordered_seeds, k)
            models = [get_model_by_seed(s) for s in sampled]
            acc = evaluate_top_k_ensemble(models, testloader)
            ensemble_accuracies.append(acc)
            seeds_str = ", ".join([f"(mseed={s})" for s in sampled])
            logging.info(f"\nRandom Ensemble {i+1}: Accuracy = {acc:.2f}, Models = [{seeds_str}]")

    mean_acc = np.mean(ensemble_accuracies)
    std_acc = np.std(ensemble_accuracies)
    min_acc = np.min(ensemble_accuracies)
    max_acc = np.max(ensemble_accuracies)

    logging.info(
        f"\nEnsembles Accuracy - Min: {min_acc:.2f}, Max: {max_acc:.2f}, (Mean, Std): ({mean_acc:.4f}, {std_acc:.4f})"
    )


if __name__ == "__main__":
    main()
