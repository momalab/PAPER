import itertools
import logging
from math import comb
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from arguments import parse_arguments
from dataloader import get_datasets, get_transforms
from logger import init_logging, setup_paths
from models import get_model, replace_model
from poly_activation import PolyActivation
from utils import ARTIFACTS_DIR, CLUSTER_DIR, CLUSTERING_SEEDS, DATASET_DIR, RESULTS_LOG_DIR, save_dataset


def pairwise_distance(x, y):
    """
    Compute pairwise distances between two sets of vectors.

    Args:
        x (torch.Tensor): Input tensor with shape [N, D].
        y (torch.Tensor): Input tensor with shape [M, D].

    Returns:
        torch.Tensor: Distance matrix with shape [N, M].
    """
    return torch.cdist(x, y)


def kmeans_plusplus_init_nd(points, k, seed):
    """
    Initialize KMeans centroids using KMeans++ for D dimensional data.

    Args:
        points (torch.Tensor): Data with shape [N, D].
        k (int): Number of centroids.
        seed (int): Random seed for sampling.

    Returns:
        torch.Tensor: Initial centroids with shape [k, D].
    """
    torch.manual_seed(seed)
    N, D = points.shape
    centroids = torch.empty((k, D), dtype=points.dtype, device=points.device)

    idx = torch.randint(0, N, (1,))
    centroids[0] = points[idx]
    dists = pairwise_distance(points, centroids[0:1]).squeeze(1)
    for i in range(1, k):
        probs = dists.pow(2)
        probs /= probs.sum()
        next_idx = torch.multinomial(probs, 1)
        centroids[i] = points[next_idx]

        new_dists = pairwise_distance(points, centroids[i:i+1]).squeeze(1)
        dists = torch.minimum(dists, new_dists)

    return centroids


def torch_kmeans_nd(points, k, seed, max_iter=1000, tol=1e-4):
    """
    Run KMeans for D dimensional data with a chosen distance metric.

    Args:
        points (torch.Tensor): Data with shape [N, D].
        k (int): Number of clusters.
        seed (int): Random seed for initialization.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance on centroid shift.

    Returns:
        tuple: (torch.Tensor, torch.Tensor): Final centroids with shape [k, D] and Cluster indices with shape [N].
    """
    centroids = kmeans_plusplus_init_nd(points, k, seed)
    print(f"  Init centroids (shape {centroids.shape})")

    for i in range(max_iter):
        dists = pairwise_distance(points, centroids)
        labels = torch.argmin(dists, dim=1)

        new_centroids = torch.empty_like(centroids)
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                new_centroids[idx] = points[mask].mean(dim=0)
            else:
                new_centroids[idx] = centroids[idx]

        shift = torch.norm(centroids - new_centroids).item()
        if shift < tol:
            print(f"    Converged after {i + 1} iterations.")
            break
        centroids = new_centroids

    return centroids, labels


def cluster_conv_parameters(models, k, seed):
    """
    Cluster and quantize Conv2d weights across an ensemble jointly.

    Args:
        models (list[torch.nn.Module]): Models with identical structure and on the same device.
        k (int): Number of centroids per slice.
        seed (int): Random seed for KMeans initialization.
    """
    model0 = models[0]
    num_models = len(models)

    for name, layer0 in model0.named_modules():
        print(f"Processing layer: {name}")
        if isinstance(layer0, torch.nn.Conv2d):
            with torch.no_grad():
                layers = [m.get_submodule(name) for m in models]
                weights = [l.weight for l in layers]
                oc, ic, h, w = weights[0].shape

                for wi in range(w):
                    slices = [w[:, :, :, wi].contiguous().view(-1) for w in weights]
                    slice_stack = torch.stack(slices, dim=1).cuda()

                    if slice_stack.size(0) < k:
                        continue
                    
                    centroids, labels = torch_kmeans_nd(slice_stack, k, seed)
                    clustered = centroids[labels]

                    for m in range(num_models):
                        clustered_m = clustered[:, m].view(oc, ic, h)
                        weights[m][:, :, :, wi].copy_(clustered_m)


def build_poly_model(args):
    """
    Construct a model and replace ReLU with polynomial activation.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            degree (int): Degree of polynomial activation.
            clip (int): Clipping threshold for polynomial activation.
            pbit (int): Precision for polynomial coefficients.

    Returns:
        torch.nn.Module: Model with polynomial activations.
    """
    model = get_model(args).cuda()
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is None:
            m.bias = nn.Parameter(torch.zeros(m.out_channels, device=m.weight.device))
    
    poly_activation = PolyActivation(degree=args.degree, clip=args.clip, pbit=args.pbit)
    replace_model(model, poly_activation)
    return model


def load_poly_model(mseed, args, model_path):
    """
    Load a fused polynomial checkpoint for a specific seed.

    Args:
        mseed (int): Seed id used to select the checkpoint.
        args (argparse.Namespace): Experiment configuration.
        model_path (str): Directory containing checkpoints.

    Returns:
        torch.nn.Module: Model in evaluation mode with loaded weights.
    """
    model = build_poly_model(args)
    path = os.path.join(model_path, f'best_model_{mseed}_fused.pth')
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def evaluate_ensemble(models, testloader):
    """
    Evaluate an ensemble by averaging logits across models.

    Args:
        models (list[torch.nn.Module]): Models in evaluation mode.
        testloader (DataLoader): Loader for evaluation subset.

    Returns:
        float: Ensemble accuracy in percent.
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


def get_best_ensemble(model_seeds, testloader, args, model_path, num_model):
    """
    Select the best ensemble by sampling combinations and evaluating accuracy.

    Args:
        model_seeds (list[int]): Candidate seed ids.
        testloader (DataLoader): Evaluation loader.
        args (argparse.Namespace): Experiment configuration.
        model_path (str): Directory containing checkpoints.
        num_model (int): Number of models in each ensemble.

    Returns:
        tuple[int]: Seed ids of the best performing ensemble.
    """
    model_cache = {}
    def get_model_by_seed(seed):
        if seed not in model_cache:
            model_cache[seed] = load_poly_model(seed, args, model_path)
        return model_cache[seed]

    n = len(model_seeds)
    k = num_model

    if n < k:
        raise ValueError(f"Insufficient models: found {n}, but {k} required for ensemble formation.")
    
    total_combos = comb(n, k)

    best_acc = -1.0
    best_comb = None

    def eval_combo(combo):
        models = [get_model_by_seed(s) for s in combo]
        return evaluate_ensemble(models, testloader)
    
    if total_combos <= 30:
        for combo in itertools.combinations(model_seeds, k):
            acc = eval_combo(combo)
            if acc > best_acc:
                best_acc = acc
                best_comb = combo
    else:
        rng = random.Random(2025)
        for _ in range(30):
            combo = tuple(rng.sample(model_seeds, k))
            acc = eval_combo(combo)
            if acc > best_acc:
                best_acc = acc
                best_comb = combo
    
    return best_comb


def build_log_file(args):
    """
    Build the clustering log file path.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            dataset (str): Dataset name.
            model (str): Model architecture name.
            degree (int): Polynomial degree.
            clip (int): Activation clipping threshold.
            pbit (int): Coefficient precision.
            penalty (float): Penalty coefficient.
            sigma (float): Gaussian noise std.
            kf (int): Number of centroids per slice.
            num_models (int): Ensemble size.

    Returns:
        str: Absolute path where the log will be written.
    """
    name = (
        f"logs_{args.dataset}_{args.model}_degree_{args.degree}_clip_{args.clip}_pbit_{args.pbit}_"
        f"zeta_{args.penalty}_sigma_{args.sigma}_slice_k_{args.k}_ens_{args.num_models}"
    )
    path = os.path.join(
        ARTIFACTS_DIR,
        RESULTS_LOG_DIR,
        CLUSTER_DIR,
        f"{args.dataset}_{args.model}",
        f"{name}.log",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def main():
    """
    Parameter clustering for an ensemble.
    """
    args = parse_arguments()
    _, model_path = setup_paths(args)

    transform_train, transform_test = get_transforms(args.dataset)
    _, testset = get_datasets(args.dataset, transform_train, transform_test)

    test_size = int(args.validation_split * len(testset))
    local_random = random.Random(42)
    random_indices = local_random.sample(range(len(testset)), test_size)
    test_subset = Subset(testset, random_indices)
    testloader = DataLoader(test_subset, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, args.dataset)
    testdataset_path = os.path.join(dataset_path, 'ensemble_testdataset.pkl')
    if not os.path.exists(testdataset_path):
        save_dataset(testloader, testdataset_path)
    
    model_seeds = args.seed_list
    best_ensemble = get_best_ensemble(model_seeds, testloader, args, model_path, args.num_models)

    log_file = build_log_file(args)
    init_logging(log_file)

    for cseed in CLUSTERING_SEEDS:
        models = []
        for seed in best_ensemble:
            model = load_poly_model(seed, args, model_path)
            models.append(model)
        acc = evaluate_ensemble(models, testloader)
        logging.info(f"\nInitial Accuracy: {acc:.4f}")

        logging.info(f"Clustering Seed {cseed}")
        cluster_conv_parameters(models, args.k, cseed)
        acc = evaluate_ensemble(models, testloader)
        logging.info(f"After Clustering Accuracy: {acc:.4f}")

        for seed, model in zip(best_ensemble, models):
            file_name = f'best_model_{seed}_fused_k_{args.k}_cseed_{cseed}_ens_{args.num_models}.pth'
            save_path = os.path.join(model_path, file_name)
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
