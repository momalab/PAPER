import os
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from arguments import parse_arguments
from logger import get_model_save_path, init_logging, setup_paths
from models import get_model, replace_model
from poly_activation import PolyActivation
from trainer import test
from utils import ARTIFACTS_DIR, CLUSTER_DIR, DATASET_DIR, RESULTS_LOG_DIR, CLUSTERING_SEEDS, load_dataset, set_seed


def load_model(model_path, args):
    """
    Construct a model, apply polynomial activation, and load a fused checkpoint.

    Args:
        model_path (str): Directory that contains checkpoints from training.
        args (argparse.Namespace): Experiment configuration. Uses
            degree (int): Degree of the polynomial activation.
            clip (int): Clipping threshold for polynomial activation.
            pbit (int): Precision setting for polynomial coefficients.
            seed (int): Seed that selects which checkpoint to load.

    Returns:
        torch.nn.Module: Model in evaluation mode with polynomial activations.
    """
    model = get_model(args).cuda()
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is None:
            m.bias = nn.Parameter(torch.zeros(m.out_channels, device=m.weight.device))
    poly_activation = PolyActivation(args.degree, args.clip, args.pbit)
    replace_model(model, poly_activation)

    model_save_path = get_model_save_path(model_path, args.seed)
    fused_path = model_save_path.replace(".pth", f"_fused.pth")
    model.load_state_dict(torch.load(fused_path, weights_only=True))
    model.eval()

    return model


def kmeans_plusplus_init(points, k):
    """
    Initialize centroids with KMeans++ for 1D data.

    Args:
        points (torch.Tensor): 1D tensor of values to be clustered.
        k (int): Number of centroids to generate.

    Returns:
        torch.Tensor: 1D tensor of length k containing initial centroids.
    """
    N = points.numel()
    centroids = torch.empty(k, dtype=points.dtype).cuda()
    idx = torch.randint(0, N, (1,)).cuda()
    centroids[0] = points[idx]

    closest_dist_sq = (points - centroids[0]).pow(2)
    for c in range(1, k):
        probs = closest_dist_sq / closest_dist_sq.sum()
        next_idx = torch.multinomial(probs, 1).cuda()
        centroids[c] = points[next_idx]
        new_dist_sq = (points - centroids[c]).pow(2)
        closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)

    return centroids


def torch_kmeans(points, k, max_iter, seed, chunk_size=100_000):
    """
    Run KMeans on 1D data with chunked assignment.

    Args:
        points (torch.Tensor): 1D tensor of values to be clustered.
        k (int): Number of clusters.
        max_iter (int): Maximum number of KMeans iterations.
        seed (int): Random seed used for initialization and sampling.
        chunk_size (int): Number of elements per assignment chunk.

    Returns:
        tuple: (torch.Tensor, torch.Tensor): Final centroids with shape [k] and cluster indices with shape [N].
    """
    set_seed(seed)
    N = points.numel()
    centroids = kmeans_plusplus_init(points, k)

    for _ in range(max_iter):
        labels = torch.empty(N, dtype=torch.long).cuda()
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk = points[i:end].view(-1, 1)
            dists = (chunk - centroids.view(1, -1)).pow(2)
            _, min_idx = torch.min(dists, dim=1)
            labels[i:end] = min_idx

        new_centroids = torch.empty_like(centroids)
        for idx in range(k):
            mask = labels == idx
            new_centroids[idx] = points[mask].mean() if mask.any() else centroids[idx]

        centroid_shift = torch.norm(centroids - new_centroids, p=2).item()
        if centroid_shift < 1e-4:
            break
        centroids = new_centroids

    return centroids, labels


def cluster_conv_parameters_full(model, k, seed):
    """
    Quantize Conv2d weights and biases by global KMeans.

    Args:
        model (torch.nn.Module): Model to modify in place. Only Conv2d layers are affected.
        k (int): Number of centroids shared across all Conv2d weights and biases.
        seed (int): Random seed for KMeans.
    """
    flattened, meta = [], []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            flattened.append(layer.weight.data.view(-1))
            meta.append((layer, "weight", layer.weight.numel()))
            if layer.bias is not None:
                flattened.append(layer.bias.data.view(-1))
                meta.append((layer, "bias", layer.bias.numel()))

    if not flattened:
        return

    all_params = torch.cat(flattened).cuda()
    centroids, labels = torch_kmeans(all_params, k, max_iter=1000, seed=seed)
    start = 0
    for layer, pname, count in meta:
        slice_labels = labels[start : start + count]
        snapped = centroids[slice_labels]
        if pname == "weight":
            layer.weight.data.copy_(snapped.view_as(layer.weight))
        else:
            layer.bias.data.copy_(snapped.view_as(layer.bias))
        start += count


def cluster_conv_parameters_slice(model, k, seed):
    """
    Quantize Conv2d weights by KMeans per width slice.

    Args:
        model (torch.nn.Module): Model to modify in place. Only weights are changed.
        k (int): Number of centroids per width slice.
        seed (int): Random seed for KMeans.
    """
    for lname, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            with torch.no_grad():
                weight = layer.weight
                oc, ic, h, w = weight.shape
                logging.info(f"Layer {lname}: weight shape = {weight.shape}")

                for wi in range(w):
                    slice_weights = weight[:, :, :, wi].contiguous().view(-1)
                    num_values = slice_weights.numel()

                    if num_values < k:
                        logging.info(f"  - Width {wi:2d}: Skipped (values={num_values} < k={k})")
                        continue
                    
                    centroids, labels = torch_kmeans(slice_weights, k=k, max_iter=1000, seed=seed)
                    clustered = centroids[labels].view(oc, ic, h)
                    weight[:, :, :, wi].copy_(clustered)


def build_log_file(args, mode, k):
    """
    Build the clustering log file path.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            dataset (str): Dataset name.
            model (str): Model architecture name.
            degree (int): Polynomial degree.
            clip (int): Activation clipping threshold.
            pbit (int): Coefficient precision.
            penalty (float): Penalty coefficient for polynomial activation.
            sigma (float): Gaussian noise std used during training.
            seed (int): Seed identifying the base checkpoint.
        mode (str): Clustering mode. Either "slice" or "full".
        k (int): Number of centroids.

    Returns:
        str: Absolute path where the log will be written.
    """
    name = (
        f"logs_degree_{args.degree}_clip_{args.clip}_pbit_{args.pbit}_"
        f"zeta_{args.penalty}_sigma_{args.sigma}_seed_{args.seed}_{mode}_k_{k}"
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
    Clustering for Conv2d parameters.
    """
    args = parse_arguments()
    _, model_path = setup_paths(args)
    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, args.dataset)
    
    testdataset_path = os.path.join(dataset_path, f'testdataset_seed_{args.seed}.pkl')
    test_dataset = load_dataset(testdataset_path)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    k = args.k
    mode = "slice" if args.slice else "full"
    log_file = build_log_file(args, mode, k)
    init_logging(log_file)
    
    print(f"\n\n --- Number of Clusters = {k} ---")
    for cseed in CLUSTERING_SEEDS:
        model = load_model(model_path, args)
        acc = test(model, testloader)
        logging.info(f"\nInitial Accuracy: {acc:.4f}")
        logging.info(f"Clustering Seed {cseed}\n")

        if args.slice:
            cluster_conv_parameters_slice(model, k=k, seed=cseed)
        else:
            cluster_conv_parameters_full(model, k=k, seed=cseed)

        acc = test(model, testloader)
        logging.info(f"After Clustering Accuracy: {acc:.4f}")

        model_save_path = get_model_save_path(model_path, args.seed)
        model_save_path = model_save_path.replace(".pth", f"_fused_{mode}_k_{k}_cseed_{cseed}.pth")
        torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    main()
