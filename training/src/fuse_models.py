import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from arguments import parse_arguments
from logger import get_model_save_path, init_logging, setup_paths
from models import get_model, replace_model
from poly_activation import PolyActivation
from trainer import test
from utils import ARTIFACTS_DIR, DATASET_DIR, FUSING_DIR, RESULTS_LOG_DIR, load_dataset


def _fuse_conv_bn_pair(conv, bn):
    """
    Fuse a Conv2d and BatchNorm2d into Conv2d parameters in place.

    Args:
        conv (nn.Conv2d): Convolution layer that absorbs batch norm statistics.
        bn (nn.BatchNorm2d): Batch normalization layer to be folded.
    """
    if conv is None or bn is None:
        return

    with torch.no_grad():
        w = conv.weight
        b = conv.bias
        if b is None:
            b = torch.zeros(conv.out_channels, device=w.device)

        mu = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(var + eps)
        scale = gamma / std
        shift = beta - mu * scale

        w.mul_(scale.view(-1, 1, 1, 1))
        b = b * scale + shift

        conv.bias = nn.Parameter(b)

        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.eps = 0.0


def fuse_model(model):
    """
    Traverse the module tree and fuse all Conv2d followed by BatchNorm2d pairs.

    Args:
        model (nn.Module): Model that may contain Conv2d-BatchNorm2d sequences.
    """
    for m in model.modules():
        children = list(m.named_children())
        for idx, (_, child) in enumerate(children):
            if isinstance(child, nn.Conv2d) and idx + 1 < len(children):
                next_child = children[idx + 1][1]
                if isinstance(next_child, nn.BatchNorm2d):
                    _fuse_conv_bn_pair(child, next_child)


def build_model(args):
    """
    Build the model and optionally replace ReLU with a polynomial activation.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            degree (int): Degree of the polynomial.
            clip (int or float): Activation clipping threshold.
            pbit (int): Precision bits used in coefficient generation.

    Returns:
        torch.nn.Module: Model ready for evaluation.
    """
    model = get_model(args).cuda()
    poly_activation = PolyActivation(args.degree, args.clip, args.pbit)
    replace_model(model, poly_activation)
    return model


def build_log_file(args):
    """
    Create the fusion log file path from the configuration.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            dataset (str): Dataset name.
            model (str): Model architecture name.
            degree (int): Degree of the polynomial.
            clip (int or float): Activation clipping threshold.
            pbit (int): Precision bits used in coefficient generation.
            penalty (float): Penalty coefficient for polynomial activation.
            sigma (float): Standard deviation for Gaussian noise.

    Returns:
        str: Absolute path where the fusion log will be written.
    """
    name = (
        f"logs_degree_{args.degree}"
        f"_clip_{args.clip}_pbit_{args.pbit}_zeta_{args.penalty}_sigma_{args.sigma}"
    )
    path = os.path.join(
        ARTIFACTS_DIR,
        RESULTS_LOG_DIR,
        FUSING_DIR,
        f"{args.dataset}_{args.model}",
        f"{name}.log",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def main():
    """
    Run Conv BatchNorm fusion and evaluation across seeds and log outcomes.
    """
    args = parse_arguments()
    _, model_path = setup_paths(args)
    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, args.dataset)

    log_file = build_log_file(args)
    init_logging(log_file)

    model = build_model(args)

    seeds = args.seed_list
    for seed in seeds:
        model = get_model(args).cuda()
        poly_activation = PolyActivation(args.degree, args.clip, args.pbit)
        replace_model(model, poly_activation)

        testdataset_path = os.path.join(dataset_path, f'testdataset_seed_{seed}.pkl')
        model_save_path = get_model_save_path(model_path, seed)

        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval()

        test_dataset = load_dataset(testdataset_path)
        testloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size_test, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

        acc_before = test(model, testloader)
        logging.info(f"Accuracy before fusion (seed={seed}): {acc_before:.2f}")

        fuse_model(model)
        acc_after = test(model, testloader)
        logging.info(f"Accuracy after fusion (seed={seed}): {acc_after:.2f}")

        fused_path = model_save_path.replace(".pth", "_fused.pth")
        torch.save(model.state_dict(), fused_path)
        logging.info(f"Fused model saved to: {fused_path}\n")


if __name__ == "__main__":
    main()
