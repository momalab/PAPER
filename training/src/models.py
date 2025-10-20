import torch
from resnet import resnet18, resnet20, resnet32
from utils import num_classes_map


def get_model(args):
    """
    Select and initialize a model architecture based on arguments.

    Args:
        args (argparse.Namespace): Experiment configuration containing with:
            model (str): Model architecture name.
            dataset (str): Dataset name (used to determine number of classes).

    Returns:
        torch.nn.Module: Instantiated model ready for training.
    """
    arch = args.model.lower()
    num_classes = num_classes_map[args.dataset]
    if "resnet18" in arch:
        return resnet18(num_classes)
    elif "resnet20" in arch:
        return resnet20(num_classes)
    elif "resnet32" in arch:
        return resnet32(num_classes)
    else:
        raise ValueError(f"Unknown model_arch: {args.model}")


def replace_model(model, poly_activation):
    """
    Recursively replace ReLU activations in a model with a custom activation.

    Args:
        model (torch.nn.Module): Model in which activations will be replaced.
        poly_activation (torch.nn.Module): Custom activation module.
    """
    if not isinstance(model, torch.nn.Module):
        return

    if hasattr(model, "relu"):
        setattr(model, "relu", poly_activation)
    
    if hasattr(model, "_modules"):
        for key, module in model._modules.items():
            if isinstance(module, torch.nn.ReLU):
                model._modules[key] = poly_activation
    
    for child in model.children():
        replace_model(child, poly_activation)
