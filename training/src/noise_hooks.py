import torch
import torch.nn as nn


def add_activation_noise(model, sigma, module_types=(nn.Conv2d,)):
    """
    Add Gaussian noise to the activations of specific module types during training.

    Args:
        model (torch.nn.Module): Model whose activations will be noised.
        sigma (float): Standard deviation of the Gaussian noise.
        module_types (tuple): Module types to which noise will be applied.

    Returns:
        torch.nn.Module: Model with noise hooks registered.
    """
    if sigma <= 0:
        return model

    def _noise_hook(module, _, output):
        """
        Forward hook function that adds Gaussian noise to the output of the given module during training.
        """
        if module.training:
            return output + torch.randn_like(output) * sigma
        return output

    for m in model.modules():
        if isinstance(m, module_types):
            m.register_forward_hook(_noise_hook)

    return model
