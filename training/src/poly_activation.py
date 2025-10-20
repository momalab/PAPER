import numpy as np
import torch
from torch import nn
from utils import ARTIFACTS_DIR, COEFF_DIR


def get_polynomial(degree, clip, pbit):
    """
    Load polynomial coefficients for the activation function.

    Args:
        degree (int): Degree of the polynomial.
        clip (int or float): Input clipping value used in coefficient generation.
        pbit (int): Precision bits used in coefficient generation.

    Returns:
        numpy.ndarray: Coefficient array ordered from highest to lowest degree.
    """
    coeffs = np.asarray(np.loadtxt(f"{ARTIFACTS_DIR}/{COEFF_DIR}/deg_{degree}_clip_{int(clip)}_pbit_{pbit}_coeffs.txt"))
    return coeffs


class PolyActivation(nn.Module):
    def __init__(self, degree, clip, pbit):
        """
        Initialize the polynomial activation.

        Args:
            degree (int): Degree of the polynomial.
            clip (int or float): Clipping value for inputs during training.
            pbit (int): Precision bits used in coefficient generation.
        """
        super().__init__()
        self.register_buffer("coeffs", torch.tensor(get_polynomial(degree, clip, pbit), dtype=torch.float32))
        self.clip = clip
        self.buff = []

    def forward(self, x):
        """
        Apply clipping in training mode then evaluate the polynomial.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        if self.training:
            self.buff.append(x)
            x = torch.clamp(x, -self.clip, self.clip)
        
        y = self.polyval(x)
        return y

    def polyval(self, x):
        """
        Evaluate the polynomial function on input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after polynomial evaluation.
        """
        degree = len(self.coeffs) - 1
        y = 0
        for i, coeff in enumerate(self.coeffs):
            y = y + coeff * (x ** (degree - i))
        return y

    def reset(self):
        """
        Clear the stored activation buffer.
        """
        self.buff = []
