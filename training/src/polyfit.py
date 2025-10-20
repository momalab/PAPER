import numpy as np
import argparse
import os
from gekko import GEKKO
from utils import COEFF_DIR, ARTIFACTS_DIR

def relu(x):
    """
    Standard ReLU activation.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: ReLU output (max(0, x)).
    """
    return np.maximum(0, x)

def quantized_poly_fit(xs, ys, degree, pbit):
    """
    Fit a quantized polynomial approximation to data points.

    Args:
        xs (numpy.ndarray): Input samples.
        ys (numpy.ndarray): Target function values.
        degree (int): Degree of the polynomial.
        pbit (int): Number of precision bits for coefficient quantization.

    Returns:
        numpy.ndarray: Fitted polynomial coefficients (low to high degree).
    """
    X = np.vstack([xs**i for i in range(degree + 1)]).T
    ys_scaled = np.round(ys * (2 ** pbit)).astype(int)

    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    coeffs = [m.Var(integer=True, lb=-2**(pbit-1), ub=2**(pbit-1)) for _ in range(degree + 1)]

    errors = []
    for i in range(len(xs)):
        pred = sum(X[i][j] * coeffs[j] for j in range(degree + 1))
        errors.append(m.Intermediate((pred - ys_scaled[i]) ** 2))

    m.Obj(sum(errors))
    m.solve(disp=False)

    result = np.array([c.value[0] for c in coeffs]) / (2 ** pbit)
    return result

def fit_polynomial(degree, clip, pbit, granularity=0.01):
    """
    Fit and save quantized polynomial approximation to ReLU.

    Args:
        degree (int): Degree of polynomial.
        clip (float): Range of input values (-clip to clip).
        pbit (int): Precision bits for coefficients.
        granularity (float, optional): Step size for input sampling. Default 0.01.
    """
    xs = np.linspace(-clip, clip, int(2 * clip / granularity))
    ys = relu(xs)

    coeffs = quantized_poly_fit(xs, ys, degree=degree, pbit=pbit)
    coeffs = coeffs[::-1]

    output_dir = os.path.join(ARTIFACTS_DIR, COEFF_DIR)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"deg_{degree}_clip_{int(clip)}_pbit_{pbit}_coeffs.txt")

    np.savetxt(filename, coeffs, fmt="%.10f")

    print("Coefficients (high to low degree):", coeffs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--pbit", type=int)
    args = parser.parse_args()

    fit_polynomial(degree=args.degree, clip=args.clip, pbit=args.pbit)
