import argparse


def parse_arguments():
    """
    Parse command line arguments for experiments.

    Returns:
        argparse.Namespace: Object with attributes
            dataset (str): Name of the dataset. Valid options are 'cifar10' and 'cifar100'.
            model (str): Model architecture name. Use 'resnet18', 'resnet20', or 'resnet32' for training with ReLU. 
                         Append '_poly' to use the corresponding model with polynomial activations, for example 'resnet18_poly'.
            num_epochs (int): Number of training epochs.
            batch_size_train (int): Batch size for training.
            batch_size_test (int): Batch size for validation and testing.
            validation_split (float): Fraction of the test set used for validation.
            learning_rate (float): Initial learning rate.
            momentum (float): SGD momentum.
            weight_decay (float): L2 regularization weight decay.
            seed (int): Random seed for training.
            seed_list (list[int]): Space-separated list of all trained seed values.
            degree (int): Degree of the polynomial used to approximate ReLU.
            clip (int): Clipping range for polynomial activation.
            penalty (float): Regularization strength applied during training.
            sigma (float): Standard deviation for activation noise.
            pbit (int): Bit precision for quantized polynomial coefficients.
            slice (bool): Enables slice-wise clustering mode.
            k (int): Number of clusters for K-means.
            k_list (list[int]):Space-separated list of all cluster sizes to evaluate.
            num_models (int): Number of models to include in the ensemble
            num_models_list (list[int]): Space-separated list of ensemble sizes to evaluate.
            model_save_path (str): Custom model save directory path.
            json_save_path (str): Custom JSON save directory path.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, choices=[
        'resnet18', 'resnet20', 'resnet32',
        'resnet18_poly', 'resnet20_poly', 'resnet32_poly'
    ])
    parser.add_argument('--num_epochs', type=int, default=185)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--validation_split', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.013)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_list', type=int, nargs='+')
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--clip', type=int, default=2)
    parser.add_argument("--penalty", type=float, default=1e-3)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--pbit', type=int, default=10)
    parser.add_argument('--slice', action='store_true')
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--k_list', type=int, nargs='+')
    parser.add_argument('--num_models', type=int, default=1)
    parser.add_argument('--num_models_list', type=int, nargs='+')
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--json_save_path', type=str)

    return parser.parse_args()
