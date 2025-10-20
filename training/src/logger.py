import logging
import os
from utils import ARTIFACTS_DIR, LOG_DIR, MODEL_DIR


def setup_paths(args):
    """
    Create logging and model saving directories for the experiment.

    Args:
        args (argparse.Namespace): Experiment configuration containing with:
            dataset (str): Dataset name.
            model (str): Model architecture.
            degree (int): Degree of polynomial activation (if poly model).
            clip (int): Clipping parameter for polynomial activation.
            pbit (int): Precision bits for polynomial coefficients.
            penalty (float): Penalty coefficient.
            sigma (float): Noise standard deviation.

    Returns:
        tuple: (log_path, model_path), both are strings with directory paths.
    """
    if "poly" in args.model:
        base = lambda folder: os.path.join(
            folder, args.dataset, args.model, 
            f"degree_{args.degree}", f"clip_{args.clip}", f"pbit_{args.pbit}", 
            f"zeta_{args.penalty}", f"sigma_{args.sigma}"
        )
    else:
        base = lambda folder: os.path.join(folder, args.dataset, args.model)
    log_path = os.path.join(ARTIFACTS_DIR, base(LOG_DIR))
    model_path = os.path.join(ARTIFACTS_DIR, base(MODEL_DIR))
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    return log_path, model_path


def init_logging(logfile):
    """
    Initialize logging to both file and console.

    Args:
        logfile (str): Path to the log file.
    """
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w"), 
            logging.StreamHandler()
        ]
    )


def setup_logging(log_path, seed):
    """
    Setup logging for a given seed-specific run.

    Args:
        log_path (str): Directory where logs are saved.
        seed (int): Random seed identifier for the run.
    """
    logfile = os.path.join(log_path, f"log_seed_{seed}.log")
    logging.root.handlers.clear()
    logging.basicConfig(
        filename=logfile, 
        filemode='w', format='%(asctime)s - %(message)s', 
        level=logging.INFO
    )


def log_epoch_info(epoch, train_loss, train_acc, valid_acc):
    """
    Log training and validation statistics for an epoch.

    Args:
        epoch (int): Epoch number.
        train_loss (float): Average training loss.
        train_acc (float): Training accuracy (%).
        valid_acc (float): Validation accuracy (%).
    """
    logging.info(
        f'Epoch {epoch}: Training Loss: {train_loss:.6f}, '
        f'Training Acc: {train_acc:.4f}, Validation Acc: {valid_acc:.4f}'
    )


def log_test_info(test_acc):
    """
    Log final test accuracy.

    Args:
        test_acc (float): Test accuracy (%).
    """
    logging.info(f'Test Accuracy: {test_acc:.4f}')


def get_model_save_path(model_path, seed):
    """
    Construct path for saving the best model checkpoint.

    Args:
        model_path (str): Base directory for model saving.
        seed (int): Random seed identifier for the run.

    Returns:
        str: Path to the model checkpoint file.
    """
    model_save_path = os.path.join(model_path, f"best_model_{seed}.pth")
    return model_save_path
