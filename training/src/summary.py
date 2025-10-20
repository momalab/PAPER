from arguments import parse_arguments
from glob import glob
import logging
import os
import re

from utils import ARTIFACTS_DIR, ACCURACY_DIR, ENSEMBLE_DIR, RESULTS_LOG_DIR


def log_table(title, header, rows):
    """
    Log a formatted table with a title, header, and data rows.

    Args:
        title (str): Title of the table to display in the logs.
        header (list[str]): Column headers for the table.
        rows (list[list[Any]]): List of data rows, each a list of column values.
    """
    col_widths = [max(len(str(cell)) for cell in column) + 2 for column in zip(header, *rows)] if rows else [len(h) + 2 for h in header]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    def fmt_row(row):
        return "|" + "|".join(f" {str(cell):<{w - 2}} " for cell, w in zip(row, col_widths)) + "|"

    logging.info("")
    logging.info(title)
    logging.info(sep)
    logging.info("|" + "|".join(f"{str(cell):^{w}}" for cell, w in zip(header, col_widths)) + "|")
    logging.info(sep)
    for row in rows:
        logging.info(fmt_row(row))
    logging.info(sep)


def _print_simple_table(title, left_label, rows):
    """
    Log a simplified four-column table with a title.

    Args:
        title (str): Title of the table section.
        left_label (str): Label for the first column.
        rows (list[list[Any]]): Table rows. Each row should have four elements: [left_label_value, (mean, std), min, max].
    """
    header = [left_label, "(mean, std)", "min", "max"]
    col_widths = [max(len(str(cell)) for cell in col) + 2 for col in zip(header, *rows)] if rows else [len(h) + 2 for h in header]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    def fmt_row(row):
        return "|" + "|".join(f" {str(cell):<{w - 2}} " for cell, w in zip(row, col_widths)) + "|"

    logging.info("")
    logging.info(title)
    logging.info(sep)
    logging.info("|" + "|".join(f"{str(cell):^{w}}" for cell, w in zip(header, col_widths)) + "|")
    logging.info(sep)
    for r in rows:
        logging.info(fmt_row(r))
    logging.info(sep)


def extract_stats(filepath):
    """
    Extract accuracy statistics (mean, std, min, max) from a log file.

    Args:
        filepath (str): Path to the log file containing accuracy statistics.

    Returns:
        tuple[str, str, str]:
            - (mean, std): Formatted mean and standard deviation string.
            - min: Minimum accuracy value as string.
            - max: Maximum accuracy value as string.
    """
    try:
        with open(filepath) as f:
            content = f.read()
    except FileNotFoundError:
        return "-", "-", "-"
    mean_std = re.search(r"Accuracy \(mean, std\): \(([^,]+), ([^)]+)\)", content)
    min_val = re.search(r"Min: ([^,]+)", content)
    max_val = re.search(r"Max: ([^\n]+)", content)
    if mean_std and min_val and max_val:
        return f"({mean_std.group(1)}, {mean_std.group(2)})", min_val.group(1), max_val.group(1)
    return "-", "-", "-"


def extract_stats_from_block(text):
    """
    Extract accuracy statistics (mean, std, min, max) from a text block.

    Args:
        text (str): Text content of a log section containing accuracy info.

    Returns:
        tuple[str, str, str]:
            - (mean, std): Mean and standard deviation pair.
            - min: Minimum accuracy.
            - max: Maximum accuracy.
    """
    mean_std = re.search(r"Accuracy \(mean, std\): \(([^,]+), ([^)]+)\)", text)
    min_val = re.search(r"Min: ([^,]+)", text)
    max_val = re.search(r"Max: ([^\n]+)", text)
    if mean_std and min_val and max_val:
        return f"({mean_std.group(1)}, {mean_std.group(2)})", min_val.group(1), max_val.group(1)
    return "-", "-", "-"


def parse_ensemble_file(path):
    """
    Parse a single ensemble accuracy log file and extract summary stats.

    Args:
        path (str): Path to the ensemble log file.

    Returns:
        tuple[str, str, str]:
            - (mean, std): Ensemble accuracy mean and standard deviation.
            - min: Minimum ensemble accuracy.
            - max: Maximum ensemble accuracy.
    """
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        return "-", "-", "-"
    mean_std_match = re.search(r"Mean, Std\): \(([^,]+), ([^)]+)\)", content)
    min_match = re.search(r"Min: ([^,]+)", content)
    max_match = re.search(r"Max: ([^,\n]+)", content)
    if mean_std_match and min_match and max_match:
        mean_std = f"({mean_std_match.group(1)}, {mean_std_match.group(2)})"
        return mean_std, min_match.group(1), max_match.group(1)
    return "-", "-", "-"


def parse_cluster_block(filepath):
    """
    Parse a clustering accuracy log containing multiple K values.

    Args:
        filepath (str): Path to the log file containing multiple "Accuracy statistics" sections.

    Returns:
        dict[int, tuple[str, str, str]]: Dictionary mapping K values (int) to tuples of (mean, std), min, and max accuracy results.
    """
    try:
        with open(filepath) as f:
            content = f.read()
    except FileNotFoundError:
        return {}
    blocks = re.split(r"(?=Accuracy statistics: \(K=\d+\))", content)
    result = {}
    for block in blocks:
        k = re.search(r"K=(\d+)", block)
        stats = extract_stats_from_block(block)
        if k and all(val != "-" for val in stats):
            result[int(k.group(1))] = stats
    return result


def parse_cluster_ensemble_file(path):
    """
    Parse ensemble clustering accuracy logs containing multiple ensemble sizes.

    Args:
        path (str): Path to the ensemble cluster log file.

    Returns:
        dict[int, tuple[str, str, str]]: Dictionary mapping ensemble size (int) to tuples of (mean, std), min, and max accuracy results.
    """
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        return {}
    blocks = re.finditer(
        r"Accuracy statistics: \(Ensemble=(\d+)\)(.*?)(?=Accuracy statistics: \(Ensemble=\d+\)|\Z)",
        content,
        flags=re.DOTALL,
    )
    out = {}
    for m in blocks:
        n = int(m.group(1))
        stats = extract_stats_from_block(m.group(2))
        if all(val != "-" for val in stats):
            out[n] = stats
    return out


def write_standard_section(poly_config, sigma, relu_log_file, poly_dir):
    """
    Write a table comparing standard ReLU and polynomial model accuracies.

    Args:
        poly_config (str): Polynomial configuration string.
        sigma (float): Sigma value used during training.
        relu_log_file (str): Path to the ReLU accuracy log file.
        poly_dir (str): Directory containing polynomial accuracy logs.
    """
    rows = [["ReLU", *extract_stats(relu_log_file)]]
    filename = f"accuracy_summary_{poly_config}_sigma_{sigma}.log"
    path = os.path.join(poly_dir, filename)
    label = "Poly"
    rows.append([label, *extract_stats(path)] if os.path.exists(path) else [label, "-", "-", "-"])
    log_table("Standard Accuracy", ["Variant", "(mean, std)", "min", "max"], rows)


def write_standard_ensemble_section(dataset, model, poly_config, sigma):
    """
    Write a table of ensemble accuracy results for different ensemble sizes.

    Args:
        dataset (str): Dataset name.
        model (str): Model name.
        poly_config (str): Polynomial configuration string.
        sigma (float): Sigma value used during ensemble evaluation.
    """

    ensemble_base = os.path.abspath(
        os.path.join(ARTIFACTS_DIR, RESULTS_LOG_DIR, ENSEMBLE_DIR, f"{dataset}_{model}_poly")
    )
    pattern = f"logs_{poly_config}_sigma_*_num_models_*.log"

    regex = re.compile(
        r"^logs_degree_(?P<degree>\d+)_clip_(?P<clip>\d+)_pbit_(?P<pbit>\d+)_zeta_(?P<zeta>[0-9.eE+-]+)_sigma_(?P<sigma>[0-9.eE+-]+)_num_models_(?P<nmodels>\d+)\.log$"
    )

    by_sigma = {}
    for path in glob(os.path.join(ensemble_base, pattern)):
        base = os.path.basename(path)
        m = regex.match(base)
        if not m:
            continue
        sigma_in_file = float(m.group("sigma"))
        nmodels = int(m.group("nmodels"))
        by_sigma.setdefault(sigma_in_file, {})[nmodels] = parse_ensemble_file(path)

    if not by_sigma:
        logging.warning("No ensemble logs found.")
        return

    target = float(sigma)
    if target not in by_sigma:
        logging.warning("No ensemble logs for requested sigmas.")
        return

    Ns = sorted(set(by_sigma[target].keys()) & {2, 4, 8}) or sorted(by_sigma[target].keys())
    rows = [[n, *by_sigma[target].get(n, ("-", "-", "-"))] for n in Ns]
    log_table("Standard Ensemble Accuracy", ["# models", "(mean, std)", "min", "max"], rows)


def write_cluster_section(poly_config, sigma, mode, label, poly_dir):
    """
    Write a clustering accuracy table for either full or slice clustering modes.

    Args:
        poly_config (str): Polynomial configuration string.
        sigma (float): Sigma value.
        mode (str): Clustering mode, either 'full' or 'slice'.
        label (str): Descriptive label for the table title.
        poly_dir (str): Directory path to accuracy logs.
    """

    s_str = f"{sigma:.3f}".rstrip("0").rstrip(".")
    if s_str == "0":
        s_str = "0.0"
    fname = (
        f"accuracy_cluster_summary_{poly_config}_"
        f"sigma_{s_str}_mode_{mode}.log"
    )
    path = os.path.join(poly_dir, fname)
    stats = parse_cluster_block(path)
    if not stats:
        logging.warning(f"Missing {mode} log {path}")
        return

    title = f"Accuracy of {label} Clustering after Polynomial Training"
    rows = []
    for k in sorted(stats.keys()):
        rows.append([k, *stats.get(k, ("-", "-", "-"))])
    _print_simple_table(title, "K", rows)


def write_ensemble_cluster_section(poly_config, sigma, poly_dir):
    """
    Write a table summarizing ensemble clustering accuracies.

    Args:
        poly_config (str): Polynomial configuration string.
        sigma (float): Sigma value.
        poly_dir (str): Directory path containing ensemble cluster logs.
    """

    pattern = f"accuracy_ensemble_cluster_summary_{poly_config}_sigma_*_k_*.log"
    paths = glob(os.path.join(poly_dir, pattern))
    if not paths:
        logging.warning("No ensemble cluster logs found.")
        return

    filename_re = re.compile(
        rf"^accuracy_ensemble_cluster_summary_{re.escape(poly_config)}_sigma_(?P<sigma>[0-9.eE+-]+)_k_(?P<K>\d+)\.log$"
    )

    data = {}
    for path in paths:
        base = os.path.basename(path)
        m = filename_re.match(base)
        if not m:
            continue
        sigma_in_file = float(m.group("sigma"))
        if sigma_in_file != float(sigma):
            continue
        kf = int(m.group("K"))
        models_map = parse_cluster_ensemble_file(path)
        for n_models, stats in models_map.items():
            data[(n_models, kf)] = stats

    if not data:
        logging.warning("No parsable ensemble cluster blocks for the provided sigma.")
        return

    rows = []
    for (n_models, kf) in sorted(data.keys(), key=lambda x: (x[0], x[1])):
        mean_std, mn, mx = data[(n_models, kf)] if data[(n_models, kf)] else ("-", "-", "-")
        rows.append([n_models, kf, mean_std, mn, mx])

    header = ["# models", "K", "(mean, std)", "min", "max"]
    col_widths = [max(len(str(cell)) for cell in col) + 2 for col in zip(header, *rows)] if rows else [len(h) + 2 for h in header]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    def fmt_row(row):
        return "|" + "|".join(f" {str(cell):<{w - 2}} " for cell, w in zip(row, col_widths)) + "|"

    logging.info("")
    logging.info("Accuracy after Ensemble Clustering")
    logging.info(sep)
    logging.info("|" + "|".join(f"{str(cell):^{w}}" for cell, w in zip(header, col_widths)) + "|")
    logging.info(sep)

    last_models = None
    for r in rows:
        if last_models is not None and r[0] != last_models:
            logging.info(sep)
        logging.info(fmt_row(r))
        last_models = r[0]

    logging.info(sep)


def main(dataset, model, degree, clip, pbit, penalty, sigma):
    """
    Generate summary tables for all experiment results.

    Args:
        dataset (str): Dataset name.
        model (str): Model architecture name.
        degree (int): Degree of the polynomial activation.
        clip (int): Clipping value used during coefficient generation.
        pbit (int): Bit precision used for polynomial coefficients.
        penalty (float): Regularization penalty coefficient.
        sigma (float): Noise sigma used during training.
    """

    poly_config = f"degree_{degree}_clip_{clip}_pbit_{pbit}_zeta_{penalty}"

    relu_log_file = os.path.join(
        ARTIFACTS_DIR,
        RESULTS_LOG_DIR,
        ACCURACY_DIR,
        f"{dataset}_{model}",
        f"accuracy_summary.log"
    )
    poly_dir = os.path.join(
        ARTIFACTS_DIR,
        RESULTS_LOG_DIR,
        ACCURACY_DIR,
        f"{dataset}_{model}_poly"
    )

    summary_log = os.path.join(
        ARTIFACTS_DIR,
        f"{dataset}_{model}_summary.log"
    )

    logging.basicConfig(
        filename=summary_log,
        level=logging.INFO,
        format='%(message)s',
        filemode='w'
    )

    write_standard_section(poly_config, sigma, relu_log_file, poly_dir)
    write_standard_ensemble_section(dataset, model, poly_config, sigma)
    write_cluster_section(poly_config, sigma, mode="full", label="Full", poly_dir=poly_dir)
    write_cluster_section(poly_config, sigma, mode="slice", label="Slice", poly_dir=poly_dir)
    write_ensemble_cluster_section(poly_config, sigma, poly_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(
        dataset=args.dataset,
        model=args.model,
        degree=args.degree,
        clip=args.clip,
        pbit=args.pbit,
        penalty=args.penalty,
        sigma=args.sigma,
    )
