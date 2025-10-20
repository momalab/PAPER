import argparse
import os
import re
import json
import glob
import ast

from utils import ACCURACY_DIR, ARTIFACTS_DIR, ENSEMBLE_DIR, JSON_MODEL_PATH, RESULTS_LOG_DIR

"""
Constants and regex patterns used for parsing experiment logs and model filenames.

Attributes
    ARG_NAMES: List of argument keys that define model configurations and JSON naming structure.
    BEST_LINE_PREFIX: Prefix string that identifies lines containing best model paths in log files.
    SEED_RE_STRICT: Regex pattern matching strictly formatted model filenames like 'best_model_<seed>.pth'.
    SEED_RE_LOOSE: Regex pattern matching more flexible filename variants containing a seed number.
    CSEED_RE: Regex for capturing clustering seed ('cseed') values embedded in model filenames.
    K_HEADER_RE: Regex pattern that identifies cluster log sections associated with specific K values.
    MSEED_ONLY_RE: Regex for extracting main seed values ('mseed') from ensemble log entries.
    FULL_CLUSTER_FILE_RE: Regex matching filenames for full-mode clustering summary logs.
    SLICE_CLUSTER_FILE_RE: Regex matching filenames for slice-mode clustering summary logs.
    ED_FILE_RE: Regex pattern for ensemble clustering summary logs containing degree, clip, pbit, penalty, sigma, and K values.
    BEST_MODEL_ED_RE: Regex for model filenames that encode seed, ensemble, and clustering details for ensemble clustering experiments.
    ENSEMBLE_SECTION_HEADER_RE: Regex used to identify ensemble section headers in ensemble clustering log files.
"""

ARG_NAMES = [
    "dataset", "model", "seed", "degree", "clip", "penalty", "sigma",
    "pbit", "slice", "k", "num_models"
]

BEST_LINE_PREFIX = "Best model path:"
SEED_RE_STRICT = re.compile(r"^best_model_(\d+)\.pth$")
SEED_RE_LOOSE = re.compile(r"best_model_(\d+)(?:_|\.pth)")
CSEED_RE = re.compile(r"(?:^|_)cseed_(\d+)(?:_|\.pth$)")
K_HEADER_RE = re.compile(r"Accuracy statistics:\s*\(K\s*=\s*(\d+)\)", re.I)

MSEED_ONLY_RE = re.compile(r"\(mseed\s*=\s*(\d+)\)", re.I)

FULL_CLUSTER_FILE_RE = re.compile(
    r"^accuracy_cluster_summary"
    r"_degree_(?P<degree>\d+)_clip_(?P<clip>\d+)_pbit_(?P<pbit>\d+)"
    r"_zeta_(?P<penalty>[^_]+)_sigma_(?P<sigma>[^_]+)_mode_full\.log$"
)

SLICE_CLUSTER_FILE_RE = re.compile(
    r"^accuracy_cluster_summary"
    r"_degree_(?P<degree>\d+)_clip_(?P<clip>\d+)_pbit_(?P<pbit>\d+)"
    r"_zeta_(?P<penalty>[^_]+)_sigma_(?P<sigma>[^_]+)_mode_slice\.log$"
)

ED_FILE_RE = re.compile(
    r"^accuracy_ensemble_cluster_summary"
    r"_degree_(?P<degree>\d+)_clip_(?P<clip>\d+)_pbit_(?P<pbit>\d+)_zeta_(?P<penalty>[^_]+)"
    r"_sigma_(?P<sigma>[^_]+)_k_(?P<k>[^.]+)\.log$"
)
BEST_MODEL_ED_RE = re.compile(
    r"^best_model_(?P<seed>\d+)"
    r"(?:_fused)?_k_(?P<k>\d+)_cseed_(?P<cseed>\d+)_ens_(?P<ens>\d+)\.pth$"
)
ENSEMBLE_SECTION_HEADER_RE = re.compile(
    r"Accuracy\s+statistics:?\s*\(\s*Ensemble\s*=\s*(\d+)\s*\)", re.I
)

def to_number(val):
    """
    Convert a string to a numeric value (int or float) when possible.

    Args:
        val (str): String value to convert.

    Returns:
        int or float or str:
            - Integer if the string represents an integer.
            - Float if the string represents a float.
            - Original string if conversion fails.
    """

    try:
        if re.fullmatch(r"[+-]?\d+", val):
            return int(val)
        return float(val)
    except ValueError:
        return val

def ensure_parent_dir(path):
    """
    Ensure that the parent directory of a given path exists.

    Args:
        path (str): File path for which the parent directory should be created.
    """

    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def upsert_mapping(mapping, key, value):
    """
    Insert or update a mapping with a new value.

    Args:
        mapping (dict): Dictionary to update.
        key (str): Key to insert or update.
        value (Any): Value or list of values to associate with the key.
    """

    if key not in mapping:
        mapping[key] = value
        return
    existing = mapping[key]
    if isinstance(existing, list):
        if value not in existing:
            existing.append(value)
    else:
        if existing != value:
            mapping[key] = [existing, value]

def json_name_for_params(params):
    """
    Generate a JSON filename representing model parameters.

    Args:
        params (dict): Dictionary containing model configuration parameters.

    Returns:
        str: A filename string encoding all key-value pairs of model parameters.
    """

    return "best_model_" + "_".join(
        f"{k}_{params[k] if params[k] is not None else 'None'}" for k in ARG_NAMES
    ) + ".json"

def write_method_json(mapping, method_name, dataset, model):
    """
    Write a mapping dictionary to a JSON file for a specific method.

    Args:
        mapping (dict): Dictionary mapping model paths to JSON file names.
        method_name (str): Name of the processing method.
        dataset (str): Dataset name.
        model (str): Model name.
    """

    folder = os.path.join(ARTIFACTS_DIR, JSON_MODEL_PATH, f"{dataset}_{model}")
    os.makedirs(folder, exist_ok=True)
    safe_name = method_name.lower().replace(" ", "_") + ".json"
    file_path = os.path.join(folder, safe_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

def extract_seeds_from_json_name(json_name):
    """
    Extract seed values from a JSON filename.

    Args:
        json_name (str): Filename string containing seed information.

    Returns:
        list[int]: List of extracted seed integers.
    """

    seeds = []
    for m in re.finditer(r"(?:^|_)seed_(\d+)(?:_|\.json$)", json_name):
        seeds.append(int(m.group(1)))
    return seeds

def _join_with_log_dir(filepath, maybe_rel):
    """
    Join a relative log path to the directory of the given file.

    Args:
        filepath (str): Path of the reference file.
        maybe_rel (str): Relative or absolute path to resolve.

    Returns:
        str: Absolute or normalized path after resolution.
    """

    base_dir = os.path.dirname(filepath)
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.normpath(os.path.join(base_dir, maybe_rel))

def normalize_model_path(p):
    """
    Normalize a model path to a consistent format under the artifacts directory.

    Args:
        p (str): File path to normalize.

    Returns:
        str: Normalized absolute path pointing to the model file.
    """

    p = os.path.normpath(p)
    if os.path.isabs(p):
        return p
    parts = p.split(os.sep)
    if "models" in parts:
        i = parts.index("models")
        rel = os.path.join(*parts[i:])
        return os.path.normpath(os.path.join(ARTIFACTS_DIR, rel))
    return os.path.normpath(os.path.join(ARTIFACTS_DIR, p))

def resolve_to_models_root(path_or_name, dataset, model, sigma_str=None, degree_str=None, clip_str=None, pbit_str=None, penalty_str=None):
    """
    Resolve a path or filename to a complete models directory path.

    Args:
        path_or_name (str): Partial or full path to the model file.
        dataset (str): Dataset name.
        model (str): Model name.
        sigma_str (str, optional): Sigma value as string.
        degree_str (str, optional): Degree value as string.
        clip_str (str, optional): Clip value as string.
        pbit_str (str, optional): Precision bit value as string.
        penalty_str (str, optional): Penalty value as string.

    Returns:
        str: Resolved absolute path to the model file.
    """

    p = os.path.normpath(path_or_name)
    parts = p.split(os.sep)

    if "models" in parts:
        return normalize_model_path(p)

    base = os.path.basename(p)
    search_roots = [os.path.join("models", dataset, f"{model}_poly")]
    specific = []
    if all(x is not None for x in [degree_str, clip_str, pbit_str, penalty_str, sigma_str]):
        specific_root = os.path.join(
            "models", dataset, f"{model}_poly",
            f"degree_{degree_str}", f"clip_{clip_str}", f"pbit_{pbit_str}",
            f"zeta_{penalty_str}", f"sigma_{sigma_str}"
        )
        specific.append(os.path.join(specific_root, base))

    partials = [
        os.path.join("models", dataset, f"{model}_poly", "**", base),
        os.path.join("models", dataset, "**", base),
    ]

    for pat in specific + partials:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return normalize_model_path(sorted(hits)[0])

    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            if base in filenames:
                return normalize_model_path(os.path.join(dirpath, base))

    if all(x is not None for x in [degree_str, clip_str, pbit_str, penalty_str, sigma_str]):
        fallback = os.path.join(
            "models", dataset, f"{model}_poly",
            f"degree_{degree_str}", f"clip_{clip_str}", f"pbit_{pbit_str}",
            f"zeta_{penalty_str}", f"sigma_{sigma_str}", base
        )
    else:
        fallback = os.path.join("models", dataset, f"{model}_poly", base)

    return normalize_model_path(fallback)

def extract_best_model_paths_from_log(filepath):
    """
    Extract paths to best models from a log file.

    Args:
        filepath (str): Path to the log file.

    Returns:
        list[str]: List of absolute normalized paths to the best model files.
    """

    paths = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(BEST_LINE_PREFIX):
                rel = line.split(BEST_LINE_PREFIX, 1)[1].strip()
                joined = _join_with_log_dir(filepath, rel)
                paths.append(normalize_model_path(joined))
    return paths

def extract_cluster_best_paths_and_k(filepath):
    """
    Extract best model paths and their corresponding K values from a cluster log.

    Args:
        filepath (str): Path to the cluster accuracy log.

    Returns:
        list[tuple[str, int]]: List of tuples containing path to the best model and K value associated with that section of the log.
    """

    out = []
    current_k = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = K_HEADER_RE.search(line)
            if m:
                current_k = int(m.group(1))
                continue
            if line.startswith(BEST_LINE_PREFIX):
                rel = line.split(BEST_LINE_PREFIX, 1)[1].strip()
                best_path = normalize_model_path(_join_with_log_dir(filepath, rel))
                out.append((best_path, current_k))
    return out

def params_from_model_path(model_path, dataset, model, allow_suffixes=False):
    """
    Extract model parameter values from its file path.

    Args:
        model_path (str): Path to the model file.
        dataset (str): Dataset name.
        model (str): Model name.
        allow_suffixes (bool, optional): Whether to allow suffixes in file name patterns.

    Returns:
        dict or None:
            Dictionary mapping argument names to parsed parameter values.
            Returns None if required values (like seed) cannot be extracted.
    """

    params = {k: None for k in ARG_NAMES}
    params["dataset"] = dataset
    params["model"] = model
    params["num_models"] = 1

    parts = os.path.normpath(model_path).split(os.sep)
    for seg in parts:
        if "_" not in seg:
            continue
        key, val = seg.split("_", 1)
        if key == "zeta":
            key = "penalty"
        if key in {"degree", "clip", "pbit", "penalty", "sigma", "k", "num_models", "slice", "k", "cseed"}:
            if key == "slice":
                v = str(val).lower()
                if v in {"1", "true", "yes"}:
                    params["slice"] = True
                elif v in {"0", "false", "no"}:
                    params["slice"] = False
                else:
                    params["slice"] = to_number(val)
            else:
                params[key] = to_number(val)

    base = os.path.basename(model_path)
    if allow_suffixes:
        m = SEED_RE_LOOSE.search(base)
        if not m:
            return None
        params["seed"] = int(m.group(1))
    else:
        m = SEED_RE_STRICT.fullmatch(base)
        if not m:
            return None
        params["seed"] = int(m.group(1))

    if params.get("cseed") is None:
        mc = CSEED_RE.search(base)
        if mc:
            params["cseed"] = int(mc.group(1))

    return params

def fill_from_filename(params, filename_regex, filename):
    """
    Fill missing parameter values from a filename based on a regex pattern.

    Args:
        params (dict): Existing parameters dictionary to update.
        filename_regex (re.Pattern): Compiled regex with named groups for parameters.
        filename (str): Filename to parse.

    Returns:
        dict: Updated parameter dictionary with new extracted values.
    """

    m = filename_regex.match(os.path.basename(filename))
    if not m:
        return params
    for key in ["degree", "clip", "pbit", "penalty", "sigma", "k"]:
        if key in m.groupdict():
            if params.get(key) is None:
                params[key] = to_number(m.group(key))
    return params

def process_standard_accuracy(log_dir, dataset, model):
    """
    Process accuracy summary logs and extract best model paths.

    Args:
        log_dir (str): Directory containing accuracy summary logs.
        dataset (str): Dataset name.
        model (str): Model name.

    Returns:
        dict: Mapping of best model paths to corresponding JSON configuration names.
    """

    mapping = {}
    pattern = os.path.join(
        log_dir,
        f"accuracy_summary_degree_*_clip_*_pbit_*_zeta_*_sigma_*.log"
    )

    for filepath in sorted(glob.glob(pattern)):
        for best_path in extract_best_model_paths_from_log(filepath):
            best_path = normalize_model_path(best_path)

            dname, fname = os.path.split(best_path)
            if "_fused" not in fname:
                fname = re.sub(r"(best_model_\d+)(?=\.pth$)", r"\1_fused", fname)
            fused_path = normalize_model_path(os.path.join(dname, fname))

            params = params_from_model_path(best_path, dataset, model, allow_suffixes=False)
            if params is None:
                continue

            params = fill_from_filename(params, FULL_CLUSTER_FILE_RE, os.path.basename(filepath))
            params = fill_from_filename(params, SLICE_CLUSTER_FILE_RE, os.path.basename(filepath))

            json_name = json_name_for_params(params)
            upsert_mapping(mapping, fused_path, json_name)
    return mapping

def process_full_clustering_after_polynomial_training(log_dir, dataset, model):
    """
    Process clustering logs (full mode) and extract best models after polynomial training.

    Args:
        log_dir (str): Directory containing clustering accuracy logs.
        dataset (str): Dataset name.
        model (str): Model name.

    Returns:
        dict: Mapping of best model paths to JSON filenames.
    """

    mapping = {}
    pattern = os.path.join(
        log_dir,
        f"accuracy_cluster_summary_degree_*_clip_*_pbit_*_zeta_*_sigma_*_mode_full.log"
    )
    for filepath in sorted(glob.glob(pattern)):
        for best_path, k_val in extract_cluster_best_paths_and_k(filepath):
            best_path = normalize_model_path(best_path)

            params = params_from_model_path(best_path, dataset, model, allow_suffixes=True)
            if params is None:
                continue
            if k_val is not None:
                params["k"] = k_val
            params["slice"] = False

            params = fill_from_filename(params, FULL_CLUSTER_FILE_RE, os.path.basename(filepath))

            json_name = json_name_for_params(params)
            upsert_mapping(mapping, best_path, json_name)
    return mapping

def process_slice_clustering_after_polynomial_training(log_dir, dataset, model):
    """
    Process clustering logs (slice mode) and extract best models after polynomial training.

    Args:
        log_dir (str): Directory containing slice clustering logs.
        dataset (str): Dataset name.
        model (str): Model name.

    Returns:
        dict: Mapping of best model paths to JSON filenames.
    """

    mapping = {}
    pattern = os.path.join(
        log_dir,
        f"accuracy_cluster_summary_degree_*_clip_*_pbit_*_zeta_*_sigma_*_mode_slice.log"
    )
    for filepath in sorted(glob.glob(pattern)):
        for best_path, k_val in extract_cluster_best_paths_and_k(filepath):
            best_path = normalize_model_path(best_path)

            params = params_from_model_path(best_path, dataset, model, allow_suffixes=True)
            if params is None:
                continue
            if k_val is not None:
                params["k"] = k_val
            params["slice"] = True

            params = fill_from_filename(params, SLICE_CLUSTER_FILE_RE, os.path.basename(filepath))

            json_name = json_name_for_params(params)
            upsert_mapping(mapping, best_path, json_name)
    return mapping

def process_standard_ensemble(ens_dir, dataset, model):
    """
    Process ensemble logs to identify the best-performing seed combinations.

    Args:
        ens_dir (str): Directory containing ensemble log files.
        dataset (str): Dataset name.
        model (str): Model name.

    Returns:
        dict: Mapping of best ensemble model paths to JSON filenames.
    """

    mapping = {}

    inv_file_re = re.compile(
        r"^logs_degree_(?P<degree>\d+)"
        r"_clip_(?P<clip>\d+)_pbit_(?P<pbit>\d+)"
        r"_zeta_(?P<penalty>[^_]+)_sigma_(?P<sigma>[^_]+)"
        r"_num_models_(?P<num_models>\d+)\.log$"
    )

    inv_entry_re = re.compile(
        r"Ensemble\s+\d+\s*:\s*Accuracy\s*=\s*([0-9.]+)\s*,\s*Models\s*=\s*\[(.*?)\]\s*$",
        re.I,
    )
    mseed_only_re = re.compile(r"\(mseed\s*=\s*(\d+)\)", re.I)

    pattern = os.path.join(
        ens_dir,
        "logs_degree_*_clip_*_pbit_*_zeta_*_sigma_*_num_models_*.log",
    )

    for filepath in sorted(glob.glob(pattern)):
        basename = os.path.basename(filepath)
        mfile = inv_file_re.match(basename)
        if not mfile:
            continue

        sigma_str = mfile.group("sigma")
        sigma_val = to_number(sigma_str)
        num_models = int(mfile.group("num_models"))

        best_acc = None
        best_seeds = None

        with open(filepath, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                mm = inv_entry_re.search(line)
                if not mm:
                    continue
                acc = float(mm.group(1))
                seeds_blob = mm.group(2)
                seeds = [int(s) for s in mseed_only_re.findall(seeds_blob)]
                if len(seeds) != num_models:
                    continue
                if best_acc is None or acc > best_acc:
                    best_acc = acc
                    best_seeds = seeds

        if not best_seeds:
            continue

        for mseed in best_seeds:
            candidates = []
            glob_pats = [
                os.path.join(ARTIFACTS_DIR,
                    "models", dataset, f"{model}_poly",
                    "degree_*", "clip_*", "pbit_*", "zeta_*",
                    f"sigma_{sigma_str}", f"best_model_{mseed}_fused.pth"
                ),
                os.path.join(ARTIFACTS_DIR,
                    "models", dataset, f"{model}_poly",
                    "degree_*", "clip_*", "pbit_*", "zeta_*",
                    f"sigma_{sigma_str}", f"best_model_{mseed}.pth"
                ),
            ]
            for gp in glob_pats:
                hits = sorted(glob.glob(gp))
                if hits:
                    candidates.extend(hits)

            if not candidates:
                root = os.path.join(ARTIFACTS_DIR, "models", dataset, f"{model}_poly")
                for dirpath, _, filenames in os.walk(root):
                    if f"sigma_{sigma_str}" not in dirpath:
                        continue
                    for fn in filenames:
                        if fn.startswith(f"best_model_{mseed}") and fn.endswith(".pth"):
                            candidates.append(os.path.join(dirpath, fn))

            if not candidates:
                continue

            chosen = next((c for c in candidates if c.endswith("_fused.pth")), candidates[0])
            chosen = normalize_model_path(chosen)
            dname, fname = os.path.split(chosen)
            if not fname.endswith("_fused.pth"):
                fname = re.sub(r"(best_model_\d+)(?=\.pth$)", r"\1_fused", fname)
                full_path = normalize_model_path(os.path.join(dname, fname))
            else:
                full_path = chosen

            params = params_from_model_path(full_path, dataset, model, allow_suffixes=True)
            if params is None:
                continue

            params["num_models"] = num_models
            if params.get("sigma") is None:
                params["sigma"] = sigma_val

            json_value = json_name_for_params(params)
            upsert_mapping(mapping, full_path, json_value)

    return mapping

def process_ensemble_ed_clustering(log_dir, dataset, model):
    """
    Process ensemble clustering logs and extract best model paths for each ensemble size.

    Args:
        log_dir (str): Directory containing ensemble cluster accuracy logs.
        dataset (str): Dataset name.
        model (str): Model name.

    Returns:
        dict: Mapping of best model file paths to JSON filenames for ensemble clustering.
    """

    mapping = {}
    pattern = os.path.join(
        log_dir,
        f"accuracy_ensemble_cluster_summary_degree_*_clip_*_pbit_*_zeta_*_sigma_*_k_*.log"
    )

    for filepath in sorted(glob.glob(pattern)):
        base = os.path.basename(filepath)
        mfile = ED_FILE_RE.match(base)
        if not mfile:
            continue

        sigma_str = mfile.group("sigma")
        sigma_val = to_number(sigma_str)
        degree_str = mfile.group("degree")
        clip_str = mfile.group("clip")
        pbit_str = mfile.group("pbit")
        penalty_str = mfile.group("penalty")
        k_val = to_number(mfile.group("k"))

        current_E = None
        with open(filepath, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()

                mh = ENSEMBLE_SECTION_HEADER_RE.search(line)
                if mh:
                    current_E = int(mh.group(1))
                    continue

                if line.startswith("Best model path"):
                    try:
                        list_str = line.split("Best model path", 1)[1].strip()
                        if list_str.startswith(":"):
                            list_str = list_str[1:].strip()
                        model_list = ast.literal_eval(list_str)
                        if not isinstance(model_list, list):
                            continue
                    except Exception:
                        continue

                    for fname in model_list:
                        bm = BEST_MODEL_ED_RE.match(os.path.basename(fname))
                        if not bm:
                            continue

                        ens_in_name = int(bm.group("ens"))
                        if current_E is not None and ens_in_name != current_E:
                            continue

                        seed_val = int(bm.group("seed"))
                        cseed_val = int(bm.group("cseed"))
                        k_from_name = bm.group("k")
                        k_val = int(k_from_name) if k_from_name is not None else None

                        pre = _join_with_log_dir(filepath, fname)
                        full_path = resolve_to_models_root(
                            pre, dataset, model,
                            sigma_str=sigma_str,
                            degree_str=degree_str,
                            clip_str=clip_str,
                            pbit_str=pbit_str,
                            penalty_str=penalty_str
                        )

                        dname, bname = os.path.split(full_path)
                        if not bname.endswith("_fused.pth") and bname.startswith("best_model_") and bname.endswith(".pth"):
                            bname = re.sub(r"(best_model_\d+)(?=\.pth$)", r"\1_fused", bname)
                            full_path = normalize_model_path(os.path.join(dname, bname))

                        params = params_from_model_path(full_path, dataset, model, allow_suffixes=True)
                        if params is None:
                            continue

                        for key, val in [
                            ("sigma", sigma_val),
                            ("degree", to_number(degree_str)),
                            ("clip", to_number(clip_str)),
                            ("pbit", to_number(pbit_str)),
                            ("penalty", to_number(penalty_str)),
                            ("k", k_val),
                        ]:
                            if params.get(key) is None:
                                params[key] = val

                        params["k"] = params.get("k", k_val)
                        params["num_models"] = ens_in_name
                        params["cseed"] = cseed_val
                        params["seed"] = seed_val
                        params["slice"] = True

                        json_value = "best_model_" + "_".join(
                            f"{k}_{params.get(k) if params.get(k) is not None else 'None'}"
                            for k in ARG_NAMES
                        ) + ".json"

                        upsert_mapping(mapping, full_path, json_value)
    return mapping

def main(dataset, model):
    """
    Run all extraction routines and write JSON summaries for each method.

    Args:
        dataset (str): Dataset name.
        model (str): Model name.
    """

    log_dir = os.path.join(ARTIFACTS_DIR, RESULTS_LOG_DIR, ACCURACY_DIR, f"{dataset}_{model}_poly")
    ens_dir = os.path.join(ARTIFACTS_DIR, RESULTS_LOG_DIR, ENSEMBLE_DIR, f"{dataset}_{model}_poly")

    processors = [
        ("standard_accuracy", process_standard_accuracy, log_dir, dataset, model),
        ("full_clustering_after_polynomial_training", process_full_clustering_after_polynomial_training, log_dir, dataset, model),
        ("slice_clustering_after_polynomial_training", process_slice_clustering_after_polynomial_training, log_dir, dataset, model),
        ("standard_ensemble", process_standard_ensemble, ens_dir, dataset, model),
        ("ensemble_ed_clustering", process_ensemble_ed_clustering, log_dir, dataset, model),
    ]

    for name, func, directory, ds, mdl in processors:
        if ds is None:
            mapping = func(directory)
            ds = dataset
            mdl = model
        else:
            mapping = func(directory, ds, mdl)
        write_method_json(mapping, name, ds, mdl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(dataset=args.dataset, model=args.model)
