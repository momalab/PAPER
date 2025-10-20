import argparse
import os
import json
import subprocess
import concurrent.futures

from utils import ARTIFACTS_DIR, DATASET_DIR, JSON_MODEL_PATH


def run_cmd(cmd):
    """
    Run a command and return True on success.

    Args:
        cmd (list[str]): Command as a list of strings.

    Returns:
        bool: True if the command exits successfully.
    """
    subprocess.run(cmd, check=True)
    return True


def build_model_commands(dataset, model):
    """
    Build subprocess commands to generate model templates from JSON specs.

    Args:
        dataset (str): Dataset name.
        model (str): Base model name without the _poly suffix.

    Returns:
        list[list[str]]: Commands to execute.
    """
    cmds = []
    model_poly = f"{model}_poly"
    folder = f"{ARTIFACTS_DIR}/{JSON_MODEL_PATH}/{dataset}_{model}"
    os.makedirs(folder, exist_ok=True)

    json_files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])

    for filename in json_files:
        base_name = os.path.splitext(filename)[0]
        file_path = os.path.join(folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    full_json_path = f"{folder}/{base_name}/{v}"
                    cmd = [
                        "python", "src/create_model_template.py",
                        "--dataset", dataset,
                        "--model", model_poly,
                        "--model_save_path", key,
                        "--json_save_path", full_json_path,
                    ]
                    cmds.append(cmd)
            else:
                full_json_path = f"{folder}/{base_name}/{value}"
                cmd = [
                    "python", "src/create_model_template.py",
                    "--dataset", dataset,
                    "--model", model_poly,
                    "--model_save_path", key,
                    "--json_save_path", full_json_path,
                ]
                cmds.append(cmd)
    return cmds


def build_dataset_commands(dataset):
    """
    Build subprocess commands to generate dataset templates from cached files.

    Args:
        dataset (str): Dataset name.

    Returns:
        list[list[str]]: Commands to execute.
    """
    cmds = []
    target_dir = os.path.join(f"{ARTIFACTS_DIR}", f"{DATASET_DIR}", dataset)
    os.makedirs(target_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(target_dir)
        if os.path.isfile(os.path.join(target_dir, f))
    )

    for fname in files:
        cmd = [
            "python", "src/create_dataset_template.py",
            "--dataset", dataset,
            "--filename", fname,
        ]
        cmds.append(cmd)
    return cmds


def run_commands(cmds, max_workers):
    """
    Execute commands in parallel and record results.

    Args:
        cmds (list[list[str]]): Commands to run.
        max_workers (int): Maximum parallel workers.

    Returns:
        list[tuple]: Tuples of (cmd, success, error_message).
    """
    results = []
    if not cmds:
        return results
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_cmd = {ex.submit(run_cmd, cmd): cmd for cmd in cmds}
        for fut in concurrent.futures.as_completed(future_to_cmd):
            cmd = future_to_cmd[fut]
            try:
                fut.result()
                results.append((cmd, True, None))
            except Exception as e:
                results.append((cmd, False, str(e)))
                print("Failed", " ".join(cmd), "error", e)
    return results


def main(dataset, model, workers):
    """
    Build command lists then run them in parallel.

    Args:
        dataset (str): Dataset name.
        model (str): Base model name.
        workers (int): Max parallel workers.
    """
    model_cmds = build_model_commands(dataset, model)
    data_cmds = build_dataset_commands(dataset)

    _ = run_commands(model_cmds, max_workers=workers)
    _ = run_commands(data_cmds, max_workers=workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(dataset=args.dataset, model=args.model, workers=args.workers)
