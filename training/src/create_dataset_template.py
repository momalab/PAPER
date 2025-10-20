import argparse
import json
import os
import pickle
import re
import numpy as np

from utils import ARTIFACTS_DIR, DATASET_DIR, DATASET_JSON_PATH


def format_floats(obj, precision):
    """
    Format floating-point values in nested structures to scientific notation.

    Args:
        obj (Any): Input object, which may be a float, list, dict, or nested structure.
        precision (int): Number of decimal places for formatting.

    Returns:
        Any: Object with floats formatted as strings in scientific notation.
    """

    if isinstance(obj, (float, np.float32, np.float64)):
        return f"{obj:.{precision}e}"
    elif isinstance(obj, list):
        return [format_floats(item, precision) for item in obj]
    elif isinstance(obj, dict):
        return {key: format_floats(value, precision) for key, value in obj.items()}
    return obj


def dataloader_to_json(dataloader, output_path, precision=20):
    """
    Convert a PyTorch dataloader into a JSON file representation.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader object containing (input, output) batches.
        output_path (str): File path to save the generated JSON file.
        precision (int, optional): Decimal precision for floating-point numbers. Default is 20.

    Returns:
        list[dict]: Formatted list of dictionaries containing input and output data.
    """

    json_data = []
    for x, y in dataloader:
        json_data.append({
            "input": x.numpy().tolist(),
            "output": [y]
        })
    formatted_json_data = format_floats(json_data, precision)

    with open(output_path, "w") as f:
        json_string = json.dumps(formatted_json_data, indent=2)
        json_string = re.sub(r'"(-?\d+\.\d+e[+-]?\d+)"', r'\1', json_string)
        f.write(json_string)
    return formatted_json_data


def main(dataset, filename):
    """
    Convert a serialized dataloader file into a JSON dataset file.

    Args:
        dataset (str): Dataset name, used to determine directory paths.
        filename (str): Name of the serialized dataloader file.

    Returns:
        list[dict]: JSON-formatted dataset representation.
    """

    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, dataset)
    dataloader_path = os.path.join(dataset_path, filename)
    json_path = f"{ARTIFACTS_DIR}/{DATASET_JSON_PATH}/{dataset}"
    os.makedirs(json_path, exist_ok=True)
    output_json_path = os.path.join(json_path, os.path.splitext(filename)[0] + ".json")
    
    with open(dataloader_path, 'rb') as f:
        dataloader = pickle.load(f)
    
    data_json = dataloader_to_json(dataloader, output_json_path)
    return data_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()
    main(dataset=args.dataset, filename=args.filename)