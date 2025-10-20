import json
import os
import re
import numpy as np

import torch
from torch import nn
from torch.fx import symbolic_trace

from arguments import parse_arguments
from models import get_model, replace_model
from poly_activation import PolyActivation
from utils import JSON_MODEL_PATH


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


def get_layer_template(layer_type):
    """
    Return a default template dictionary for a specific layer type.

    Args:
        layer_type (str): Layer type identifier.

    Returns:
        dict or None: Template dictionary defining the layer structure or None if invalid.
    """

    templates = {
        "CONV2D": {
            "id": None,
            "layer": "CONV2D",
            "previous": [],
            "next": [],
            "kernel": None,
            "bias": None,
            "stride": [],
            "padding": []
        },
        "BATCHNORM2D": {
            "id": None,
            "layer": "BATCHNORM2D",
            "previous": [],
            "next": [],
            "mu": None,
            "var": None,
            "gamma": None,
            "beta": None,
            "epsilon": None
        },
        "POLY": {
            "id": None,
            "layer": "POLY",
            "previous": [],
            "next": [],
            "coeff": []
        },
        "AVGPOOL2D": {
            "id": None,
            "layer": "AVGPOOL2D",
            "previous": [],
            "next": [],
            "kernel": [],
            "stride": [],
            "padding": [],
            "divisor": None
        },
        "LINEAR": {
            "id": None,
            "layer": "LINEAR",
            "previous": [],
            "next": [],
            "kernel": None,
            "bias": None
        },
        "ADD": {
            "id": None,
            "layer": "ADD",
            "previous": [],
            "next": []
        }
    }
    return templates.get(layer_type, None)


def get_model_properties(model, traced, node_name):
    """
    Extract numerical properties from a given model layer.

    Args:
        model (torch.nn.Module): The full model instance.
        traced (torch.fx.GraphModule): Symbolically traced model for graph analysis.
        node_name (str): Name of the target node in the traced graph.

    Returns:
        dict: Layer-specific parameters such as weights, biases, and hyperparameters.
    """

    model_dict = dict(model.named_modules())
    property_dictionary = {}

    for node in traced.graph.nodes:
        if node.name == node_name:
            module = model_dict.get(node.target)
            if isinstance(module, torch.nn.Conv2d):
                conv_layer = module
                property_dictionary = {
                    "kernel": conv_layer.weight.detach().cpu().numpy().tolist(),
                    "bias": conv_layer.bias.detach().cpu().numpy().tolist() if conv_layer.bias is not None else [],
                    "stride": list(conv_layer.stride),
                    "padding": list(conv_layer.padding)
                }
            elif isinstance(module, torch.nn.BatchNorm2d):
                bn_layer = module
                property_dictionary = {
                    "mu": bn_layer.running_mean.detach().cpu().numpy().tolist(),
                    "var": bn_layer.running_var.detach().cpu().numpy().tolist(),
                    "gamma": bn_layer.weight.detach().cpu().numpy().tolist() if bn_layer.weight is not None else [],
                    "beta": bn_layer.bias.detach().cpu().numpy().tolist() if bn_layer.bias is not None else [],
                    "epsilon": bn_layer.eps
                }
            elif isinstance(module, torch.nn.AvgPool2d):
                avgpool_layer = module
                kernel_size = list(avgpool_layer.kernel_size) if isinstance(avgpool_layer.kernel_size, tuple) else [
                    avgpool_layer.kernel_size, avgpool_layer.kernel_size]
                divisor = avgpool_layer.divisor_override if avgpool_layer.divisor_override is not None else (
                        kernel_size[0] * kernel_size[1])
                property_dictionary = {
                    "kernel": kernel_size,
                    "stride": list(avgpool_layer.stride) if isinstance(avgpool_layer.stride, tuple) else [
                        avgpool_layer.stride, avgpool_layer.stride],
                    "padding": list(avgpool_layer.padding) if isinstance(avgpool_layer.padding, tuple) else [
                        avgpool_layer.padding, avgpool_layer.padding],
                    "divisor": divisor
                }
            elif isinstance(module, torch.nn.Linear):
                linear_layer = module
                property_dictionary = {
                    "kernel": linear_layer.weight.detach().cpu().numpy().tolist(),
                    "bias": linear_layer.bias.detach().cpu().numpy().tolist() if linear_layer.bias is not None else []
                }
    return property_dictionary


def map_layer_types(layers):
    """
    Map layer names to standard layer type identifiers.

    Args:
        layers (dict): Dictionary mapping node names to numeric IDs.

    Returns:
        dict: Mapping of layer names to canonical layer type labels.
    """

    mapping = {}
    for key in layers:
        if "conv" in key:
            mapping[key] = "CONV2D"
        elif "poly" in key:
            mapping[key] = "POLY"
        elif "batch_norm" in key or "bn" in key:
            mapping[key] = "BATCHNORM2D"
        elif "pool" in key:
            mapping[key] = "AVGPOOL2D"
        elif "fc" in key or "linear" in key:
            mapping[key] = "LINEAR"
        elif "add" in key:
            mapping[key] = "ADD"
    return mapping


def get_adjacent_nodes(node_mapping, node):
    """
    Find the previous and next nodes for a given graph node.

    Args:
        node_mapping (dict): Mapping of node names to integer IDs.
        node (torch.fx.Node): Node from the traced computation graph.

    Returns:
        tuple[list[int], list[int]]: Lists of previous and next node indices.
    """

    prev_node = [node_mapping[n.name] for n in node.all_input_nodes]
    next_node = [node_mapping[n.name] for n in node.users if n.op != "output"]
    return prev_node, next_node


def initial_mapping(traced):
    """
    Generate an initial mapping of nodes and their connections from a traced model.

    Args:
        traced (torch.fx.GraphModule): Symbolically traced model.

    Returns:
        tuple[dict, dict]: 
            - Node name to index mapping.
            - Dictionary describing previous and next node connections.
    """

    node_mapping = {}
    node_connections = {}

    for idx, node in enumerate(traced.graph.nodes):
        if node.op not in ["output"]:
            node_mapping[node.name] = idx
            node_connections[idx] = {"previous": [], "next": []}

    for node in traced.graph.nodes:
        if node.op not in ["output"]:
            node_id = node_mapping[node.name]
            prev_node, next_node = get_adjacent_nodes(node_mapping, node)
            node_connections[node_id]["previous"] = prev_node
            node_connections[node_id]["next"] = next_node

    return node_mapping, node_connections


def final_mapping(node_mapping, node_connections, traced):
    """
    Refine node mappings and merge polynomial subgraphs.

    Args:
        node_mapping (dict): Mapping of node names to IDs from initial mapping.
        node_connections (dict): Node connectivity information.
        traced (torch.fx.GraphModule): Traced model graph.

    Returns:
        tuple[dict, dict, dict]:
            - Updated node mapping.
            - Updated connection mapping.
            - Polynomial details including entry, exit, and coefficient information.
    """

    new_node_mapping = {}
    new_id = 0

    skip_count = 12
    skip_flag = False
    skipped_nodes = []
    subgraph = False

    poly_count = 1
    poly_mapping = {}
    poly_details = {"entry": {}, "exit": {}, "coeff": {}}
    poly_coeff = []

    for node_name, idx in node_mapping.items():
        if not skip_flag and "pow" in node_name:
            skip_flag = True
            subgraph = True
            skipped_nodes = []
            poly_coeff = []

        if skip_flag:
            skipped_nodes.append(idx)
            poly_name = f"poly{poly_count}"
            poly_mapping[node_name] = poly_name

            if poly_name not in poly_details["entry"]:
                poly_details["entry"][poly_name] = node_connections[node_mapping[node_name]]["previous"]

            if "_tensor_constant" in node_name:
                for node in traced.graph.nodes:
                    if node.name == node_name:
                        constant = getattr(traced, node.target)
                        poly_coeff.append(constant.detach().numpy().item())

            if len(skipped_nodes) == skip_count:
                skip_flag = False
                poly_details["exit"][poly_name] = node_connections[node_mapping[node_name]]["next"]
                poly_details["coeff"][poly_name] = poly_coeff[::-1]
            else:
                continue

        if "flatten" in node_name:
            continue

        if "add" in node_name and subgraph:
            new_node_mapping[f"poly{poly_count}"] = new_id
            poly_count += 1
            subgraph = False
        else:
            new_node_mapping[node_name] = new_id
        new_id += 1

    new_node_connections = {}
    for node_id, node_name in enumerate(new_node_mapping.keys()):
        if node_name in node_mapping.keys():
            new_node_connections[node_id] = node_connections[node_mapping[node_name]]
        else:
            new_node_connections[node_id] = {"previous": poly_details["entry"][node_name],
                                             "next": poly_details["exit"][node_name]}

    id_to_name = {v: k for k, v in node_mapping.items()}
    for node_id, connections in new_node_connections.items():
        prev_nodes = connections["previous"]
        next_nodes = connections["next"]
        for idx in range(len(prev_nodes)):
            item_name = id_to_name[prev_nodes[idx]]
            if item_name in new_node_mapping.keys():
                prev_nodes[idx] = new_node_mapping[item_name]
            else:
                if item_name != "flatten":
                    prev_nodes[idx] = new_node_mapping[poly_mapping[item_name]]
                else:
                    prev_flatten = node_connections[prev_nodes[idx]]["previous"][0]
                    name_prev_flatten = id_to_name[prev_flatten]
                    if name_prev_flatten in new_node_mapping.keys():
                        prev_nodes[idx] = new_node_mapping[name_prev_flatten]
                    else:
                        prev_nodes[idx] = new_node_mapping[poly_mapping[name_prev_flatten]]
        for idx in range(len(next_nodes)):
            item_name = id_to_name[next_nodes[idx]]
            if item_name in new_node_mapping.keys():
                next_nodes[idx] = new_node_mapping[item_name]
            else:
                if item_name != "flatten":
                    next_nodes[idx] = new_node_mapping[poly_mapping[item_name]]
                else:
                    next_flatten = node_connections[next_nodes[idx]]["next"][0]
                    name_next_flatten = id_to_name[next_flatten]
                    if name_next_flatten in new_node_mapping.keys():
                        next_nodes[idx] = new_node_mapping[name_next_flatten]
                    else:
                        next_nodes[idx] = new_node_mapping[poly_mapping[name_next_flatten]]
        connections["previous"] = sorted(list(dict.fromkeys(prev_nodes)))
        connections["next"] = sorted(list(dict.fromkeys(next_nodes)))

    return new_node_mapping, new_node_connections, poly_details


def main():
    """
    Generate a structured JSON template representing the model architecture.
    """

    args = parse_arguments()
    model = get_model(args)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is None:
            m.bias = nn.Parameter(torch.zeros(m.out_channels, device=m.weight.device))

    poly_activation = PolyActivation(args.degree, args.clip, args.pbit)
    replace_model(model, poly_activation)
    model.load_state_dict(torch.load(args.model_save_path, map_location="cpu", weights_only=True))
    model.eval()

    traced = symbolic_trace(model)
    node_mapping, node_connections = initial_mapping(traced)
    # print("\nInitial Node Connections:")
    # for node_id, connections in node_connections.items():
    #     node_name = [name for name, id in node_mapping.items() if id == node_id][0]
    #     print(f'id: {node_id}, layer: {node_name}, previous: {connections["previous"]}, next: {connections["next"]}')

    new_node_mapping, new_node_connection, poly_details = final_mapping(node_mapping, node_connections, traced)
    # print("\nNew Node Connections:")
    # for node_id, connections in new_node_connection.items():
    #     node_name = [name for name, id in new_node_mapping.items() if id == node_id][0]
    #     print(f'id: {node_id}, layer: {node_name}, previous: {connections["previous"]}, next: {connections["next"]}')
    
    mapped_layers = map_layer_types(new_node_mapping)
    model_json = []
    for node_name, node_id in new_node_mapping.items():
        if node_name != 'x':
            layer_template = get_layer_template(mapped_layers[node_name])
            layer_template["id"] = node_id
            layer_template["previous"] = new_node_connection[node_id]["previous"]
            layer_template["next"] = new_node_connection[node_id]["next"]
            if "poly" not in node_name:
                model_properties = get_model_properties(model, traced, node_name)
            else:
                model_properties = {"coeff": poly_details["coeff"][node_name]}
            for key, value in model_properties.items():
                if key in layer_template:
                    layer_template[key] = value
            model_json.append(layer_template)

    formatted_model_json = format_floats(model_json, precision=20)
    os.makedirs(os.path.dirname(args.json_save_path), exist_ok=True)
    with open(args.json_save_path, "w") as f:
        json_string = json.dumps(formatted_model_json, indent=2)
        json_string = re.sub(r'"(-?\d+\.\d+e[+-]?\d+)"', r'\1', json_string)
        f.write(json_string)


if __name__ == "__main__":
    main()
