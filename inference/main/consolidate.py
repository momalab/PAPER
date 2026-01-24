import argparse
import numpy as np
import os

all_dataset_model_pairs = [
    "cifar10_vgg16", "cifar10_resnet18", "cifar10_resnet20", "cifar10_resnet32",
    "cifar100_resnet18", "cifar100_resnet20", "cifar100_resnet32", "tiny_resnet32"
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidate LOG files into tables.",
        add_help=False
    )

    # re-add only the long form --help
    parser.add_argument(
        "--help", "-h",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )

    # Path
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to the root directory containing JSON models and datasets."
    )

    # Mode: accuracy / time / both
    parser.add_argument(
        "--mode", "-e",
        type=str,
        choices=["accuracy", "time", "both"],
        default="time",
        help="Tells which data to collect: {accuracy, time, both}. (default: time)"
    )

    # Dataset: cifar10 / cifar100 / tiny
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["all", "cifar10", "cifar100", "tiny"],
        default="all",
        help="Dataset to use: {all, cifar10, cifar100, tiny}. (default: all)"
    )

    # Model: vgg16 / resnet18 / resnet20 / resnet32
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["all", "vgg16", "resnet18", "resnet20", "resnet32"],
        default="all",
        help="Model architecture: {all, vgg16, resnet18, resnet20, resnet32}. (default: all)"
    )

    args = parser.parse_args()
    
    return args

def parse_log_memory(logfile):
    parsed_lines = []
    with open(logfile, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Maximum resident set size (kbytes):"):
                parsed_lines.append(line)
    return parsed_lines

def parse_log_runtime(logfile):
    parsed_lines = []
    with open(logfile, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Accuracy"):
                parsed_lines.append(line)
    parsed_lines = parsed_lines[:-1] # remove last line
    return parsed_lines

def get_accuracy(logfile):
    accuracies = []
    parsed_lines = parse_log_runtime(logfile)
    for line in parsed_lines:
        accuracies.append(float(line.split(' ')[1]))
    return accuracies

def get_memory(logfile):
    memory = []
    parsed_lines = parse_log_memory(logfile)
    for line in parsed_lines:
        memory.append(float(line.split(' ')[-1]))
    return memory

def get_runtime(logfile):
    runtimes = []
    parsed_lines = parse_log_runtime(logfile)
    for line in parsed_lines:
        runtimes.append(float(line.split(' ')[4][:-1]))
    return runtimes

def summarize_logfile(logfile, mode):
    if mode == "accuracy":
        scale = 100
        measurement = get_accuracy(logfile)
    elif mode == "runtime":
        scale = 1.0
        measurement = get_runtime(logfile)
    elif mode == "memory":
        scale = 1.0 / 2**20
        measurement = get_memory(logfile)
    else:
        raise("Invalid mode in 'summarize_logfile'")

    if measurement:
        mean = np.mean(measurement) * scale
        std  = np.std(measurement)  * scale
        min  = np.min(measurement)  * scale
        max  = np.max(measurement)  * scale
        return (mean, std, min, max)
    return None

def measurement_header(mode):
    return {
        "accuracy" : "accuracy (%)",
        "runtime"  : "runtime (s) ",
        "memory"   : "memory (GB) "
    }[mode]

def fancy(text):
    return {
        "cifar10"  : "CIFAR-10",
        "cifar100" : "CIFAR-100",
        "tiny"     : "Tiny-ImageNet",
        "vgg16"    : "VGG-16",
        "resnet18" : "ResNet-18",
        "resnet20" : "ResNet-20",
        "resnet32" : "ResNet-32",
        "standard" : "Standard",
        "full_clustering" : "Full Clustering",
        "slice_clustering" : "Slice Clustering"
    }[text]

def build_table(path, dataset, model, mode, nmodels, output, method):
    title = fancy(method)
    print(title)
    directory = f"{path}/models/{dataset}_{model}/{method}"
    has_clusters = method != "standard"
    nclusters = {
        1: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        2: [64, 128, 256, 512, 1024, 2048, 4096, 8192],
        4: [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    }
    with open(output, 'a') as f:
        f.write(f"{title}\n")
        f.write(f"+----------+{'------+' if has_clusters else ''}-------------------+-------+-------+\n")
        f.write(f"| # models |{'  k   |' if has_clusters else ''}            {measurement_header(mode)}           |\n")
        f.write(f"+----------+{'------+' if has_clusters else ''}-------------------+-------+-------+\n")
        f.write(f"|          |{'      |' if has_clusters else ''}    (mean, std)    |  min  |  max  |\n")
        f.write(f"+----------+{'------+' if has_clusters else ''}-------------------+-------+-------+\n")

        for m in nmodels:
            ks = nclusters[m] if has_clusters else [0]
            for k in ks:
                line = f"|    {m:2d}    |{f' {k:4d} |' if has_clusters else ''}"
                try:
                    logfile = f"{dataset}_{model}_{method}_M_{m}_{f'K_{k}_' if has_clusters else ''}{mode}.log"
                    print(f"Processing {logfile}")
                    logfile = f"{directory}/{logfile}"
                    mean, std, min, max = summarize_logfile(logfile, mode)
                    if mode == "accuracy":
                        line +=  f"   ({mean:5.1f}, {std:4.1f})   | {min:5.1f} | {max:5.1f} |"
                    else:
                        line +=  f"    ({mean:4.0f}, {std:2.0f})     | {min:4.0f}  | {max:4.0f}  |"
                except:
                    line +=  f"     (N/A, N/A)    |  N/A  |  N/A  |"
                f.write(f"{line}\n")
            if has_clusters:
                f.write(f"+----------+------+-------------------+-------+-------+\n")
        if not has_clusters:
            f.write(f"+----------+-------------------+-------+-------+\n")
        f.write("\n")
        print()

def consolidate(path, dataset, model, mode):
    if mode == "time":
        consolidate(path, dataset, model, "runtime")
        consolidate(path, dataset, model, "memory")
        return
    
    # Summarize experimental results
    filename = f"{path}/{dataset}_{model}_summary_{mode}.log"
    print(f"Generating summary for {fancy(model)} on {fancy(dataset)}\n")
    open(filename, 'w').close() # clear file
    build_table(path, dataset, model, mode, nmodels = [1, 2, 4], output = filename, method = "standard")
    build_table(path, dataset, model, mode, nmodels = [1]      , output = filename, method = "full_clustering")
    build_table(path, dataset, model, mode, nmodels = [1, 2, 4], output = filename, method = "slice_clustering")

def dispatch(args):
    datasets = ["cifar10", "cifar100", "tiny"] if args.dataset == "all" else [args.dataset]
    models = ["vgg16", "resnet18", "resnet20", "resnet32"] if args.model == "all" else [args.model]
    modes = ["accuracy", "time"] if args.mode == "both" else [args.mode]
    for dataset in datasets:
        for model in models:
            if f"{dataset}_{model}" in all_dataset_model_pairs:
                for mode in modes:
                    consolidate(args.path, dataset, model, mode)

if __name__ == "__main__":
    args = parse_args()
    dispatch(args)