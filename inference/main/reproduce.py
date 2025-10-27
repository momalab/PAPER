import argparse
import os

program = "python3 experiment.py"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce PAPER experiments with configurable parameters.",
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

    # Evaluation: minimal / quick / normal
    parser.add_argument(
        "--evaluation", "-E",
        type=str,
        choices=["minimal", "quick", "normal"],
        default="normal",
        help="Define how many iterations in time experiments: {minimal, quick, normal}."
    )

    # Mode: accuracy / time / both
    parser.add_argument(
        "--mode", "-e",
        type=str,
        choices=["accuracy", "time", "both"],
        default="both",
        help="Tells which data to collect: {accuracy, time, both}."
    )

    # Dataset: cifar10 / cifar100
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["all", "cifar10", "cifar100"],
        default="all",
        help="Dataset to use: {all, cifar10, cifar100}."
    )

    # Model: resnet18 / resnet20 / resnet32
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["all", "resnet18", "resnet20", "resnet32"],
        default="all",
        help="Model architecture: {all, resnet18, resnet20, resnet32}."
    )

    # Method: standard / full_clustering / slice_clustering
    parser.add_argument(
        "--method", "-t",
        type=str,
        choices=["all", "standard", "full_clustering", "slice_clustering"],
        default="all",
        help="Experiment method: {all, standard, full_clustering, slice_clustering}."
    )

    # Number of iterations
    parser.add_argument(
        "--niters", "-n",
        type=int,
        default=-1,
        help="Number of iterations/inferences to perform. If 0, run for all entries."
    )

    args = parser.parse_args()

    if args.niters == -1:
        args.niters = {
            "minimal": 1,
            "quick": 3,
            "normal": 5
        }[args.evaluation]
    
    return args

def run(dataset, model, method, mode, args):
    nmodels = [1] if method == "full_clustering" else [1, 2, 4]
    nclusters = [0]
    skip_compilation = False
    niters = args.niters if mode == "time" else 0
    for m in nmodels:
        if method in ["full_clustering", "slice_clustering"]:
            nclusters = [2**i for i in (range(6,14) if m > 1 else range(0,11))]
        for k in nclusters:
            cmd = f"{program} {'-s ' if skip_compilation else ' '}--path {args.path} --mode {mode} --dataset {dataset} --model {model} --method {method} -M {m} -K {k} --niters {niters}"
            print(cmd)
            os.system(cmd)
            skip_compilation = True

def reproduce(args):
    datasets = ["cifar10", "cifar100"] if args.dataset == "all" else [args.dataset]
    models = ["resnet18", "resnet20", "resnet32"] if args.model == "all" else [args.model]
    methods = ["standard", "full_clustering", "slice_clustering"] if args.method == "all" else [args.method]
    modes = ["accuracy", "time"] if args.mode == "both" else [args.mode]
    for dataset in datasets:
        for model in models:
            for method in methods:
                for mode in modes:
                    run(dataset, model, method, mode, args)

if __name__ == "__main__":
    args = parse_args()
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    reproduce(args)