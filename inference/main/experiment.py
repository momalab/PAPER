import argparse
import fnmatch
import os
import subprocess

program = "ensemble_adapt"

logN = {
    "resnet18": "15",
    "resnet20": "15",
    "resnet32": "16"
}

scale = {
    "resnet18": "22",
    "resnet20": "21",
    "resnet32": "26"
}
 
moduli = {
    "resnet18": "18x44,1x23,1x54",
    "resnet20": "20x42,1x22,1x44",
    "resnet32": "32x52,1x27,1x54"
}

steps = {
    "resnet18": "1,2,4,8,16,24,31,32,33,62,64,66,124,128,132,248,256,264,272,280,512,520,528,536,768,776,784,792,16120,16128,16136,16252,16256,16260,16318,16320,16322,16351,16352,16353,16376,16380,16382,16383",
    "resnet20": "1,2,4,8,12,16,31,32,33,62,64,66,124,128,132,136,140,256,260,264,268,384,388,392,396,512,528,16252,16256,16260,16318,16320,16322,16351,16352,16353,16380,16382,16383",
    "resnet32": "1,2,4,8,12,16,31,32,33,62,64,66,124,128,132,136,140,256,260,264,268,384,388,392,396,512,528,32636,32640,32644,32702,32704,32706,32735,32736,32737,32764,32766,32767"
}

def list_files(path, filter):
    try:
        # List all entries in the given path
        entries = os.listdir(path)
        
        # Filter only files (exclude directories)
        return [
            f for f in entries
            if os.path.isfile(os.path.join(path, f)) 
            and fnmatch.fnmatch(f, filter)
        ]

    except FileNotFoundError:
        print(f"Error: The path '{path}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied for path '{path}'.")

def extract_seed(model_file):
    # find the integer represented by X "*_seed_X_*"
    parts = model_file.split('_')
    for i in range(len(parts)-1):
        if parts[i] == 'seed':
            return int(parts[i+1])
    return None

def fancy(text):
    return {
        "cifar10" : "CIFAR-10",
        "cifar100" : "CIFAR-100",
        "resnet18" : "ResNet-18",
        "resnet20" : "ResNet-20",
        "resnet32" : "ResNet-32",
        "standard" : "Standard",
        "full_clustering" : "Full Clustering",
        "slice_clustering" : "Slice Clustering"
    }[text]

def filter_dataset(seed, ext = "*"):
    return f"testdataset_seed_{seed}.{ext}" if seed is not None else f"ensemble_testdataset.{ext}"

def filter_model(m, k = None, ext = "*"):
    filter = "*"
    if k is not None and k:
        filter += f"_k_{k}"
    filter += f"_num_models_{m}.{ext}"
    return filter
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PAPER experiment with configurable parameters.",
        add_help=False
    )

    # re-add only the long form --help
    parser.add_argument(
        "--help", "-h",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )

    # Boolean flag
    parser.add_argument(
        "--skip-compilation", "-s",
        action="store_true",
        help="Run experiment without recompiling the program. Do not use if you are not sure of what you are doing."
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
        required=True,
        help="Tells which data to collect: {accuracy, time, both}."
    )

    # Dataset: cifar10 / cifar100
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["cifar10", "cifar100"],
        required=True,
        help="Dataset to use: {cifar10, cifar100}."
    )

    # Model: resnet18 / resnet20 / resnet32
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["resnet18", "resnet20", "resnet32"],
        required=True,
        help="Model architecture: {resnet18, resnet20, resnet32}."
    )

    # Method: standard / full_clustering / slice_clustering
    parser.add_argument(
        "--method", "-t",
        type=str,
        choices=["standard", "full_clustering", "slice_clustering"],
        required=True,
        help="Experiment method: {standard, full_clustering, slice_clustering}."
    )

    # Number of models
    parser.add_argument(
        "--nmodels", "-M",
        type=int,
        required=True,
        help="Number of models in the ensemble."
    )

    # Number of clusters
    parser.add_argument(
        "--nclusters", "-K",
        type=int,
        default=0,
        help="Number of clusters for full or slice clustering."
    )

    # Number of iterations
    parser.add_argument(
        "--niters", "-n",
        type=int,
        default=0,
        help="Number of iterations/inferences to perform. If 0 or not set, run for all entries."
    )

    # Convolution : classical / map / mapmem
    parser.add_argument(
        "--convolution", "-c",
        type=str,
        choices=["classical", "map", "mapmem"],
        help="Convolution algorithm to use: {classical, map, mapmem} (default: classical if method is standard else mapmem). Ignored if --skip-compilation is used."
    )

    # Library: mockup / seal
    parser.add_argument(
        "--library", "-l",
        type=str,
        choices=["mockup", "seal"],
        help="Library to use: {mockup, seal} (default: mockup if mode is accuracy else seal). Ignored if --skip-compilation is used."
    )

    # RNS moduli: forward / reverse / seal
    parser.add_argument(
        "--rns", "-r",
        type=str,
        choices=["forward", "reverse", "seal"],
        default="forward",
        help="RNS moduli selection strategy: {forward, reverse, seal} (default: forward). Ignored if --skip-compilation is used."
    )

    args = parser.parse_args()

    # --- Logic for automatic defaults ---
    if not args.skip_compilation:
        if args.convolution is None:
            # Automatically select convolution algorithm based on method
            if args.method == "standard":
                args.convolution = "classical"
            else:
                args.convolution = "mapmem"

    if not args.skip_compilation:
        if args.library is None:
            # Automatically select library based on mode
            if args.mode == "accuracy":
                args.library = "mockup"
            else:
                args.library = "seal"
    else:
        # skip-compilation overrides library
        args.library = None

    return args

def compile(args):
    print("Compiling ... ", end="", flush=True)
    os.system(f"make {program} CONVOLUTION={args.convolution} LIBRARY={args.library} RNS={args.rns} > /dev/null 2>&1")
    print("ok")

def run(args):
    print(f"Experiment {fancy(args.dataset)} x {fancy(args.model)} {fancy(args.method)} M={args.nmodels} {f'K={args.nclusters}' if args.nclusters is not None and args.nclusters else ''}")

    # find model file(s)
    print("Searching model(s) ... ", end="", flush=True)
    model_path = f"{args.path}/models/{args.dataset}_{args.model}/{args.method}"
    model_files = list_files(model_path, filter_model(args.nmodels, args.nclusters, ext = "7z"))
    if len(model_files) != args.nmodels:
        print(f"Expected {args.nmodels} files but found {len(model_files)}")
        exit(1)
    print("ok")
    
    # find dataset
    print("Searching dataset ... ", end="", flush=True)
    dataset_path = f"{args.path}/datasets/{args.dataset}"
    seed = extract_seed(model_files[0]) if len(model_files) == 1 else None
    dataset_file = list_files(dataset_path, filter_dataset(seed, ext = "7z"))
    if len(dataset_file) != 1:
        print(f"Expected 1 dataset file but found {len(dataset_file)}")
        exit(1)
    print("ok")

    # extract models to the local directory
    for model_file in model_files:
        print(f"Extracting {model_file} ... ", end="", flush=True)
        os.system(f"7z -y -bd x {model_path}/{model_file} -o. > /dev/null 2>&1")
        print("ok")

    # extract dataset to the local directory
    print(f"Extracting {dataset_file[0]} ... ", end="", flush=True)
    os.system(f"7z -y -bd x {dataset_path}/{dataset_file[0]} -o. > /dev/null 2>&1")
    print("ok")
    
    # run experiment
    print("Running (this may take a while) ... ", end="", flush=True)
    logname = f"{args.dataset}_{args.model}_{args.method}_M_{args.nmodels}{f'_K_{args.nclusters}' if args.nclusters is not None and args.nclusters else ''}"
    log_accuracy = f"{model_path}/{logname}_accuracy.log"
    log_runtime = f"{model_path}/{logname}_runtime.log"
    log_memory = f"{model_path}/{logname}_memory.log"
    cmd = ["/usr/bin/time", "-v", f"./{program}.exe"]
    for model_file in model_files:
        json_file = model_file[:-2] + "json"
        cmd.append(f"{json_file}")
    cmd.append(f"{dataset_file[0][:-2] + "json"}")
    cmd.append(logN[args.model])
    cmd.append(scale[args.model])
    cmd.append(moduli[args.model])
    cmd.append(steps[args.model])
    cmd.append(str(args.niters))
    if args.mode == "accuracy":
        process = subprocess.Popen(cmd[2:], stdout=open(log_accuracy, "w"))
    else:
        process = subprocess.Popen(cmd, stdout=open(log_runtime, "w"), stderr=open(log_memory, "w"))
    process.wait()
    if args.mode == "both":
        # copy log_runtime to log_accuracy
        os.system(f"cp {log_runtime} {log_accuracy}")
    print("ok")

    print(f"Removing JSON files ... ", end="", flush=True)
    # delete JSON model files
    for model_file in model_files:
        json_file = model_file[:-2] + "json"
        os.remove(json_file)

    # delete dataset file
    json_dataset_file = dataset_file[0][:-2] + "json"
    os.remove(json_dataset_file)
    print("ok")

if __name__ == "__main__":
    args = parse_args()
    # print("Arguments:")
    # for key, value in vars(args).items():
    #     print(f"  {key}: {value}")
    if not args.skip_compilation:
        compile(args)
    run(args)
