import os
import fnmatch
import subprocess
import sys

# parameters
root = "/data/paperview"
root_models = f"{root}/model_jsons"
root_datasets = f"{root}/dataset_jsons"
experiment = "ensemble_ed_clustering"
program = "ensemble_adapt"
compilation_parameters = "LIBRARY=seal"
sigmas = ["0.0"] # ["0.01", "0.05"]
ensembles = [4] # [2, 4]
cluster_sizes = [4096] # [64, 128, 256, 512, 1024, 2048, 4096, 8192]

dlogN = {
    "vgg16"   : "15",
    "resnet18": "15",
    "resnet20": "15",
    "resnet32": "16"
}

dscale = {
    "vgg16"   : "25",
    "resnet18": "22",
    "resnet20": "21",
    "resnet32": "26"
}
 
dmoduli = {
    "vgg16"   : "6x50,10x49,1x26,1x54",
    "resnet18": "18x44,1x23,1x54",
    "resnet20": "20x42,1x22,1x44",
    "resnet32": "32x52,1x27,1x54"
}

dsteps = {
    "vgg16"   : {
        "cifar10" : "1,2,4,8,16,31,32,33,62,64,66,124,128,132,248,256,264,496,512,528,15856,15872,15888,16120,16128,16136,16252,16256,16260,16318,16320,16322,16351,16352,16353,16368,16376,16380,16382,16383",
        "cifar100": "1,2,4,8,16,31,32,33,62,64,66,124,128,132,248,256,264,496,512,528,15856,15872,15888,16120,16128,16136,16252,16256,16260,16318,16320,16322,16351,16352,16353,16368,16376,16380,16382,16383"
    },
    "resnet18": {
        "cifar10" : "1,2,4,8,16,24,31,32,33,62,64,66,124,128,132,248,256,264,272,280,512,520,528,536,768,776,784,792,16120,16128,16136,16252,16256,16260,16318,16320,16322,16351,16352,16353,16376,16380,16382,16383",
        "cifar100": "1,2,4,8,16,24,31,32,33,62,64,66,124,128,132,248,256,264,272,280,512,520,528,536,768,776,784,792,16120,16128,16136,16252,16256,16260,16318,16320,16322,16351,16352,16353,16376,16380,16382,16383"
    },
    "resnet20": {
        "cifar10" : "1,2,4,8,12,16,31,32,33,62,64,66,124,128,132,136,140,256,260,264,268,384,388,392,396,512,528,16252,16256,16260,16318,16320,16322,16351,16352,16353,16380,16382,16383",
        "cifar100": "1,2,4,8,12,16,31,32,33,62,64,66,124,128,132,136,140,256,260,264,268,384,388,392,396,512,528,16252,16256,16260,16318,16320,16322,16351,16352,16353,16380,16382,16383"
    },
    "resnet32": {
        "cifar10" : "1,2,4,8,12,16,31,32,33,62,64,66,124,128,132,136,140,256,260,264,268,384,388,392,396,512,528,32636,32640,32644,32702,32704,32706,32735,32736,32737,32764,32766,32767",
        "cifar100": "1,2,4,8,12,16,31,32,33,62,64,66,124,128,132,136,140,256,260,264,268,384,388,392,396,512,528,32636,32640,32644,32702,32704,32706,32735,32736,32737,32764,32766,32767",
        "tiny"    : "1,2,4,8,12,16,20,24,28,32,63,64,65,126,128,130,252,256,260,264,268,272,276,280,284,512,516,520,524,528,532,536,540,768,772,776,780,784,788,792,796,1024,1028,1032,1036,1040,1044,1048,1052,1280,1284,1288,1292,1296,1300,1304,1308,1536,1540,1544,1548,1552,1556,1560,1564,1792,1796,1800,1804,1808,1812,1816,1820,2048,2080,32508,32512,32516,32638,32640,32642,32703,32704,32705,32764,32766,32767"
    }
}

def list_files(path, sigma, nmodels, kf):
    try:
        # List all entries in the given path
        entries = os.listdir(path)
        
        # Filter only files (exclude directories)
        return [
            f for f in entries
            if os.path.isfile(os.path.join(path, f)) 
            and fnmatch.fnmatch(f, f"*_sigma_{sigma}_*_kf_{kf}_num_models_{nmodels}.json")
        ]

    except FileNotFoundError:
        print(f"Error: The path '{path}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied for path '{path}'.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: python3 {os.path.basename(__file__)} <model> <dataset> <niters>")
        sys.exit(1)

    model = sys.argv[1]
    dataset = sys.argv[2]
    niters = sys.argv[3]

    logN = dlogN[model]
    scale = dscale[model]
    moduli = dmoduli[model]
    steps = dsteps[model][dataset]

    # Compile
    os.system(f"make {program} {compilation_parameters}")

    # copy dataset to local    
    file_dataset = f"ensemble_testdataset.json"
    os.system(f"cp {root_datasets}/{dataset}/{file_dataset} .")

    # run experiment
    model_path = f"{root_models}/{dataset}_{model}/{experiment}/"
    for s in sigmas:
        for e in ensembles:
            for kf in cluster_sizes:
                    # print
                print(f"Running experiment with ensemble={e}, kf={kf}, sigma={s}")

                # list files
                filenames = list_files(model_path, s, e, kf)

                # copy files
                for filename in filenames:
                    os.system(f"cp {model_path}/{filename} .")

                # run experiment
                name = f"ensemble_{e}_kf_{kf}_sigma_{s}"
                log_run = f"{name}_run.log"
                log_time = f"{name}_time.log"
                cmd = ["/usr/bin/time", "-v", f"./{program}.exe"]
                for i,filename in enumerate(filenames):
                    cmd.append(f"{filename}")
                cmd.append(f"{file_dataset}")
                cmd.append(logN)
                cmd.append(scale)
                cmd.append(moduli)
                cmd.append(steps)
                cmd.append(niters)
                print(cmd)
                process = subprocess.Popen(cmd, stdout=open(log_run, "w"), stderr=open(log_time, "w"))
                process.wait()

                # delete copied model files
                for filename in filenames:
                    os.remove(filename)

                # move logs
                os.rename(log_run, f"{model_path}/{log_run}")
                os.rename(log_time, f"{model_path}/{log_time}")

    # delete copied dataset
    os.remove(file_dataset)
    