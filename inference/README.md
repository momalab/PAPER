# Privacy-Preserving Inference with PAPER

This directory contains the source code for the privacy-preserving inference of PAPER models.

# 🛠️ Dependencies

The versions listed below represent the earliest ones on which PAPER inference has been tested.
Testing was performed on **Ubuntu 22.04** and **Ubuntu 24.04**.
It may work with older versions or other environments, but these have not been verified.

### Required Tools and Libraries

| Dependency    | Minimum Version |
| ------------- | --------------- |
| Make          | 4.3             |
| CMake         | 3.22.1          |
| Git           | 2.34.1          |
| GNU C/C++     | C++17           |
| GMP           | 6.2.1           |
| nlohmann/json | 3.10.5          |
| Python        | 3.10.12         |
| NumPy         | 1.26.4          |
| 7-Zip         | 23.01           |
| curl          | 8.5.0           |

To install the required packages on Ubuntu:

```bash
sudo apt install make cmake git gcc g++ libgmp-dev nlohmann-json3-dev python3 python3-numpy p7zip-full curl
```

### Microsoft SEAL Library

PAPER requires the **Microsoft SEAL** library (version 4.1).
To install it, run:

```bash
cd third-party
bash install_seal.sh
```

For more details, see: [Microsoft SEAL GitHub Repository](https://github.com/microsoft/SEAL)

# 🧠 Programs

All executable programs are located in the `main` directory:

| Program                                     | Description                                                        |
| ------------------------------------------- | ------------------------------------------------------------------ |
| [`model_plain`](main/model_plain.cpp)       | Plaintext single-model runner (Python equivalent).                 |
| [`model_naive`](main/model_naive.cpp)       | Privacy-preserving single-model runner using naive encoding.       |
| [`model_adapt`](main/model_adapt.cpp)       | Privacy-preserving single-model runner using adaptive HW encoding. |
| [`ensemble_plain`](main/ensemble_plain.cpp) | Plaintext ensemble runner (Python equivalent).                     |
| [`ensemble_adapt`](main/ensemble_adapt.cpp) | Privacy-preserving ensemble runner using adaptive HW encoding.     |

> **Note:** The *ensemble* variants generalize the *model* ones.
> The *naive* version exists solely for validating FHE-related components.

# ⚙️ Compilation

To compile any program, run:
```bash
cd main
make <program> [PARAMETERS]
```
where `<program>` is the target program name, and `[PARAMETERS]` are optional build configuration flags.

### Build Parameters

| Parameter       | Options                    | Affected Programs |
| --------------- | -------------------------- | ----------------- |
| **LIBRARY**     | **mockup**, seal           | adapt, naive      |
| **CONVOLUTION** | classical, map, **mapmem** | adapt             |
| **RNS**         | **forward**, reverse, seal | adapt, naive      |

**Bold** entries indicate the default settings.

### Parameter Descriptions

* **LIBRARY**: Selects the FHE backend.
  * `seal`: Uses Microsoft SEAL for FHE operations.
  * `mockup`: Lightweight mockup for testing and debugging (insecure but faster).

* **CONVOLUTION**: Specifies the algorithm used for convolutions.
    * `classical`: Standard convolution algorithm.
    * `map`: Optimized for the *Weight Clustering* technique.
    * `mapmem`: Memory-optimized variant of `map`.

* **RNS**: Determines how the RNS moduli are generated.
    * `seal`: Default SEAL moduli set.
    * `forward` / `reverse`: Custom forward or reverse moduli generation.

### Examples

* `make ensemble_plain` compiles the `ensemble_plain` program.
* `make ensemble_adapt LIBRARY=seal` compiles `ensemble_adapt` using the Microsoft SEAL library for FHE operations, the default convolution algorithm (`mapmem`), and custom forward-mode RNS moduli generation.

# 🚀 Running Programs

To view the arguments for any program, simply run it without arguments:
```bash
./program.exe
```

### Command-Line Arguments:
* `<model.json>`: JSON file defining the model. Multiple files may be provided for ensembles.
* `<input.json>`: JSON file with input data for inference.
* `|N|`: $\log_2$ of the polynomial degree $N$.
* `|scale|`: $\log_2$ of the scaling factor $\Delta$.
* `|q0|,|q1|,...|qL|,|P|`: Bit sizes of the RNS moduli.

    Example: `1x54,1x23,18x44` → $|P| = 54$, $|q_L| = 23$, and $|q_{L-1}| = ... = |q_0| = 44$.

* `<rotation_steps>`: Galois keys for rotation steps (comma-separated, no spaces). Example: `1,2,4,8,16,24,31,32,64,-1,-2,-4,-8,-16,-32`.
* `[# iterations]` Optional number of inference runs (default: all input entries in `<input.json>`).

**Program-specific argument notes:**

* `model` programs handle a single model.
* `ensemble` programs handle one or more models.
* `plain` programs use only model and dataset files.
* `naive` and `adapt` programs require FHE-specific parameters.
* `<rotation_steps>` are used in `adapt` programs only.

The program corresponding to the PAPER implementation is `ensemble_adapt`.
For this program, `make` commands are available to run ResNet models.
For example, to run ResNet18 with two models, use:
```bash
make resnet18_32x32 MODEL1=model1.json MODEL2=model2.json DATASET=dataset.json NITERS=1
```

### Examples

#### Decompress Example Data
```bash
7z x cifar10.7z
7z x resnet18_cifar10_model1.7z
7z x resnet18_cifar10_model2.7z
```

#### Running Plaintext Ensemble
```bash
./ensemble_plain.exe resnet18_cifar10_model1.json resnet18_cifar10_model2.json cifar10.json
```

#### Running Privacy-Preserving Ensemble
```bash
make resnet18_32x32 MODEL1=resnet18_cifar10_model1.json MODEL2=resnet18_cifar10_model2.json DATASET=cifar10.json
```

> All model and dataset JSON files must be located in the current directory.

Due to large model and dataset sizes, only one example is included.
For the complete set of models and datasets used in the paper, see the instructions below.

# 📊 Reproduction of Results

This section describes how to reproduce the results reported in the evaluation of private inference in **PAPER**.

All experiments were conducted on a machine equipped with two AMD EPYC 64-core processors and 2 TB of RAM across 32 DDR4 DIMMs (2933 MT/s), running Ubuntu 22.04.5 LTS.

## Downloading Models and Datasets

We provide the models and datasets used in our experiments to facilitate reproduction of results.

From the root of this repository, run:

```bash
cd resources
bash download_models_and_datasets.sh
```

This script downloads a 13.4 GB ZIP archive containing all models and datasets used in the experiments. We recommend having at least **30 GB of free disk space** to allow for file extraction and organization.

After extraction, the script creates a `reproducibility` directory containing:

* `datasets`: CIFAR-10, CIFAR-100, and Tiny-ImageNet
* `models`:

  * VGG-16 on CIFAR-10
  * ResNet-18 and ResNet-20 on CIFAR-10 and CIFAR-100
  * ResNet-32 on CIFAR-10, CIFAR-100, and Tiny-ImageNet

Each dataset and model is stored as a `.7z` file to reduce storage requirements.

## Running a Single Case

To run an individual experiment, use the `experiment.py` script. For a full list of arguments, run:

```bash
python3 experiment.py --help
```

As an example, to measure the private inference time of ResNet-18 on CIFAR-10, with no ensemble ($M=1$) and *Slice Clustering* with $K=64$ clusters, run:

```bash
python3 experiment.py --path ../../resources/reproducibility/ --dataset cifar10 --model resnet18 --method slice_clustering -M 1 -K 64
```

This command generates a logfile named `cifar10_resnet18_slice_clustering_M_1_K_64_runtime.log` located in `resources/reproducibility/models/cifar10_resnet18/slice_clustering`.

The logfile contains the execution log and the measured inference time.

## Running Multiple Cases

The `reproduce.py` script invokes `experiment.py` to execute a series of experiments.
To run all experiments in **PAPER** (this may take several days), execute:

```bash
python3 reproduce.py --path ../../resources/reproducibility/
```

You can optionally restrict the experiments by dataset, model, and method. For example:

```bash
python3 reproduce.py --path ../../resources/reproducibility/ --dataset cifar10 --model resnet18 --method slice_clustering
```

This runs all *Slice Clustering* experiments for ResNet-18 on CIFAR-10.
Check `python3 reproduce.py --help` for additional options.

**Note**: By default, the experiments report inference time and peak memory. Accuracy can also be evaluated using private inference; however, this requires a large number of iterations per case, making exhaustive evaluation across all cases prohibitively slow. We therefore recommend sampling a subset of cases to verify that the accuracy is consistent with the results reported in the plaintext evaluation.

## Consolidating Results

To aggregate the results into an easy-to-read format, use the consolidation script:

```bash
python3 consolidate.py --path ../../resources/reproducibility/ --dataset cifar10 --model resnet18
```
This creates two files with tabulated results in `resources/reproducibility/`: `cifar10_resnet18_runtime.log` for inference time and `cifar10_resnet18_memory.log` for peak memory.

If the dataset and model are not specified, the script consolidates results for all available cases.

**Note:** By default, each experiment is executed once per configuration. To compute statistics such as standard deviation, run experiments with multiple iterations (e.g., `-n 5`).