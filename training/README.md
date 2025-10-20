# 📑 Overview
This repository provides a complete experimental framework for training, evaluating, and compressing deep neural networks that use polynomial activation functions instead of conventional ReLU activations. It supports ResNet architectures on CIFAR-10 and CIFAR-100 datasets and provides end-to-end tools for polynomial coefficient generation, model training, clustering, and evaluation.

# 🖥️ System Requirements

* **GPU:** NVIDIA GPU with CUDA support (Experiments were conducted on NVIDIA A100 GPUs).
* **Python Version:** Python 3.12.7 (Confirmed compatibility).
* **CUDA Toolkit:** CUDA compilation tools (Tested with release 12.8, V12.8.93).

# 🛠️ Setup and Installation

### Create and Activate Virtual Environment

```bash
python -m venv paper
source paper/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_pytorch.txt
```

# 🚀 Experiment Workflow Overview

## 1. Polynomial Coefficient Generation

This step computes the polynomial activation coefficients that approximate the ReLU function using polynomial regression.

```bash
python src/polyfit.py --degree 2 --clip 2 --pbit 10
```

Details of all command line arguments are mentioned in `arguments.py`.

**Outputs:**

* The generated coefficients from this configuration are stored under:

  ```
  artifacts/poly_coeffs/
  └──deg_2_clip_2_pbit_10_coeffs.txt
  ```

## 2. Baseline Model Training (ReLU Network)

### Model Training

The framework supports training ResNet models (e.g., Resnet-18, Resnet-20, Resnet-32) on CIFAR-10 and CIFAR-100 datasets with ReLU activations. This example demonstrates training ResNet-20 on CIFAR-10 using five random seeds, which can be run in parallel across multiple GPUs or compute nodes for faster processing. 

> **Note:** For subsequent experiments such as ensembling and clustering, at least four independently trained models are required.

```bash
python src/training.py --dataset cifar10 --model resnet20 --seed 12099
python src/training.py --dataset cifar10 --model resnet20 --seed 12911
python src/training.py --dataset cifar10 --model resnet20 --seed 18144
python src/training.py --dataset cifar10 --model resnet20 --seed 18573
python src/training.py --dataset cifar10 --model resnet20 --seed 27452
```

**Outputs:**

* Cached test dataset files are stored under:

  ```
  artifacts/datasets/cifar10/
  └──testdataset_seed_<seed>.pkl
  ```
* Trained model checkpoints are saved under:

  ```
  artifacts/models/cifar10/resnet20/
  └──best_model_<seed>.pth
  ```
* Training logs (containing epoch-wise loss, accuracy, and validation metrics) are stored under:

  ```
  artifacts/logs/cifar10/resnet20/
  └──log_seed_<seed>.log
  ```

### Model Evaluation

This step evaluates each trained models and computes statistical performance metrics aggregated over the provided seeds.

```bash
python src/compute_accuracy.py --dataset cifar10 --model resnet20 --seed_list 12099 12911 18144 18573 27452
```

**Outputs:**

* Evaluation results and accuracy statistics are stored under:

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet20/
  └──accuracy_summary.log
  ```

## 3. Polynomial Activation Model Training

### Model Training

The framework supports training ResNet architectures with polynomial activation functions as a replacement for ReLU. The example below demonstrates independent training using five random seeds, which can be run in parallel across multiple GPUs or compute nodes  for faster processing.

```bash
python src/training.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 12099
python src/training.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 12911
python src/training.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 18144
python src/training.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 18573
python src/training.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 27452
```

**Outputs:**

* Cached test dataset files are stored under:

  ```
  artifacts/datasets/cifar10/
  └──testdataset_seed_<seed>.pkl
  ```
* Trained model checkpoints are saved under:

  ```
  artifacts/models/cifar10/resnet20_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_<seed>.pth
  ```
  > **Note:** `<sigma>` refers to the noise parameter defined in `arguments.py`. It can be used to introduce controlled noise during training, which may improve generalization and enhance accuracy in certain cases.
* Training logs (containing epoch-wise loss, accuracy, and validation metrics) are stored under:

  ```
  artifacts/logs/cifar10/resnet20_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──log_seed_<seed>.log
  ```

### Model Evaluation

This step evaluates each trained model and computes statistical performance metrics aggregated over the provided seeds.

```bash
python src/compute_accuracy.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452
```

**Outputs:**

* Evaluation results and accuracy statistics are stored under:

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet20_poly/
  └──accuracy_summary_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>.log
  ```

## 4. Fuse Batch Normalization and Convolution Layers
This step fuses Batch Normalization layers with Convolutional layers to simplify the model for inference. The fusion is applied after training is completed.

```bash
python src/fuse_models.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452
```

**Outputs:**

* The fused models are saved under:

  ```
  artifacts/models/fused/cifar10/resnet20_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_<seed>_fused.pth
  ```
* A fusion summary log is stored under:

  ```
  artifacts/results_logs/fusing_logs/cifar10_resnet20_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>.log


## 5. Ensemble Inference

This step evaluates ensembles created by combining predictions from multiple independently trained models using polynomial activations. The example below demonstrates ensemble evaluation with 2 and 4 models, which can be run in parallel across multiple GPUs or compute nodes to accelerate inference.

```bash
python src/ensemble_inference.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --num_models 2

python src/ensemble_inference.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --num_models 4
```

**Outputs:**

* Ensemble inference results and averaged accuracy metrics are stored under:

  ```
  artifacts/results_logs/ensemble_results/cifar10_resnet20_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_num_models_<num_models>.log
  ```

## 6. Convolutional Parameter Clustering
This stage applies *K-means clustering* to convolutional layer parameters to reduce model size by grouping similar weights into representative clusters. Two clustering modes are supported: **full-parameter clustering** and **slice-wise clustering**. 

### 6.1 Full-Parameter Clustering

In this mode, clustering is applied globally across all convolutional layer weights within each trained model. The example below demonstrates full-parameter clustering executed across five independently trained models and a range of cluster sizes (`k = 2` to `k = 1024`).

```bash
for seed in 12099 12911 18144 18573 27452; do
  for k in 2 4 8 16 32 64 128 256 512 1024; do
    python src/conv_cluster.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --k $k --seed $seed
  done
done
```

Each (`seed`, `k`) combination represents an independent clustering experiment that can be executed in parallel across multiple GPUs or compute nodes for faster experimentation.

**Outputs:**

* Clustered model checkpoints are stored under:

  ```
  artifacts/models/cifar10/resnet20_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_seed_<seed>_fused_full_k_<k>_cseed_<cseed>.pth
  ```
  > **Note:** `<cseed>` refers to the clustering seed used for K-means centroid initialization. The framework uses a set of pre-defined seeds in `utils.py`, which can be modified to explore different clustering behaviors.

* Clustering logs (including per-layer statistics, inertia, and quantization distortion) are saved under:

  ```
  artifacts/results_logs/cluster_logs/cifar10_resnet20_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_seed_<seed>_full_k_<k>.log
  ```

After clustering, the accuracy of each compressed model is evaluated, and the results are aggregated across all specified model seeds and cluster sizes.

```bash
python src/compute_cluster_accuracy.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --k_list 2 4 8 16 32 64 128 256 512 1024
```

**Outputs:**

* Evaluation summaries and accuracy statistics are stored under:

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet20_poly/
  └──accuracy_cluster_summary_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_mode_full.log
  ```


### 6.2 Slice-wise Clustering

In this mode, clustering is applied independently across kernel-width slices within each convolutional layer rather than globally over the entire weight tensor. The example below demonstrates slice-wise clustering executed across five independently trained models and a range of cluster sizes `k = 2` to `k = 1024`.

```bash
for seed in 12099 12911 18144 18573 27452; do
  for k in 2 4 8 16 32 64 128 256 512 1024; do
    python src/conv_cluster.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --k $k --seed $seed --slice
  done
done
```

Each `(seed, k)` combination represents an independent clustering experiment that can be executed in parallel across multiple GPUs or compute nodes to accelerate experimentation.

**Outputs:**

* Clustered model checkpoints are stored under

  ```
  artifacts/models/cifar10/resnet20_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_seed_<seed>_fused_slice_k_<k>_cseed_<cseed>.pth
  ```

* Clustering logs including per-layer statistics, inertia, and quantization distortion are saved under

  ```
  artifacts/results_logs/cluster_logs/cifar10_resnet20_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_seed_<seed>_slice_k_<k>.log
  ```

After clustering, the accuracy of each compressed model is evaluated, and the results are aggregated across all specified model seeds and cluster sizes.

```bash
python src/compute_cluster_accuracy.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --k_list 2 4 8 16 32 64 128 256 512 1024 --slice
```

**Outputs:**

* Evaluation summaries and accuracy statistics are stored under

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet20_poly/
  └──accuracy_cluster_summary_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_mode_slice.log
  ```


## 7. Ensemble Model Clustering

### Multi-Model Clustering

This stage performs joint clustering of convolutional parameters across an ensemble of polynomial networks using *multi-dimensional K-means*. The example below demonstrates clustering for two ensemble configurations with 2 and 4 models over a range of cluster sizes (`k = 64` to `k = 8192`). Each experiment corresponding to a specific cluster size and ensemble configuration can be executed in parallel across multiple GPUs or compute nodes to accelerate clustering and evaluation.

```bash
for k in 64 128 256 512 1024 2048 4096 8192; do
  python src/ensemble_cluster.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --num_models 2 --k $k
done

for k in 64 128 256 512 1024 2048 4096 8192; do
  python src/ensemble_cluster.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --num_models 4 --k $k
done
```

**Outputs**

* Clustered ensemble checkpoints are stored under:

  ```
  artifacts/models/cifar10/resnet20_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_<seed>_fused_k_<k>_cseed_<cseed>_ens_<num_models>.pth
  ```

* Clustering logs are stored under:

  ```
  artifacts/results_logs/cluster_logs/cifar10_resnet20_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_slice_k_<k>_ens_<num_models>.log
  ```

### Evaluation

This step evaluates each ensemble configuration after clustering. Results are aggregated across all specified ensemble sizes and cluster counts.

```bash
for k in 64 128 256 512 1024 2048 4096 8192; do
  python src/evaluate_multi_cluster.py --dataset cifar10 --model resnet20_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --k $k --num_models_list 2 4
done
```

**Outputs**

* Evaluation summaries and ensemble accuracy statistics are stored under:

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet20_poly/
  └──accuracy_ensemble_cluster_summary_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_k_<k>.log
  ```

## 8. Comprehensive Results Summary

This step generates a structured summary combining all experiment phases:

```bash
python src/summary.py --dataset cifar10 --model resnet20
```

**Outputs**

* Evaluation summaries are stored under:

  ```
  artifacts/
    └──cifar10_resnet20_summary.log
  ```

# 📁 Process Files for C++ Inference

## 1. Best Model Extraction

This step parses accuracy and ensemble logs to identify and index the best-performing model checkpoints from previous experiments. It compiles structured JSON files containing paths and configuration details for each best model, enabling easy reference for downstream analysis.

```bash
python src/extract_best_models.py --dataset cifar10 --model resnet20
```

**Prerequisites**

* Accuracy and ensemble log files generated from previous stages and located in:
  * `artifacts/results_logs/accuracy_results/`
  * `artifacts/results_logs/ensemble_results/`

**Outputs**

* The generated JSON files are stored under:

  ```
  artifacts/model_jsons/cifar10_resnet20/
  ├── standard_accuracy.json
  ├── full_clustering_after_polynomial_training.json
  ├── slice_clustering_after_polynomial_training.json
  ├── standard_ensemble.json
  └── ensemble_ed_clustering.json
  ```

## 2. Generate JSON Templates for C++ Integration

This step launches two subprocesses to produce deployable JSON templates. It reads best model mappings and calls `create_model_template.py` for each checkpoint. It also scans cached dataset files and calls `create_dataset_template.py` to export dataset samples.

```bash
python src/create_all_jsons.py --dataset cifar10 --model resnet20 --workers 10
```

> **Note:** The script builds command lists then runs them in parallel using the `--workers` limit for faster generation.

**Prerequisites**

* Best model JSONs exist under

  ```
  artifacts/model_jsons/cifar10_resnet20/
  └── *.json
  ```

* Cached dataset files exist under

  ```
  artifacts/datasets/cifar10/
  └── *.pkl
  ```

**Outputs**

* Model templates are written under

  ```
  artifacts/json_models/cifar10_resnet20/<source_json_name>/
  └── <per_checkpoint_jsons>.json
  ```

* Dataset templates are written under

  ```
  artifacts/dataset_jsons/cifar10/
  └── <cached_file_basename>.json
  ```


