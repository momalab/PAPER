# 📑 Overview
This repository provides a complete experimental framework for training, evaluating, and compressing deep convolutional neural networks that use polynomial activation functions. It supports ResNet and VGG architectures on CIFAR and Tiny-ImageNet datasets, with end-to-end tools for polynomial coefficient generation, regularized training, model fusion, ensembling, parameter clustering, and deployment-ready JSON export for inference.

# 🖥️ System Requirements

* **GPU:** NVIDIA GPU with CUDA support (Experiments were conducted on NVIDIA A100 GPUs).
* **Python Version:** Python 3.12.7 (Confirmed compatibility).
* **CUDA Toolkit:** CUDA compilation tools (Tested with release 12.8, V12.8.93).

# 🛠️ Setup and Installation

### Create and Activate a Virtual Environment

```bash
python -m venv paper
source paper/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_pytorch.txt
```
> **Note:** Ensure your CUDA toolkit and NVIDIA drivers are properly installed and compatible with the PyTorch version specified in requirements_pytorch.txt.

# 📦 Dataset Preparation
This project supports three image classification benchmarks: `CIFAR10`, `CIFAR100`, and `Tiny-ImageNet`.

- When `cifar10` or `cifar100` is specified via the `--dataset` argument, the dataset is automatically downloaded using the PyTorch dataset utilities and cached locally in the `./data/` directory.
- Download the Tiny-ImageNet (64x64) dataset from the official ImageNet website: https://www.image-net.org/. After downloading, extract it into the `./data/` directory with the following structure:
  ```
  ./data/tiny-imagenet-200/
  ├── train/
  │   ├── n01443537/
  │   ├── n01629819/
  │   ├── n01641577/
  │   └── ...
  └── val/
      ├── n01443537/
      ├── n01629819/
      ├── n01641577/
      └── ...
  ```

# 🚀 Experiment Workflow

## 1️⃣ Polynomial Coefficient Generation

This step computes fixed-point polynomial activation coefficients that approximate the ReLU function using quantization-aware regression.

```bash
python src/polyfit.py --degree 2 --clip 2 --pbit 10
```

Details of all command line arguments are documented in `arguments.py`.

**Outputs:**

* The generated coefficients are stored under:

  ```
  artifacts/poly_coeffs/
  └──deg_2_clip_2_pbit_10_coeffs.txt
  ```

## 2️⃣ Baseline Model Training

Baseline models use standard ReLU activations and serve as reference points for accuracy and stability comparisons.

### 2.1 Training

The framework supports ResNet (e.g., Resnet-18, Resnet-20, Resnet-32) and VGG16 on CIFAR-10, CIFAR-100, and Tiny-ImageNet. This example demonstrates training ResNet-18 on CIFAR-10 using five random seeds. 

> **Note:** For subsequent experiments such as ensembling and clustering, at least four independently trained models are required.

```bash
python src/training.py --dataset cifar10 --model resnet18 --seed 12099
python src/training.py --dataset cifar10 --model resnet18 --seed 12911
python src/training.py --dataset cifar10 --model resnet18 --seed 18144
python src/training.py --dataset cifar10 --model resnet18 --seed 18573
python src/training.py --dataset cifar10 --model resnet18 --seed 27452
```

**Outputs:**

* Cached test dataset files are stored under:

  ```
  artifacts/datasets/cifar10/
  └──testdataset_seed_<seed>.pkl
  ```
* Trained model checkpoints are saved under:

  ```
  artifacts/models/cifar10/resnet18/
  └──best_model_<seed>.pth
  ```
* Training logs (containing epoch-wise loss, accuracy, and validation metrics) are stored under:

  ```
  artifacts/logs/cifar10/resnet18/
  └──log_seed_<seed>.log
  ```

### 2.2 Evaluation

This step evaluates trained baseline models and computes statistical performance metrics aggregated over the specified random seeds.

```bash
python src/compute_accuracy.py --dataset cifar10 --model resnet18 --seed_list 12099 12911 18144 18573 27452
```

**Outputs:**

* Evaluation results and accuracy statistics are stored under:

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet18/
  └──accuracy_summary.log
  ```

## 3️⃣ Polynomial Activation Model Training

This stage trains networks with polynomial activation functions as a replacement for ReLU using regularized objectives.

### 3.1 Training

The example below demonstrates independent training using five random seeds.

```bash
python src/training.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 12099
python src/training.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 12911
python src/training.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 18144
python src/training.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 18573
python src/training.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed 27452
```

**Outputs:**

* Cached test dataset files are stored under:

  ```
  artifacts/datasets/cifar10/
  └──testdataset_seed_<seed>.pkl
  ```
* Trained model checkpoints are saved under:

  ```
  artifacts/models/cifar10/resnet18_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_<seed>.pth
  ```
  > **Note:** `<sigma>` refers to the noise parameter defined in `arguments.py`. It can be used to introduce controlled noise during training, which may improve generalization and enhance accuracy in certain cases.
* Training logs (containing epoch-wise loss, accuracy, and validation metrics) are stored under:

  ```
  artifacts/logs/cifar10/resnet18_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──log_seed_<seed>.log
  ```

### 3.2 Evaluation

This step evaluates trained model and computes statistical performance metrics aggregated over the specified random seeds.

```bash
python src/compute_accuracy.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452
```

**Outputs:**

* Evaluation results and accuracy statistics are stored under:

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet18_poly/
  └──accuracy_summary_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>.log
  ```

## 4️⃣ Fuse Batch Normalization and Convolution Layers
This step fuses Batch Normalization layers into adjacent Convolutional layers to simplify the model for inference. The fusion is applied after training is completed.

```bash
python src/fuse_models.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452
```

**Outputs:**

* The fused models are saved under:

  ```
  artifacts/models/fused/cifar10/resnet18_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_<seed>_fused.pth
  ```
* A fusion summary log is stored under:

  ```
  artifacts/results_logs/fusing_logs/cifar10_resnet18_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>.log
  ```

## 5️⃣ Ensemble Inference

This step evaluates ensembles formed by combining predictions from multiple independently trained models using polynomial activations. The example below demonstrates ensemble evaluation with 2 and 4 models.

```bash
python src/ensemble_inference.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --num_models 2

python src/ensemble_inference.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --num_models 4
```

**Outputs:**

* Ensemble inference results and averaged accuracy metrics are stored under:

  ```
  artifacts/results_logs/ensemble_results/cifar10_resnet18_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_num_models_<num_models>.log
  ```

## 6️⃣ Convolutional Parameter Clustering
This stage applies *K-means clustering* to convolutional layer parameters to reduce model size by grouping similar weights into representative clusters. Two clustering modes are supported: **full-parameter clustering** and **slice-wise clustering**. 

### 6.1 Full-Parameter Clustering

#### 6.1.1 Clustering

In this mode, clustering is applied globally across all convolutional layer weights within each trained model. The example below demonstrates full-parameter clustering executed across five independently trained models and a range of cluster sizes (`k = 2` to `k = 1024`).

```bash
for seed in 12099 12911 18144 18573 27452; do
  for k in 2 4 8 16 32 64 128 256 512 1024; do
    python src/conv_cluster.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --k $k --seed $seed
  done
done
```

**Outputs:**

* Clustered model checkpoints are stored under:

  ```
  artifacts/models/cifar10/resnet18_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_seed_<seed>_fused_full_k_<k>_cseed_<cseed>.pth
  ```
  > **Note:** `<cseed>` refers to the clustering seed used for K-means centroid initialization. The framework uses a set of pre-defined seeds in `utils.py`, which can be modified to explore different clustering behaviors.

* Clustering logs (including per-layer statistics, inertia, and quantization distortion) are saved under:

  ```
  artifacts/results_logs/cluster_logs/cifar10_resnet18_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_seed_<seed>_full_k_<k>.log
  ```

#### 6.1.2 Evaluation

After clustering, the accuracy of each compressed model is evaluated, and the results are aggregated across all specified model seeds and cluster sizes.

```bash
python src/compute_cluster_accuracy.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --k_list 2 4 8 16 32 64 128 256 512 1024
```

**Outputs:**

* Evaluation summaries and accuracy statistics are stored under:

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet18_poly/
  └──accuracy_cluster_summary_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_mode_full.log
  ```

### 6.2 Slice-wise Clustering

#### 6.2.1 Clustering

In this mode, clustering is applied independently across kernel-width slices within each convolutional layer rather than globally over the entire weight tensor. The example below demonstrates slice-wise clustering executed across five independently trained models and a range of cluster sizes `k = 2` to `k = 1024`.

```bash
for seed in 12099 12911 18144 18573 27452; do
  for k in 2 4 8 16 32 64 128 256 512 1024; do
    python src/conv_cluster.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --k $k --seed $seed --slice
  done
done
```

**Outputs:**

* Clustered model checkpoints are stored under

  ```
  artifacts/models/cifar10/resnet18_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_seed_<seed>_fused_slice_k_<k>_cseed_<cseed>.pth
  ```

* Clustering logs including per-layer statistics, inertia, and quantization distortion are saved under

  ```
  artifacts/results_logs/cluster_logs/cifar10_resnet18_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_seed_<seed>_slice_k_<k>.log
  ```

#### 6.2.2 Evaluation

After clustering, the accuracy of each compressed model is evaluated, and the results are aggregated across all specified model seeds and cluster sizes.

```bash
python src/compute_cluster_accuracy.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --k_list 2 4 8 16 32 64 128 256 512 1024 --slice
```

**Outputs:**

* Evaluation summaries and accuracy statistics are stored under

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet18_poly/
  └──accuracy_cluster_summary_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_mode_slice.log
  ```


## 7️⃣ Ensemble Model Clustering

This stage performs joint clustering of convolutional parameters across an ensemble of polynomial networks using *multi-dimensional K-means*. 

### 7.1 Clustering

The example below demonstrates clustering for two ensemble configurations with 2 and 4 models over a range of cluster sizes (`k = 64` to `k = 8192`).

```bash
for k in 64 128 256 512 1024 2048 4096 8192; do
  python src/ensemble_cluster.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --num_models 2 --k $k
done

for k in 64 128 256 512 1024 2048 4096 8192; do
  python src/ensemble_cluster.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --seed_list 12099 12911 18144 18573 27452 --num_models 4 --k $k
done
```

**Outputs**

* Clustered ensemble checkpoints are stored under:

  ```
  artifacts/models/cifar10/resnet18_poly/degree_2/clip_2/pbit_10/zeta_0.001/sigma_<sigma>/
  └──best_model_<seed>_fused_k_<k>_cseed_<cseed>_ens_<num_models>.pth
  ```

* Clustering logs are stored under:

  ```
  artifacts/results_logs/cluster_logs/cifar10_resnet18_poly/
  └──logs_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_slice_k_<k>_ens_<num_models>.log
  ```

### 7.2 Evaluation

This step evaluates each ensemble configuration after clustering. Results are aggregated across all specified ensemble sizes and cluster counts.

```bash
for k in 64 128 256 512 1024 2048 4096 8192; do
  python src/evaluate_multi_cluster.py --dataset cifar10 --model resnet18_poly --degree 2 --clip 2 --pbit 10 --penalty 0.001 --k $k --num_models_list 2 4
done
```

**Outputs**

* Evaluation summaries and ensemble accuracy statistics are stored under:

  ```
  artifacts/results_logs/accuracy_results/cifar10_resnet18_poly/
  └──accuracy_ensemble_cluster_summary_degree_2_clip_2_pbit_10_zeta_0.001_sigma_<sigma>_k_<k>.log
  ```

## 8️⃣ Comprehensive Results Summary

This step generates a structured summary combining all experiment phases:

```bash
python src/summary.py --dataset cifar10 --model resnet18
```

**Outputs**

* Evaluation summaries are stored under:

  ```
  artifacts/
    └──cifar10_resnet18_summary.log
  ```

# 📁 Process Files for C++ Inference

## 1️⃣ Best Model Extraction

This step parses accuracy and ensemble logs to identify and index the best-performing model checkpoints from previous experiments. It compiles structured JSON files containing paths and configuration details for each best model, enabling easy reference for downstream analysis.

```bash
python src/extract_best_models.py --dataset cifar10 --model resnet18
```

**Prerequisites**

* Accuracy and ensemble log files generated from previous stages and located in:
  * `artifacts/results_logs/accuracy_results/`
  * `artifacts/results_logs/ensemble_results/`

**Outputs**

* The generated JSON files are stored under:

  ```
  artifacts/model_jsons/cifar10_resnet18/
  ├──standard_accuracy.json
  ├──full_clustering_after_polynomial_training.json
  ├──slice_clustering_after_polynomial_training.json
  ├──standard_ensemble.json
  └──ensemble_ed_clustering.json
  ```

## 2️⃣ Generate JSON Templates for C++ Integration

This step launches two subprocesses to produce deployable JSON templates. It reads best model mappings and calls `create_model_template.py` for each checkpoint. It also scans cached dataset files and calls `create_dataset_template.py` to export dataset samples.

```bash
python src/create_all_jsons.py --dataset cifar10 --model resnet18 --workers 10
```

> **Note:** The script builds command lists then runs them in parallel using the `--workers` limit for faster generation.

**Prerequisites**

* Best model JSONs exist under

  ```
  artifacts/model_jsons/cifar10_resnet18/
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
  artifacts/model_jsons/cifar10_resnet18/<source_json_name>/
  └──<per_checkpoint_jsons>.json
  ```

* Dataset templates are written under

  ```
  artifacts/dataset_jsons/cifar10/
  └──<cached_file_basename>.json
  ```

## 3️⃣ Organize JSONs for C++ Inference
This step consolidates all generated model JSON templates and dataset JSON templates into a unified directory structure expected by the C++ inference runtime.

```bash 
bash src/organize_jsons_for_inference.sh cifar10 resnet18
```

**Prerequisites**

* Dataset JSON templates and model JSON templates exist under:

  * ```artifacts/dataset_jsons/cifar10/```
  * ```artifacts/model_jsons/cifar10_resnet18/```

**Outputs**

  * After successful execution, all JSON files required for C++ inference are organized into the following directory structure located outside the training directory:

    ```
    ../resources/
    └── json_files/
        ├── datasets/
        │   └── cifar10/
        │       └── <dataset_sample_jsons>.json
        └── models/
            └── cifar10_resnet18/
                ├── standard/
                │   └── <standard_and_ensemble_jsons>.json
                ├── full_clustering/
                │   └── <full_clustered_model_jsons>.json
                └── slice_clustering/
                    └── <slice_and_ensemble_clustered_jsons>.json
    ```