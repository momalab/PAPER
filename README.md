# PAPER: Privacy-Preserving Convolutional Neural Networks using Low-Degree Polynomial Approximations and Structural Optimizations on Leveled FHE

# 📌 Overview
This repository contains the reference implementation for _PAPER: Privacy-Preserving Convolutional Neural Networks using Low-Degree Polynomial Approximations and Structural Optimizations on Leveled FHE_. It provides an end-to-end framework for non-interactive, privacy-preserving inference of convolutional neural networks (CNNs) using leveled Fully Homomorphic Encryption (LFHE). The project integrates low-degree polynomial approximations of non-linear activations, specialized training procedures, and architecture-aware structural optimizations to enable efficient evaluation of deep CNNs under LFHE constraints.

# 🗂️ Repository Structure
- `training/`: Model training, evaluation, and clustering
- `inference/`: Privacy-preserving inference using LFHE 
- `resources/`: Training outputs (models and datasets) for inference and reproducibility
- `README.md`: This file. Project overview and top-level documentation.

# 🛠️ Training

See [training/README.md](training/README.md) for environment setup, dataset preparation, experiment workflows, and file preparation for private inference.

# 🔐 Inference

See [inference/README.md](inference/README.md) for dependencies, compilation, and execution of private inference.

# 🔁 Reproduction of Results

See the _Reproduction of Results_ section in [inference/README.md](inference/README.md) for step-by-step instructions.

<!-- # Cite Us -->