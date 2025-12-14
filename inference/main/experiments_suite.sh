#!/bin/bash

# Define arrays
# programs=("experiments_standard_accuracy.py" "experiments_full_clustering.py" "experiments_slice_clustering.py" "experiments_ensemble_edclustering.py" "experiments_standard_ensemble.py")
programs=("experiments_ensemble_edclustering.py" "experiments_standard_ensemble.py")
# programs=("experiments_standard_accuracy.py")
models=("resnet32")
# datasets=("cifar10" "cifar100" "tiny")
datasets=("tiny")

# Nested loops
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for program in "${programs[@]}"; do
            echo "Running: python3 $program $model $dataset $1"
            python3 "$program" "$model" "$dataset" $1
        done
    done
done
