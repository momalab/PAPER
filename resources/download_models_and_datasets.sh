#!/usr/bin/env bash
set -euo pipefail

PROJECT="PAPER_Models_and_Datasets"
VERSION="1.0"
DIRECTORY="reproducibility"

curl -fLO "https://github.com/anonymous-usenix26/PAPER_Models_and_Datasets/archive/refs/tags/${VERSION}.zip"
7z x "${VERSION}.zip"
rm "${VERSION}.zip"

mv "${PROJECT}-${VERSION}" "${DIRECTORY}"
cd "${DIRECTORY}/datasets"

datasets=("cifar10" "cifar100" "tiny")

for dataset in "${datasets[@]}"; do
    if [[ -f "${dataset}.7z.001" ]]; then
        7z x "${dataset}.7z.001"
        rm "${dataset}.7z"*
    elif [[ -f "${dataset}.7z" ]]; then
        7z x "${dataset}.7z"
        rm "${dataset}.7z"
    else
        echo "Warning: no archive found for dataset ${dataset}"
    fi
done

cd ../models

models=(
    "cifar10_vgg16"
    "cifar10_resnet18"
    "cifar10_resnet20"
    "cifar10_resnet32"
    "cifar100_resnet18"
    "cifar100_resnet20"
    "cifar100_resnet32"
    "tiny_resnet32"
)

for model in "${models[@]}"; do
    if [[ -f "${model}.7z.001" ]]; then
        7z x "${model}.7z.001"
        rm "${model}.7z"*
    elif [[ -f "${model}.7z" ]]; then
        7z x "${model}.7z"
        rm "${model}.7z"
    else
        echo "Warning: no archive found for model ${model}"
    fi
done
