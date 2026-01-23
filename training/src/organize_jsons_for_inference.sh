#!/usr/bin/env bash

set -e

# =============================================================================
# Input Arguments
# =============================================================================
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <dataset> <model>"
  exit 1
fi

DATASET="$1"
MODEL="$2"
MODEL_NAME="${DATASET}_${MODEL}"

# =============================================================================
# Repository Path Resolution
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# Dataset Artifact Handling
# =============================================================================
DATASET_SRC="$PAPER_DIR/training/artifacts/dataset_jsons/$DATASET"
DATASET_DST="$PAPER_DIR/resources/json_files/datasets"


if [ ! -e "$DATASET_SRC" ]; then
  echo "Error: dataset source does not exist: $DATASET_SRC"
  exit 1
fi

mkdir -p "$DATASET_DST"
cp -r "$DATASET_SRC" "$DATASET_DST"

# =============================================================================
# Model Artifact Handling
# =============================================================================
MODEL_SRC_BASE="$PAPER_DIR/training/artifacts/model_jsons/$MODEL_NAME"
MODEL_DST_BASE="$PAPER_DIR/resources/json_files/models/$MODEL_NAME"

if [ ! -d "$MODEL_SRC_BASE" ]; then
  echo "Error: model source does not exist: $MODEL_SRC_BASE"
  exit 1
fi

mkdir -p \
  "$MODEL_DST_BASE/full_clustering" \
  "$MODEL_DST_BASE/slice_clustering" \
  "$MODEL_DST_BASE/standard"

cp -r "$MODEL_SRC_BASE/full_clustering_after_polynomial_training/"* \
      "$MODEL_DST_BASE/full_clustering/"

cp -r "$MODEL_SRC_BASE/slice_clustering_after_polynomial_training/"* \
      "$MODEL_DST_BASE/slice_clustering/"

cp -r "$MODEL_SRC_BASE/ensemble_ed_clustering/"* \
      "$MODEL_DST_BASE/slice_clustering/"

cp -r "$MODEL_SRC_BASE/standard_accuracy/"* \
      "$MODEL_DST_BASE/standard/"

cp -r "$MODEL_SRC_BASE/standard_ensemble/"* \
      "$MODEL_DST_BASE/standard/"
