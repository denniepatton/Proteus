#!/bin/bash
# Run inference with all 3 finalized PLCA models on every configured dataset.
# Runs sequentially on a single locally-grabbed GPU.
#
# Usage (from the Proteus/ workspace root, with env active):
#   bash runs/launch_plca_inference.sh
#
# Prerequisites:
#   - Zarr stores must already exist for each dataset (run --process_data if needed).
#   - Outputs go to results_holdout/<model_name>/<dataset>/

set -euo pipefail

eval "$(micromamba shell hook -s bash)"
micromamba activate proteus

DATASETS=(
    BC-Ghana
    BLCA-RA
    CRPC-Pfizer
    CRPC-Pluvicto
    CRPC-UW
    CRPC-WCDT
    Crompton
    PC-PDX
    PHS
    SCLC-PDX
)

declare -A MODEL_EXPERIMENTS=(
    ["BioDVN"]="config/experiments/plca_final-training.yaml"
    ["BioDVN_NoSeq"]="config/experiments/plca-no-seq_final-training.yaml"
    ["BioDVN_OnlySeq"]="config/experiments/plca-only-seq_final-training.yaml"
)

for MODEL in BioDVN BioDVN_NoSeq BioDVN_OnlySeq; do
    EXPERIMENT="${MODEL_EXPERIMENTS[$MODEL]}"
    echo "============================================================"
    echo "  Model: $MODEL  (experiment: $EXPERIMENT)"
    echo "============================================================"
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "  >> $MODEL on $DATASET"
        python Proteus.py \
            --run_mode run \
            --experiment "$EXPERIMENT" \
            --model_name "$MODEL" \
            --use_set "$DATASET"
    done
done

echo ""
echo "All inference runs complete."
