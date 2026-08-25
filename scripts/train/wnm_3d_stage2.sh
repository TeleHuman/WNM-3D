#!/usr/bin/env bash
# Stage II: fine-tuning on the InteriorGS DAgger dataset (single node).
#
# By default, this initializes from the latest checkpoint under
# checkpoints/wnm_3d_stage1. Set PRETRAINED_MODEL_PATH to choose an exact
# Stage-I checkpoint directly.
#
# Usage:
#   bash scripts/train/wnm_3d_stage2.sh [Hydra overrides...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WNM3D_ROOT=${WNM3D_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}

find_latest_checkpoint() {
    local checkpoint_root="$1"
    local latest=""
    latest="$(find "$checkpoint_root" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | sort -V \
        | tail -n 1 \
        || true)"
    if [ -n "$latest" ]; then
        printf '%s\n' "$latest"
    elif [ -f "$checkpoint_root/model.safetensors" ] \
        || [ -f "$checkpoint_root/model.safetensors.index.json" ]; then
        printf '%s\n' "$checkpoint_root"
    fi
}

BASE_CKPT_DIR=${BASE_CKPT_DIR:-"$WNM3D_ROOT/checkpoints/wnm_3d_stage1"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"$(find_latest_checkpoint "$BASE_CKPT_DIR")"}
if [ -z "$PRETRAINED_MODEL_PATH" ]; then
    # Keep the intended path visible in DRY_RUN output; base.sh performs the
    # strict safetensors check for a real run.
    PRETRAINED_MODEL_PATH="$BASE_CKPT_DIR"
fi

export WNM3D_ROOT
export WNM3D_TRAIN_STAGE=stage2
export INTERIORGS_DATA_ROOT=${INTERIORGS_DATA_ROOT:-"$WNM3D_ROOT/data/interiorgs_dagger_3d_stage2"}
export BASE_CKPT_DIR
export PRETRAINED_MODEL_PATH
export OUTPUT_DIR=${OUTPUT_DIR:-"$WNM3D_ROOT/checkpoints/wnm_3d_stage2"}
export TB_LOG_DIR=${TB_LOG_DIR:-"$OUTPUT_DIR/tensorboard"}
export NNODES=${NNODES:-1}
export NUM_GPUS=${NUM_GPUS:-8}
export PER_DEVICE_BS=${PER_DEVICE_BS:-12}
export MAX_STEPS=${MAX_STEPS:-70000}
export LEARNING_RATE=${LEARNING_RATE:-1e-5}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}

exec bash "$SCRIPT_DIR/wnm_3d_base.sh" "$@"
