#!/usr/bin/env bash
# Stage I: full fine-tuning on the InteriorGS expert dataset (single node).
#
# Usage:
#   bash scripts/train/wnm_3d_stage1.sh [Hydra overrides...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WNM3D_ROOT=${WNM3D_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}

export WNM3D_ROOT
export WNM3D_TRAIN_STAGE=stage1
export INTERIORGS_DATA_ROOT=${INTERIORGS_DATA_ROOT:-"$WNM3D_ROOT/data/interiorgs_lerobot_seen"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$WNM3D_ROOT/checkpoints/wnm_3d_stage1"}
export TB_LOG_DIR=${TB_LOG_DIR:-"$OUTPUT_DIR/tensorboard"}
export NNODES=${NNODES:-1}
export NUM_GPUS=${NUM_GPUS:-8}
export PER_DEVICE_BS=${PER_DEVICE_BS:-12}
export MAX_STEPS=${MAX_STEPS:-200000}
export LEARNING_RATE=${LEARNING_RATE:-1e-5}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}

exec bash "$SCRIPT_DIR/wnm_3d_base.sh" "$@"
