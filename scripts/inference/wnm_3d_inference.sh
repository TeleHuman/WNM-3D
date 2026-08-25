#!/usr/bin/env bash
# Run offline InteriorGS open-loop evaluation from a WNM-3D checkpoint.
#
# Usage:
#   bash scripts/inference/wnm_3d_inference.sh \
#       --model-path ./checkpoints/wnm_3d_stage3_release \
#       --dataset-path ./data/interiorgs_lerobot_seen \
#       --cuda-devices 0 \
#       --num-episodes 1
#
# The script saves:
#   {output-dir}/predictions.jsonl
#   {output-dir}/summary.json

set -euo pipefail

MODEL_PATH=""
DATASET_PATH="./data/interiorgs_lerobot_seen"
OUTPUT_DIR=""
CUDA_DEVICES="0"
NUM_GPUS="1"
NUM_EPISODES="1"
START_EPISODE="0"
EPISODES=""
START_STEP="0"
NAV_ACTION_SCALE=""
SKIP_DATASET_STATISTICS="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --start-episode)
            START_EPISODE="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --start-step)
            START_STEP="$2"
            shift 2
            ;;
        --nav-action-scale)
            NAV_ACTION_SCALE="$2"
            shift 2
            ;;
        --skip-dataset-statistics)
            SKIP_DATASET_STATISTICS="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --model-path <checkpoint> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset-path PATH          InteriorGS LeRobot dataset path"
            echo "  --output-dir PATH            Output directory"
            echo "  --cuda-devices IDS           CUDA_VISIBLE_DEVICES value, default 0"
            echo "  --num-gpus N                 torchrun processes, default 1"
            echo "  --episodes IDS               Comma-separated episode ids"
            echo "  --start-episode N            Dataset episode-list offset, default 0"
            echo "  --num-episodes N             Number of episodes, default 1"
            echo "  --start-step N               First trajectory step, default 0"
            echo "  --nav-action-scale X         Override checkpoint scale; missing config defaults to 1.0"
            echo "  --skip-dataset-statistics    Avoid stats recomputation when constructing raw dataset"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path is required" >&2
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: checkpoint directory not found: $MODEL_PATH" >&2
    exit 1
fi

if [[ ! -d "$DATASET_PATH" ]]; then
    echo "Error: dataset directory not found: $DATASET_PATH" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_ARGS=(
    --model-path "$MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --start-episode "$START_EPISODE"
    --num-episodes "$NUM_EPISODES"
    --start-step "$START_STEP"
)

if [[ -n "$NAV_ACTION_SCALE" ]]; then
    PYTHON_ARGS+=(--nav-action-scale "$NAV_ACTION_SCALE")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    PYTHON_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$EPISODES" ]]; then
    PYTHON_ARGS+=(--episodes "$EPISODES")
fi
if [[ "$SKIP_DATASET_STATISTICS" == "true" ]]; then
    PYTHON_ARGS+=(--skip-dataset-statistics)
fi

echo "=========================================="
echo "WNM-3D InteriorGS Open-Loop Evaluation"
echo "  Checkpoint       : $MODEL_PATH"
echo "  Dataset          : $DATASET_PATH"
echo "  CUDA devices     : $CUDA_DEVICES"
echo "  Num GPUs         : $NUM_GPUS"
echo "  Episodes         : ${EPISODES:-start=$START_EPISODE count=$NUM_EPISODES}"
echo "  Mode             : block1 from start-step"
echo "  Output           : ${OUTPUT_DIR:-${MODEL_PATH}/inference/interiorgs_nav}"
echo "=========================================="

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HYDRA_FULL_ERROR=1
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    "${REPO_ROOT}/scripts/inference/wnm_3d_inference.py" \
    "${PYTHON_ARGS[@]}"
