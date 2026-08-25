#!/usr/bin/env bash
# Start online WNM-3D InteriorGS navigation WebSocket services.
#
# Usage:
#   bash scripts/inference/wnm_3d_server.sh \
#       --model-path ./checkpoints/wnm_3d_stage3_release
#
# By default this starts 8 independent one-GPU servers:
#   GPU 0 -> port 8000
#   GPU 1 -> port 8001
#   ...
#   GPU 7 -> port 8007
#
# Client messages should contain one RGB frame using one of:
#   images, video.rgb, image, rgb, observation/image, observation/rgb
# plus optional prompt/instruction/language, state.nav_pose, and session_id.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

MODEL_PATH=""
CUDA_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_REPLICA="1"
NUM_REPLICAS="8"
HOST="127.0.0.1"
BASE_PORT="8000"
MAX_MESSAGE_SIZE_BYTES="67108864"
OUTPUT_DIR=""
NAV_ACTION_SCALE=""
ENABLE_DIT_CACHE="true"
NUM_INFERENCE_STEPS="16"
DIT_STEP_MASK=(1 1 1 0 0 0 1 0 0 0 1 0 0 1 1 1)
ENABLE_CFG="true"
CFG_SCALE="5.0"
DISABLE_TORCH_COMPILE="true"
RETURN_SCALED_ACTION="false"
RESIZE_INPUT_TO_CHECKPOINT_RESOLUTION="false"
HISTORY_SAMPLING="uniform"
HISTORY_LONG_RANGE_ANCHORS="8"
SAVE_INPUT_CLIPS="false"
SAVE_GENERATED_VIDEO="false"
MAX_CHUNK_SIZE=""
PROFILE_MODULE_TIMINGS="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --gpus-per-replica)
            GPUS_PER_REPLICA="$2"
            shift 2
            ;;
        --num-replicas)
            NUM_REPLICAS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        --max-message-size-bytes)
            MAX_MESSAGE_SIZE_BYTES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --nav-action-scale)
            NAV_ACTION_SCALE="$2"
            shift 2
            ;;
        --enable-dit-cache)
            ENABLE_DIT_CACHE="true"
            shift
            ;;
        --disable-dit-cache)
            ENABLE_DIT_CACHE="false"
            shift
            ;;
        --num-inference-steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --enable-cfg)
            ENABLE_CFG="true"
            shift
            ;;
        --disable-cfg)
            ENABLE_CFG="false"
            shift
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --disable-torch-compile)
            DISABLE_TORCH_COMPILE="true"
            shift
            ;;
        --enable-torch-compile)
            DISABLE_TORCH_COMPILE="false"
            shift
            ;;
        --return-scaled-action)
            RETURN_SCALED_ACTION="true"
            shift
            ;;
        --resize-input-to-checkpoint-resolution)
            RESIZE_INPUT_TO_CHECKPOINT_RESOLUTION="true"
            shift
            ;;
        --history-sampling)
            HISTORY_SAMPLING="$2"
            shift 2
            ;;
        --history-long-range-anchors)
            HISTORY_LONG_RANGE_ANCHORS="$2"
            shift 2
            ;;
        --save-input-clips)
            SAVE_INPUT_CLIPS="true"
            shift
            ;;
        --save-generated-video)
            SAVE_GENERATED_VIDEO="true"
            shift
            ;;
        --max-chunk-size)
            MAX_CHUNK_SIZE="$2"
            shift 2
            ;;
        --profile-module-timings)
            PROFILE_MODULE_TIMINGS="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --model-path <checkpoint> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-replicas N                   Independent policy replicas, default 8"
            echo "  --cuda-devices IDS                 Device pool, default 0,1,2,3,4,5,6,7"
            echo "  --gpus-per-replica N               GPUs per policy replica, default 1"
            echo "  --host HOST                        Bind host, default 127.0.0.1"
            echo "  --base-port PORT                   Base WebSocket port, default 8000"
            echo "  --max-message-size-bytes BYTES     Maximum request size, default 67108864"
            echo "  --output-dir PATH                  Directory for optional saved inputs/generated videos"
            echo "  --nav-action-scale X               Override checkpoint scale; missing config defaults to 1.0"
            echo "  --enable-dit-cache                 Reuse selected DiT predictions, default"
            echo "  --disable-dit-cache                Run DiT at every scheduler step"
            echo "  --num-inference-steps N            UniPC scheduler updates, default 16"
            echo "  --enable-cfg                       Enable positive/negative CFG, default"
            echo "  --disable-cfg                      Disable CFG"
            echo "  --cfg-scale X                      CFG guidance scale, default 5.0"
            echo "  --disable-torch-compile            Skip torch.compile for inference stability, default"
            echo "  --enable-torch-compile             Re-enable torch.compile"
            echo "  --return-scaled-action             Make response['action'] use training-scale units"
            echo "  --resize-input-to-checkpoint-resolution"
            echo "                                      Resize incoming frames to checkpoint resolution"
            echo "  --history-sampling MODE            recent, mixed, or uniform; default uniform"
            echo "  --history-long-range-anchors N     Mixed-mode long-range anchors, default 8"
            echo "  --save-input-clips                 Save converted 66-frame input clips, default off"
            echo "  --save-generated-video             Decode and save generated latent videos on reset/disconnect, default off"
            echo "  --max-chunk-size N                 Optional action-head max_chunk_size override"
            echo "  --profile-module-timings           Log preprocess/text/CLIP/VAE/VGGT/DiT timings"
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

if ! [[ "$NUM_REPLICAS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$GPUS_PER_REPLICA" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-replicas and --gpus-per-replica must be positive integers." >&2
    exit 1
fi
if ! [[ "$BASE_PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: --base-port must be an integer." >&2
    exit 1
fi
if ! [[ "$MAX_MESSAGE_SIZE_BYTES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max-message-size-bytes must be a positive integer." >&2
    exit 1
fi
if ! [[ "$NUM_INFERENCE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-inference-steps must be a positive integer." >&2
    exit 1
fi
if [[ "$NUM_INFERENCE_STEPS" -gt "${#DIT_STEP_MASK[@]}" ]]; then
    echo "Error: --num-inference-steps cannot exceed the fixed ${#DIT_STEP_MASK[@]}-step DiT mask." >&2
    exit 1
fi
MASK_DIT_STEPS=0
for ((mask_idx=0; mask_idx<NUM_INFERENCE_STEPS; mask_idx++)); do
    MASK_DIT_STEPS=$(( MASK_DIT_STEPS + DIT_STEP_MASK[mask_idx] ))
done

if [[ -n "${BASH_SOURCE:-}" ]]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
else
    SCRIPT_PATH="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_DEVICES="${CUDA_DEVICES// /}"
IFS="," read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_DEVICES"

REQUIRED_DEVICES=$(( NUM_REPLICAS * GPUS_PER_REPLICA ))
if [[ "${#CUDA_DEVICE_ARRAY[@]}" -lt "$REQUIRED_DEVICES" ]]; then
    echo "Error: --num-replicas $NUM_REPLICAS with --gpus-per-replica $GPUS_PER_REPLICA needs $REQUIRED_DEVICES CUDA devices, got: $CUDA_DEVICES" >&2
    exit 1
fi

echo "=========================================="
echo "WNM-3D InteriorGS Navigation Servers"
echo "  Checkpoint     : $MODEL_PATH"
echo "  Replicas       : $NUM_REPLICAS"
echo "  CUDA pool      : $CUDA_DEVICES"
echo "  GPUs/replica   : $GPUS_PER_REPLICA"
if [[ "$SAVE_INPUT_CLIPS" == "true" || "$SAVE_GENERATED_VIDEO" == "true" ]]; then
    OUTPUT_DISPLAY="${OUTPUT_DIR:-<checkpoint parent>/interiorgs_nav_online_*}"
else
    OUTPUT_DISPLAY="<disabled>"
fi
echo "  Host/base port : $HOST:$BASE_PORT"
echo "  Max message    : $MAX_MESSAGE_SIZE_BYTES bytes"
echo "  Layout         : past 3D obs | noisy video | noisy action | state"
echo "  History mode   : $HISTORY_SAMPLING"
if [[ "$ENABLE_DIT_CACHE" == "true" ]]; then
    echo "  Sampling       : $MASK_DIT_STEPS/$NUM_INFERENCE_STEPS"
else
    echo "  Sampling       : $NUM_INFERENCE_STEPS/$NUM_INFERENCE_STEPS"
fi
echo "  DiT cache      : $ENABLE_DIT_CACHE"
if [[ "$ENABLE_CFG" == "true" ]]; then
    echo "  CFG            : on (scale $CFG_SCALE)"
else
    echo "  CFG            : off"
fi
echo "  Torch compile  : $([[ "$DISABLE_TORCH_COMPILE" == "true" ]] && echo disabled || echo enabled)"
echo "  Module profile : $PROFILE_MODULE_TIMINGS"
if [[ "$HISTORY_SAMPLING" == "mixed" ]]; then
    echo "  Long anchors   : $HISTORY_LONG_RANGE_ANCHORS"
fi
echo "  Save clips     : $SAVE_INPUT_CLIPS"
echo "  Save gen video : $SAVE_GENERATED_VIDEO"
echo "  Output         : $OUTPUT_DISPLAY"
echo "=========================================="

export HYDRA_FULL_ERROR=1
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}

SERVER_PIDS=()

RUNTIME_DIR="${WNM3D_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp}/wnm-3d-${UID}}"
mkdir -p "$RUNTIME_DIR"
chmod 700 "$RUNTIME_DIR"
PID_FILE="${RUNTIME_DIR}/server-$$.pid"
PROCESS_START_ID="$(
    ps -p "$$" -o lstart= 2>/dev/null \
        | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
)"
(
    umask 077
    printf 'pid=%s\nstart=%s\nrepo_root=%s\n' \
        "$$" "$PROCESS_START_ID" "$REPO_ROOT" > "$PID_FILE"
)

remove_pid_file() {
    rm -f -- "$PID_FILE"
}

cleanup() {
    trap - SIGINT SIGTERM
    echo "Terminating InteriorGS WNM-3D navigation servers..."
    local pids
    pids="$(jobs -pr)"
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null || true
        wait $pids 2>/dev/null || true
    fi
    exit 130
}

trap remove_pid_file EXIT
trap cleanup SIGINT SIGTERM

launch_replica() {
    local replica_idx="$1"
    local replica_number="$(( replica_idx + 1 ))"
    local replica_port="$(( BASE_PORT + replica_idx ))"
    local replica_output_dir="$OUTPUT_DIR"
    local offset="$(( replica_idx * GPUS_PER_REPLICA ))"
    local replica_slice=( "${CUDA_DEVICE_ARRAY[@]:$offset:$GPUS_PER_REPLICA}" )
    local replica_devices
    replica_devices="$(IFS=,; echo "${replica_slice[*]}")"

    if [[ "$NUM_REPLICAS" -gt 1 && -n "$OUTPUT_DIR" ]]; then
        replica_output_dir="${OUTPUT_DIR}/replica_${replica_idx}"
    fi

    local python_args=(
        --model-path "$MODEL_PATH"
        --host "$HOST"
        --port "$replica_port"
        --max-message-size-bytes "$MAX_MESSAGE_SIZE_BYTES"
        --index "$replica_idx"
        --num-inference-steps "$NUM_INFERENCE_STEPS"
        --cfg-scale "$CFG_SCALE"
    )

    if [[ -n "$replica_output_dir" ]]; then
        python_args+=(--output-dir "$replica_output_dir")
    fi
    if [[ -n "$NAV_ACTION_SCALE" ]]; then
        python_args+=(--nav-action-scale "$NAV_ACTION_SCALE")
    fi
    if [[ "$ENABLE_DIT_CACHE" == "true" ]]; then
        python_args+=(--enable-dit-cache)
    fi
    if [[ "$ENABLE_CFG" == "true" ]]; then
        python_args+=(--enable-cfg)
    fi
    if [[ "$RETURN_SCALED_ACTION" == "true" ]]; then
        python_args+=(--return-scaled-action)
    fi
    if [[ "$RESIZE_INPUT_TO_CHECKPOINT_RESOLUTION" == "true" ]]; then
        python_args+=(--resize-input-to-checkpoint-resolution)
    fi
    python_args+=(--history-sampling "$HISTORY_SAMPLING")
    python_args+=(--history-long-range-anchors "$HISTORY_LONG_RANGE_ANCHORS")
    if [[ "$SAVE_INPUT_CLIPS" == "true" ]]; then
        python_args+=(--save-input-clips)
    fi
    if [[ "$SAVE_GENERATED_VIDEO" == "true" ]]; then
        python_args+=(--save-generated-video)
    fi
    if [[ -n "$MAX_CHUNK_SIZE" ]]; then
        python_args+=(--max-chunk-size "$MAX_CHUNK_SIZE")
    fi
    if [[ "$PROFILE_MODULE_TIMINGS" == "true" ]]; then
        python_args+=(--profile-module-timings)
    fi

    echo "[Replica $replica_number/$NUM_REPLICAS] CUDA_VISIBLE_DEVICES=$replica_devices port=$replica_port"
    CUDA_VISIBLE_DEVICES="$replica_devices" \
    DISABLE_TORCH_COMPILE="$DISABLE_TORCH_COMPILE" \
    torchrun \
        --standalone \
        --nproc_per_node="$GPUS_PER_REPLICA" \
        "${REPO_ROOT}/scripts/inference/wnm_3d_server.py" \
        "${python_args[@]}"
}

for ((replica_idx=0; replica_idx<NUM_REPLICAS; replica_idx++)); do
    launch_replica "$replica_idx" &
    SERVER_PIDS+=( "$!" )
done

STATUS=0
for pid in "${SERVER_PIDS[@]}"; do
    if ! wait "$pid"; then
        STATUS=1
    fi
done

exit "$STATUS"
