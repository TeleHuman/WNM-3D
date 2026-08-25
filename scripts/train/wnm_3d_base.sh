#!/usr/bin/env bash
# Shared WNM-3D InteriorGS training launcher.
#
# This is an internal entry point. Use wnm_3d_stage1.sh,
# wnm_3d_stage1_distributed.sh, wnm_3d_stage2.sh, or
# wnm_3d_stage2_distributed.sh instead.

set -euo pipefail

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WNM3D_ROOT=${WNM3D_ROOT:-"$REPO_ROOT"}

if [ ! -d "$WNM3D_ROOT/gammanav" ]; then
    echo "ERROR: No gammanav/ directory under $WNM3D_ROOT."
    echo "Set WNM3D_ROOT to the WNM-3D repository root."
    exit 1
fi

export PYTHONPATH="$WNM3D_ROOT${PYTHONPATH:+:$PYTHONPATH}"

WNM3D_TRAIN_STAGE=${WNM3D_TRAIN_STAGE:-base}
INTERIORGS_DATA_ROOT=${INTERIORGS_DATA_ROOT:-"$WNM3D_ROOT/data/interiorgs_lerobot_seen"}
OUTPUT_DIR=${OUTPUT_DIR:-"$WNM3D_ROOT/checkpoints/wnm_3d_${WNM3D_TRAIN_STAGE}"}
TB_LOG_DIR=${TB_LOG_DIR:-"$OUTPUT_DIR/tensorboard"}

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-12345}
PER_DEVICE_BS=${PER_DEVICE_BS:-12}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * NNODES * PER_DEVICE_BS))}

MAX_STEPS=${MAX_STEPS:-200000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
REPORT_TO=${REPORT_TO:-tensorboard}
WANDB_PROJECT=${WANDB_PROJECT:-wnm_3d}
SEED=${SEED:-42}

DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY:-false}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-true}
if [ "$DATALOADER_NUM_WORKERS" -eq 0 ]; then
    DATALOADER_PERSISTENT_WORKERS=false
fi

VIDEO_FRAME_STRIDE=${VIDEO_FRAME_STRIDE:-1}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}
PYTHON_BIN=${PYTHON_BIN:-python3}
AUTO_DOWNLOAD_WEIGHTS=${AUTO_DOWNLOAD_WEIGHTS:-true}
DRY_RUN=${DRY_RUN:-false}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$WNM3D_ROOT/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$WNM3D_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WNM3D_ROOT/checkpoints/umt5-xxl"}
VGGT_OMEGA_CKPT=${VGGT_OMEGA_CKPT:-"$WNM3D_ROOT/checkpoints/vggt_omega_1b_512.pt"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

download_hf_repo() {
    local repo_id="$1"
    local target_dir="$2"
    if ! command -v hf >/dev/null 2>&1; then
        echo "ERROR: '$repo_id' is missing and the 'hf' command is unavailable."
        echo "Install huggingface_hub or set the corresponding checkpoint path."
        exit 1
    fi
    hf download "$repo_id" --local-dir "$target_dir"
}

require_file_or_download_repo() {
    local required_file="$1"
    local repo_id="$2"
    local target_dir="$3"
    if [ -f "$required_file" ]; then
        return
    fi
    if is_true "$AUTO_DOWNLOAD_WEIGHTS"; then
        echo "Missing $required_file; downloading $repo_id..."
        download_hf_repo "$repo_id" "$target_dir"
    else
        echo "ERROR: Required file not found: $required_file"
        exit 1
    fi
}

if ! is_true "$DRY_RUN"; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "ERROR: Python executable not found: $PYTHON_BIN"
        exit 1
    fi
    if [ ! -d "$INTERIORGS_DATA_ROOT" ]; then
        echo "ERROR: InteriorGS dataset not found at $INTERIORGS_DATA_ROOT"
        echo "Set INTERIORGS_DATA_ROOT to the appropriate LeRobot dataset."
        exit 1
    fi

    require_file_or_download_repo \
        "$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
        Wan-AI/Wan2.2-TI2V-5B \
        "$WAN22_CKPT_DIR"
    require_file_or_download_repo \
        "$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
        Wan-AI/Wan2.2-TI2V-5B \
        "$WAN22_CKPT_DIR"
    require_file_or_download_repo \
        "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
        Wan-AI/Wan2.1-I2V-14B-480P \
        "$IMAGE_ENCODER_DIR"

    if [ ! -f "$TOKENIZER_DIR/tokenizer_config.json" ]; then
        if is_true "$AUTO_DOWNLOAD_WEIGHTS"; then
            echo "Missing UMT5 tokenizer; downloading google/umt5-xxl..."
            download_hf_repo google/umt5-xxl "$TOKENIZER_DIR"
        else
            echo "ERROR: UMT5 tokenizer not found at $TOKENIZER_DIR"
            exit 1
        fi
    fi
    if [ ! -f "$VGGT_OMEGA_CKPT" ]; then
        echo "ERROR: VGGT-Omega checkpoint not found at $VGGT_OMEGA_CKPT"
        echo "Set VGGT_OMEGA_CKPT to a compatible vggt_omega_1b_512.pt file."
        exit 1
    fi
    if [ -n "$PRETRAINED_MODEL_PATH" ] \
        && [ ! -f "$PRETRAINED_MODEL_PATH/model.safetensors" ] \
        && [ ! -f "$PRETRAINED_MODEL_PATH/model.safetensors.index.json" ]; then
        echo "ERROR: No model safetensors found under $PRETRAINED_MODEL_PATH"
        exit 1
    fi
fi

EXPERIMENT_PY="$WNM3D_ROOT/gammanav/vln/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Training entry point not found: $EXPERIMENT_PY"
    exit 1
fi

if [ "$NNODES" -eq 1 ]; then
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    RUN_CMD=(
        "$PYTHON_BIN" -m torch.distributed.run
        --nproc_per_node "$NUM_GPUS"
        --standalone
        "$EXPERIMENT_PY"
    )
else
    if [ -z "$MASTER_ADDR" ]; then
        echo "ERROR: MASTER_ADDR must be set when NNODES is greater than 1."
        exit 1
    fi
    RUN_CMD=(
        "$PYTHON_BIN" -m torch.distributed.run
        --nnodes "$NNODES"
        --node_rank "$NODE_RANK"
        --nproc_per_node "$NUM_GPUS"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
        "$EXPERIMENT_PY"
    )
fi

HYDRA_ARGS=(
    "report_to=$REPORT_TO"
    "data=wnm_3d/interiorgs_wan22"
    "wandb_project=$WANDB_PROJECT"
    "train_architecture=full"
    "num_frames=33"
    "state_horizon=4"
    "action_horizon=8"
    "model=wnm_3d/wnm_3d"
    "model/wnm_3d/action_head=wan_flow_matching_action_tf_wan22"
    "model/wnm_3d/transform=wnm_3d_cotrain"
    "num_frame_per_block=2"
    "num_action_per_block=8"
    "num_state_per_block=1"
    "seed=$SEED"
    "training_args.learning_rate=$LEARNING_RATE"
    "training_args.deepspeed=gammanav/vln/configs/deepspeed/${DEEPSPEED_CFG}.json"
    "+training_args.logging_dir=$TB_LOG_DIR"
    "logging_steps=$LOGGING_STEPS"
    "save_steps=$SAVE_STEPS"
    "training_args.warmup_ratio=$WARMUP_RATIO"
    "output_dir=$OUTPUT_DIR"
    "per_device_train_batch_size=$PER_DEVICE_BS"
    "global_batch_size=$GLOBAL_BATCH_SIZE"
    "max_steps=$MAX_STEPS"
    "weight_decay=$WEIGHT_DECAY"
    "save_total_limit=$SAVE_TOTAL_LIMIT"
    "bf16=true"
    "tf32=true"
    "eval_bf16=true"
    "dataloader_pin_memory=$DATALOADER_PIN_MEMORY"
    "dataloader_num_workers=$DATALOADER_NUM_WORKERS"
    "dataloader_persistent_workers=$DATALOADER_PERSISTENT_WORKERS"
    "image_resolution_width=320"
    "image_resolution_height=160"
    "save_lora_only=false"
    "use_vggt_geometry_adapter=true"
    "vggt_omega_checkpoint_path=$VGGT_OMEGA_CKPT"
    "vggt_image_resolution=512"
    "vggt_resize_mode=square"
    "vggt_adapter_dim=512"
    "vggt_adapter_blocks=2"
    "vggt_adapter_heads=8"
    "max_chunk_size=4"
    "video_frame_stride=$VIDEO_FRAME_STRIDE"
    "action_frame_stride=$VIDEO_FRAME_STRIDE"
    "save_strategy=$SAVE_STRATEGY"
    "interiorgs_data_root=$INTERIORGS_DATA_ROOT"
    "dit_version=$WAN22_CKPT_DIR"
    "text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
    "image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth"
    "tokenizer_path=$TOKENIZER_DIR"
)

if [ -n "$PRETRAINED_MODEL_PATH" ]; then
    HYDRA_ARGS+=(
        "pretrained_model_path=$PRETRAINED_MODEL_PATH"
        "++action_head_cfg.config.skip_component_loading=true"
    )
fi

echo "Stage: $WNM3D_TRAIN_STAGE"
echo "Dataset: $INTERIORGS_DATA_ROOT"
echo "Output: $OUTPUT_DIR"
if [ -n "$PRETRAINED_MODEL_PATH" ]; then
    echo "Initialization: $PRETRAINED_MODEL_PATH"
fi
echo "Launch: NNODES=$NNODES NODE_RANK=$NODE_RANK NUM_GPUS=$NUM_GPUS MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Batch: per_device=$PER_DEVICE_BS global=$GLOBAL_BATCH_SIZE"

cd "$WNM3D_ROOT"
if is_true "$DRY_RUN"; then
    printf 'Command:'
    printf ' %q' "${RUN_CMD[@]}" "${HYDRA_ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi

exec "${RUN_CMD[@]}" "${HYDRA_ARGS[@]}" "$@"
