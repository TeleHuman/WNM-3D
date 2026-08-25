#!/usr/bin/env bash
# Stage II: DAgger fine-tuning on multiple nodes.
#
# Run this script on every node. Example for two 8-GPU nodes:
#   MASTER_ADDR=<node-0-ip> NODE_RANK=0 bash scripts/train/wnm_3d_stage2_distributed.sh
#   MASTER_ADDR=<node-0-ip> NODE_RANK=1 bash scripts/train/wnm_3d_stage2_distributed.sh
#
# Cluster-specific network settings can be supplied through NCCL_SOCKET_IFNAME,
# GLOO_SOCKET_IFNAME, and NCCL_IB_HCA (for example, bond0 and mlx5_0,...,mlx5_7).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${MASTER_ADDR:-}" ]; then
    echo "ERROR: Set MASTER_ADDR to the address of node 0."
    exit 1
fi
if [ -z "${NODE_RANK:-${RANK:-}}" ]; then
    echo "ERROR: Set NODE_RANK=0 on node 0 and NODE_RANK=1 on node 1."
    exit 1
fi

export MAX_STEPS=${MAX_STEPS:-35000}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NUM_GPUS=${NUM_GPUS:-$GPUS_PER_NODE}
export NNODES=${NNODES:-2}
export NODE_RANK=${NODE_RANK:-$RANK}
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-12345}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

exec bash "$SCRIPT_DIR/wnm_3d_stage2.sh" "$@"
