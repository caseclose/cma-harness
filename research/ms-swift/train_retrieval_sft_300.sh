#!/bin/bash
# Retrieval SFT training launcher for 300-session benchmark data.

set -euo pipefail

# ===== 固定参数（直接写入）=====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MAX_PIXELS="100352000"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29620}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NCCL_IB_DISABLE="1"
export NCCL_DEBUG="INFO"
export NCCL_ASYNC_ERROR_HANDLING="1"

# 强制 NCCL/GLOO 走正确物理网卡（避免选到不可达 IP，比如 29.166.*）
# 自动选择与 MASTER_ADDR 同前两段（如 29.163.*）的本机 IPv4 所在网卡。
if [[ -z "${NCCL_SOCKET_IFNAME:-}" || -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  MASTER_PREFIX="$(echo "${MASTER_ADDR}" | cut -d. -f1-2)"
  IFNAME="$(ip -o -4 addr show | awk -v p="${MASTER_PREFIX}" '{split($4,a,"/"); split(a[1],b,"."); pref=b[1]"."b[2]; if(pref==p){print $2; exit}}')"
  if [[ -n "${IFNAME}" ]]; then
    export NCCL_SOCKET_IFNAME="${IFNAME}"
    export GLOO_SOCKET_IFNAME="${IFNAME}"
  else
    # 兜底：排除常见虚拟网卡
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-^lo,docker0,virbr0}"
  fi
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
# 多模态对齐版：tags+description + 448缩略图（images字段 + <image>）
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to your local dataset file}"
VAL_DATA="${VAL_DATA:?Set VAL_DATA to your local dataset file}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/train_retrieval_sft_300}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_EPOCHS=3
TRAIN_BS=1
EVAL_BS=1
GRAD_ACC=4
LR=2e-5
WARMUP_RATIO=0.1
LORA_RANK=8
LORA_ALPHA=32
MAX_LENGTH=8192
DEEPSPEED_STAGE="zero2"
LOGGING_STEPS=5
EVAL_STEPS=100
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3
DATALOADER_WORKERS=4
DATASET_NUM_PROC=4
SEED=42

mkdir -p "${OUTPUT_DIR}"

echo "Starting retrieval SFT training (300-session)..."
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "VAL_DATA=${VAL_DATA}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"

# 避免继承旧的分布式环境，导致 world_size/local_world_size 异常
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK

NPROC_PER_NODE="${NPROC_PER_NODE}" \
python -m swift.cli.main sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl \
    --train_type lora \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --split_dataset_ratio 0 \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${TRAIN_BS}" \
    --per_device_eval_batch_size "${EVAL_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules all-linear \
    --freeze_vit true \
    --max_length "${MAX_LENGTH}" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --deepspeed "${DEEPSPEED_STAGE}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --dataloader_num_workers "${DATALOADER_WORKERS}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}"

echo "SFT training done. Checkpoints are under: ${OUTPUT_DIR}"
