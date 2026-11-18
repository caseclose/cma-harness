#!/bin/bash
# Retrieval RL (DAPO/GRPO) launcher for 300-session benchmark data.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MAX_PIXELS="${MAX_PIXELS:-100352000}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to your local dataset file}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/train_retrieval_rl_dapo_300}"
PLUGIN_PATH="${PLUGIN_PATH:-${SCRIPT_DIR}/examples/train/grpo/plugin/retrieval_reward_300.py}"
USE_VLLM="${USE_VLLM:-false}"

mkdir -p "${OUTPUT_DIR}"

echo "Starting retrieval RL (DAPO/GRPO) training..."
echo "MODEL_PATH=${MODEL_PATH}"
echo "SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-<none>}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PLUGIN_PATH=${PLUGIN_PATH}"
echo "USE_VLLM=${USE_VLLM}"

# 避免继承旧的分布式环境，导致 world_size/local_world_size 异常
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK

CMD=(
  python -m swift.cli.main rlhf
  --rlhf_type grpo
  --model "${MODEL_PATH}"
  --model_type qwen3_vl
  --train_type lora
  --dataset "${DATASET_PATH}"
  --load_from_cache_file true
  --split_dataset_ratio 0
  --torch_dtype bfloat16
  --attn_impl flash_attn
  --num_train_epochs "${NUM_EPOCHS:-1}"
  --per_device_train_batch_size "${TRAIN_BS:-1}"
  --gradient_accumulation_steps "${GRAD_ACC:-8}"
  --learning_rate "${LR:-2e-6}"
  --lora_rank "${LORA_RANK:-8}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --target_modules all-linear
  --freeze_vit true
  --max_completion_length "${MAX_COMPLETION_LEN:-256}"
  --num_generations "${NUM_GENERATIONS:-8}"
  --temperature "${TEMP:-1.0}"
  --top_p "${TOP_P:-0.9}"
  --top_k "${TOP_K:-50}"
  --reward_funcs retrieval_soft_match_300 soft_overlong
  --reward_weights "${REWARD_MAIN_WEIGHT:-1.0}" "${REWARD_LEN_WEIGHT:-0.1}"
  --dynamic_sample true
  --overlong_filter true
  --soft_max_length "${SOFT_MAX_LENGTH:-256}"
  --soft_cache_length "${SOFT_CACHE_LENGTH:-64}"
  --external_plugins "${PLUGIN_PATH}"
  --deepspeed "${DEEPSPEED_STAGE:-zero2}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --save_steps "${SAVE_STEPS:-200}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
  --warmup_ratio "${WARMUP_RATIO:-0.05}"
  --dataloader_num_workers "${DATALOADER_WORKERS:-4}"
  --dataset_num_proc "${DATASET_NUM_PROC:-4}"
  --log_completions true
  --num_iterations "${NUM_ITERATIONS:-1}"
  --beta "${BETA:-0.04}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${USE_VLLM}" == "true" ]]; then
  # NOTE:
  # vLLM may not support Qwen3-VL on some environments. If you hit errors like:
  #   AttributeError: 'Qwen3VLConfig' object has no attribute 'vocab_size'
  # set USE_VLLM=false (default) to fall back to HF generation.
  CMD+=(
    --use_vllm true
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM_UTIL:-0.6}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-4096}"
  )
else
  CMD+=(--use_vllm false)
fi

if [[ -n "${SFT_ADAPTER_PATH}" ]]; then
  CMD+=(--adapters "${SFT_ADAPTER_PATH}")
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}" "${CMD[@]}"

echo "RL training done. Checkpoints are under: ${OUTPUT_DIR}"
