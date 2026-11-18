#!/bin/bash
# VLM A RL (A2) launcher:
# - Optimize VLM A (tags+description generator).
# - Reward computed by downstream retrieval accuracy from a frozen VLM B service.
#
# Requirements:
# - Start local VLM B service on each node (reserved GPU) and export VLM_B_SERVICE_URL.
# - Dataset built by build_vlm_a_rl_a2_dataset_from_retrieval_rl.py

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MAX_PIXELS="${MAX_PIXELS:-100352000}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"

# Force localhost reward-service requests to bypass any corporate/http proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to your local dataset file}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/train_vlm_a_rl_dapo_300_a2}"
PLUGIN_PATH="${PLUGIN_PATH:-${SCRIPT_DIR}/examples/train/grpo/plugin/vlm_a_downstream_retrieval_reward.py}"

# Reward->VLM B service defaults for 8-node A2 training.
export VLM_B_SERVICE_TIMEOUT_S="${VLM_B_SERVICE_TIMEOUT_S:-300}"
export VLM_B_SERVICE_BATCH_SIZE="${VLM_B_SERVICE_BATCH_SIZE:-4}"
export VLM_B_SERVICE_MAX_RETRIES="${VLM_B_SERVICE_MAX_RETRIES:-2}"
export VLM_B_SERVICE_RETRY_WAIT_S="${VLM_B_SERVICE_RETRY_WAIT_S:-2}"
# Reward mode:
# - auto: infer from sample field `reward_mode`
# - single_regen: force legacy A2 one-rank rewrite
# - all_ranks: force full memory-bank rewrite reward
export VLM_A_REWARD_MODE="${VLM_A_REWARD_MODE:-auto}"

USE_VLLM="${USE_VLLM:-false}"  # keep false (vLLM may not support Qwen3-VL in this env)

mkdir -p "${OUTPUT_DIR}"

echo "Starting VLM A RL (A2) training..."
echo "MODEL_PATH=${MODEL_PATH}"
echo "SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-<none>}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PLUGIN_PATH=${PLUGIN_PATH}"
echo "VLM_B_SERVICE_URL=${VLM_B_SERVICE_URL:-<missing>}"
echo "VLM_B_SERVICE_TIMEOUT_S=${VLM_B_SERVICE_TIMEOUT_S}"
echo "VLM_B_SERVICE_BATCH_SIZE=${VLM_B_SERVICE_BATCH_SIZE}"
echo "VLM_B_SERVICE_MAX_RETRIES=${VLM_B_SERVICE_MAX_RETRIES}"
echo "VLM_A_REWARD_MODE=${VLM_A_REWARD_MODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE:-7}"

if [[ -z "${VLM_B_SERVICE_URL:-}" ]]; then
  echo "ERROR: VLM_B_SERVICE_URL is missing. Please start local VLM B service first."
  exit 1
fi

# Wait for local VLM B service to become reachable before launching distributed RL.
python - <<'PY'
import json
import os
import time
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import URLError, HTTPError

service_url = os.environ["VLM_B_SERVICE_URL"]
timeout_s = int(os.environ.get("VLM_B_STARTUP_TIMEOUT_S", "60"))
interval_s = float(os.environ.get("VLM_B_STARTUP_RETRY_S", "2"))

u = urlparse(service_url)
health_url = urlunparse((u.scheme, u.netloc, "/health", "", "", ""))
retrieve_url = urlunparse((u.scheme, u.netloc, "/retrieve", "", "", ""))

opener = build_opener(ProxyHandler({}))
deadline = time.time() + timeout_s
last_err = ""
ok = False

while time.time() < deadline:
    try:
        with opener.open(health_url, timeout=5) as r:
            body = r.read().decode("utf-8", errors="ignore")
            if r.status == 200 and '"ok"' in body.lower():
                ok = True
                break
    except Exception as e:
        last_err = f"health: {e}"
    try:
        req = Request(
            retrieve_url,
            data=json.dumps({"requests": [{"query": "ping", "candidates": [], "images": []}]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=5) as r:
            # /retrieve can be 200 (ok true) or 400 for malformed payload in other versions,
            # either means service is up and reachable.
            if r.status in (200, 400):
                ok = True
                break
    except HTTPError as e:
        if e.code in (200, 400):
            ok = True
            break
        last_err = f"retrieve HTTP {e.code}: {e.reason}"
    except URLError as e:
        last_err = f"retrieve urlerror: {e}"
    except Exception as e:
        last_err = f"retrieve: {e}"
    time.sleep(interval_s)

if not ok:
    raise SystemExit(
        f"VLM B service not reachable within {timeout_s}s. "
        f"VLM_B_SERVICE_URL={service_url}. Last error: {last_err}"
    )

print(f"VLM B service is reachable: {service_url}")
PY

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
  --max_completion_length "${MAX_COMPLETION_LEN:-512}"
  --num_generations "${NUM_GENERATIONS:-4}"
  --temperature "${TEMP:-1.0}"
  --top_p "${TOP_P:-0.9}"
  --top_k "${TOP_K:-50}"
  --reward_funcs vlm_a_downstream_retrieval soft_overlong
  --reward_weights "${REWARD_MAIN_WEIGHT:-1.0}" "${REWARD_LEN_WEIGHT:-0.1}"
  --dynamic_sample true
  --overlong_filter true
  --soft_max_length "${SOFT_MAX_LENGTH:-512}"
  --soft_cache_length "${SOFT_CACHE_LENGTH:-64}"
  --external_plugins "${PLUGIN_PATH}"
  --deepspeed "${DEEPSPEED_STAGE:-zero2}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --save_steps "${SAVE_STEPS:-100}"
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
  CMD+=(--use_vllm true)
else
  CMD+=(--use_vllm false)
fi

if [[ -n "${SFT_ADAPTER_PATH}" ]]; then
  CMD+=(--adapters "${SFT_ADAPTER_PATH}")
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-7}" "${CMD[@]}"

echo "VLM A RL (A2) done. Checkpoints are under: ${OUTPUT_DIR}"
