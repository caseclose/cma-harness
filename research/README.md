# CMA research code

Source release for Cognitive-structured Multimodal Agent for Multimodal
Understanding, Generation, and Editing (arXiv:2607.08497).

This directory contains the memory/retrieval/controller inference implementation,
retriever SFT and RL launchers, memory-writer RL and reward service, scenario-engine
source, baseline runners, retrieval metrics and generation-quality evaluation.
M2CA-Bench, the training data and trained model checkpoints will be made publicly
available upon acceptance. They are not included in this release.

## Release scope and environment

The code is an exported research snapshot. It is separate from the extended
CMA-Harness deployment and does not contain its complete 17-tool integration.
Original scripts used a research checkout identifying itself as ms-swift
3.12.0.dev0. The public dependency below is ms-swift 3.12.0; the private fork is
not vendored. CUDA/PyTorch, FlashAttention and DeepSpeed must match your hardware.
The release has syntax and CPU unit checks; GPU training, full inference and
paper-score reproduction have not been rerun for this export.

Original model paths and output directories were replaced by public model IDs
and configurable local paths. An unused stale image-generator import was removed.
Private API clients and embedded credentials were removed; optional API calls now
use an explicitly configured chat-completions-compatible provider. The scenario
engine uses local Qwen image generation/editing in place of its private image API.
These provider substitutions do not establish equivalence to reported scores.
Evaluation prompts, score parsers, reward calculations and metric definitions
are retained from the source snapshot. SOURCE_PROVENANCE.json records source
hashes and final exported-file hashes.

## Setup

Use Python 3.10 or newer in a fresh environment. Install a suitable CUDA PyTorch
build, then install the research dependencies:

    pip install -r research/requirements.txt

Install FlashAttention separately if using the launchers' flash_attn setting.
The ms-swift directory here holds project code, not a vendored swift package.
Keep it on PYTHONPATH alongside the installed framework:

    export PYTHONPATH="$PWD/research/ms-swift:$PYTHONPATH"

Copy the values from .env.example into your shell and export the variables you
need. Checkpoint paths must refer to compatible local adapters or base models.
Dataset and checkpoint downloads will be documented when released upon acceptance.

## Training entrypoints

Run from the repository root. Dataset paths must be supplied explicitly. The
launchers retain the original optimization defaults; distributed topology is
configured through NNODES, NODE_RANK, MASTER_ADDR and NPROC_PER_NODE.

    TRAIN_DATA=/path/train.jsonl VAL_DATA=/path/val.jsonl bash research/ms-swift/train_retrieval_sft_300.sh
    DATASET_PATH=/path/retrieval_rl.jsonl SFT_ADAPTER_PATH=/path/retriever_sft bash research/ms-swift/train_retrieval_rl_dapo_300.sh

Memory-writer RL calls a frozen retriever service. Start vlmb_retrieval_service.py
on a reserved GPU (see its --help), then supply its endpoint and the writer data:

    python research/ms-swift/vlmb_retrieval_service.py --checkpoint /path/retriever --host 127.0.0.1 --port 29640 --gpu 7
    VLM_B_SERVICE_URL=http://127.0.0.1:29640/retrieve DATASET_PATH=/path/writer_rl.jsonl bash research/ms-swift/train_vlm_a_rl_dapo_300_a2.sh

Reserve distinct GPUs for the reward service and training. Conversion utilities
are in benchmark/build_retrieval_trainset_from_benchmark.py and
benchmark/build_vlm_a_rl_a2_dataset_from_retrieval_rl.py.

## Inference and evaluation

Set BASE_MODEL_PATH, VLM_A_PATH (memory writer), VLM_B_PATH (retriever), VLM_C_PATH
(controller), QWEN_GEN_PATH and QWEN_EDIT_PATH. GPU allocation is controlled in
MultimodalAgentService and by the QWEN_*_GPU(S) environment variables. The original
defaults assume multiple GPUs; inspect them before launching. Start the local API:

    python -m uvicorn agent_api_server:app --app-dir research/ms-swift --host 127.0.0.1 --port 8000

Conversation and baseline runners are in benchmark/. Pass your own scenario and
output directories. Retrieve the precise options with each script's --help after
installing dependencies. Aggregate retrieval results, for example:

    python research/benchmark/calculate_retrieval_accuracy_tags_desc.py --results-dir /path/results --scenarios-dir /path/scenarios --result-suffix kv_cache_baseline_tags_desc

The metric compares sorted predicted and target ID lists. Null or empty
predictions mean no retrieval; scenario turns without a retrieval_target are
skipped. Medium uses turns 11-20. The Hard subset additionally requires the
very_hard scenario label and must not be conflated with a plain turn-range filter.

For optional model judging, export CMA_API_BASE_URL, CMA_API_KEY and
CMA_JUDGE_MODEL. Use benchmark/evaluate_gemini_generation_quality_main.py with
--method-name, --results-dir and --scenarios-dir. Use the same judge configuration
when comparing methods. The --api-key option is also accepted. No paid API calls
were made as part of the release checks.

The gemini_multi_agent directory contains the scenario-engine source and Swift
format conversion. Configure a compatible understanding provider and local Qwen
image models before using it. No generated dialogues are bundled.

## Offline checks

These use synthetic temporary records and mocked HTTP, not M2CA-Bench experiments:

    python -m unittest discover -s research/tests -v

The offline checks require only Pillow and requests, rather than GPU packages.
