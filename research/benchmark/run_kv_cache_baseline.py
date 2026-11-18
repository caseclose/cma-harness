"""
KV Cache baseline comparison script
Uses full conversation history and all images as input to compare Qwen3VL + Qwen-image with the current agent

At turn n:
- Use conversation history and all images from turns 1~n-1 as input
- If turn n is a generate/edit task, call the Qwen-image model
- If turn n is an understand/chat task, call the Qwen3VL model

Usage:
    python run_kv_cache_baseline.py --scenario 50x10-gemini/sample_001.jsonl
    python run_kv_cache_baseline.py --start 1 --end 10 --max-workers 4
"""
import argparse
import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from PIL import Image
BENCHMARK_DIR = Path(__file__).resolve().parent

def _find_ms_swift_root() -> Path:
    env = os.environ.get('MS_SWIFT_ROOT')
    if env:
        p = Path(env).expanduser().resolve()
        if (p / 'multimodal_agent_service.py').is_file():
            return p
    a = BENCHMARK_DIR.parent / 'ms-swift'
    if (a / 'multimodal_agent_service.py').is_file():
        return a.resolve()
    b = BENCHMARK_DIR.parent.parent / 'ms-swift'
    if (b / 'multimodal_agent_service.py').is_file():
        return b.resolve()
    return a.resolve()
PROJECT_ROOT = _find_ms_swift_root()
sys.path.insert(0, str(PROJECT_ROOT))
SCENARIOS_DIR = BENCHMARK_DIR / 'scenarios' / '50x10-gemini'
OUTPUT_DIR = BENCHMARK_DIR / 'kv_cache_baseline_results'
OUTPUT_DIR.mkdir(exist_ok=True)
_shared_qwen3vl = None
_shared_qwen_image = None
_qwen_image_lock = threading.Lock()
_qwen3vl_lock = threading.Lock()
_model_lock = threading.Lock()

def align_image_to_model(image_path: str) -> str:
    """Align image size to avoid model-side size mismatch errors"""
    try:
        with Image.open(image_path) as img:
            target_size = 896
            if img.size != (target_size, target_size) or img.mode != 'RGB':
                img = img.convert('RGB')
                img = img.resize((target_size, target_size), resample=Image.LANCZOS)
                img.save(image_path, quality=95)
        return image_path
    except Exception as e:
        print(f'Warning: image alignment failed {e}')
        return image_path

def init_qwen3vl_model():
    """Initialize Qwen3VL model (for understand/chat tasks)"""
    global _shared_qwen3vl
    if _shared_qwen3vl is None:
        with _model_lock:
            if _shared_qwen3vl is None:
                print('=' * 80)
                print('Initializing Qwen3VL model...')
                print('=' * 80)
                try:
                    import torch
                    try:
                        import transformers
                        transformers_version = transformers.__version__
                        print(f'Detected transformers version: {transformers_version}')
                        from packaging import version
                        if version.parse(transformers_version) < version.parse('4.57'):
                            raise ImportError(f"transformers version is too low!\nCurrent version: {transformers_version}\nRequired version: >= 4.57\nPlease run: pip install 'transformers>=4.57' 'qwen_vl_utils>=0.0.14' 'huggingface_hub>=0.20.0' -U")
                    except ImportError as e:
                        if 'version' in str(e).lower() or '4.57' in str(e):
                            raise
                        print(f'⚠️  Warning: failed to check transformers version')
                    try:
                        import huggingface_hub
                        hf_hub_version = huggingface_hub.__version__
                        print(f'Detected huggingface_hub version: {hf_hub_version}')
                        from packaging import version
                        if version.parse(hf_hub_version) < version.parse('0.20.0'):
                            raise ImportError(f"huggingface_hub version is too low!\nCurrent version: {hf_hub_version}\nRequired version: >= 0.20.0\nPlease run: pip install 'huggingface_hub>=0.20.0' -U")
                    except ImportError as e:
                        if 'version' in str(e).lower() or '0.20' in str(e):
                            raise
                        print(f'⚠️  Warning: failed to check huggingface_hub version')
                    try:
                        import qwen_vl_utils
                        print(f'✓ qwen_vl_utils is installed')
                    except ImportError:
                        raise ImportError("Missing qwen_vl_utils package!\nPlease run: pip install 'transformers>=4.57' 'qwen_vl_utils>=0.0.14' 'huggingface_hub>=0.20.0' 'tokenizers>=0.22.0,<=0.23.0' -U")
                    try:
                        import tokenizers
                        tokenizers_version = tokenizers.__version__
                        print(f'Detected tokenizers version: {tokenizers_version}')
                        from packaging import version
                        if version.parse(tokenizers_version) < version.parse('0.22.0') or version.parse(tokenizers_version) > version.parse('0.23.0'):
                            raise ImportError(f"tokenizers version is incompatible!\nCurrent version: {tokenizers_version}\nRequired version: >= 0.22.0, <= 0.23.0\nPlease run: pip install 'tokenizers>=0.22.0,<=0.23.0' -U")
                        print(f'✓ tokenizers version meets requirement')
                    except ImportError as e:
                        if 'version' in str(e).lower() or '0.22' in str(e):
                            raise
                        print(f'⚠️  Warning: failed to check tokenizers version')
                    from swift.llm import get_model_tokenizer, get_template, PtEngine, RequestConfig, InferRequest
                    model_path = os.environ.get('QWEN3VL_MODEL_PATH', './data/Qwen3-VL-235B-A22B-Instruct')
                    model_name = Path(model_path).name
                    size_hint = 'unknown'
                    for tag in ['235B', '72B', '32B', '8B', '7B']:
                        if tag in model_name:
                            size_hint = tag
                            break
                    print(f'Loading Qwen3VL model: {model_path}')
                    print(f'Model ID: {model_name}')
                    print(f'Model size: {size_hint}')
                    print('⚠️  Warning: loading large models may take a long time, please wait...')
                    print(f'Start loading model weights...')
                    sys.stdout.flush()
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f'Model path does not exist: {model_path}')
                    import torch
                    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    print(f'Available GPU count: {gpu_count}')
                    if gpu_count == 0:
                        raise RuntimeError('No available GPU; model cannot run on CPU')
                    qwen3vl_gpus_env = os.environ.get('VLM_SHARED_GPUS') or os.environ.get('QWEN3VL_GPUS')
                    qwen_image_gpus_env = os.environ.get('QWEN_EDIT_GPUS') or os.environ.get('QWEN_IMAGE_GPUS')
                    if qwen3vl_gpus_env:
                        qwen3vl_gpus = [int(x.strip()) for x in qwen3vl_gpus_env.split(',') if x.strip()]
                    else:
                        qwen3vl_gpus = list(range(min(7, gpu_count)))
                    if qwen_image_gpus_env:
                        qwen_image_gpus = [int(x.strip()) for x in qwen_image_gpus_env.split(',') if x.strip()]
                    else:
                        qwen_image_gpus = [len(qwen3vl_gpus)] if gpu_count > len(qwen3vl_gpus) else []
                    qwen3vl_gpu_count = len(qwen3vl_gpus)
                    qwen_image_gpu_count = len(qwen_image_gpus)
                    if qwen3vl_gpu_count < 4:
                        print(f'⚠️  Warning: large model is recommended to use at least 4-7 GPUs; currently assigned {qwen3vl_gpu_count} GPUs')
                    if qwen_image_gpu_count == 0:
                        print(f'⚠️  Warning: no GPU available for Qwen-image; allocation strategy may need adjustment')
                    print(f'GPU allocation strategy:')
                    print(f'  Qwen3VL: GPU {qwen3vl_gpus} ({qwen3vl_gpu_count} cards)')
                    if qwen_image_gpu_count > 0:
                        print(f'  Qwen-image: GPU {qwen_image_gpus} ({qwen_image_gpu_count} cards)')
                    for i in range(gpu_count):
                        if torch.cuda.is_available():
                            props = torch.cuda.get_device_properties(i)
                            memory_total = props.total_memory / 1024 ** 3
                            memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                            memory_free = memory_total - memory_allocated
                            if i in qwen3vl_gpus:
                                gpu_type = 'Qwen3VL'
                            elif i in qwen_image_gpus:
                                gpu_type = 'Qwen-image'
                            else:
                                gpu_type = 'unused'
                            print(f'  GPU {i}: {props.name}, total: {memory_total:.1f}GB, used: {memory_allocated:.1f}GB, free: {memory_free:.1f}GB [{gpu_type}]')
                    sys.stdout.flush()
                    print(f'[1/4] Loading model and tokenizer (this may take a while)...')
                    print(f'     Reading weight files from disk...')
                    print(f'     Assigned to GPU 0-{qwen3vl_gpu_count - 1} ({qwen3vl_gpu_count} cards GPU)')
                    sys.stdout.flush()
                    import time
                    load_start = time.time()
                    try:
                        print(f'     Calling get_model_tokenizer...')
                        sys.stdout.flush()
                        max_memory_dict = {}
                        for gpu_id in range(gpu_count):
                            if gpu_id in qwen3vl_gpus and gpu_id not in qwen_image_gpus:
                                if torch.cuda.is_available():
                                    props = torch.cuda.get_device_properties(gpu_id)
                                    total_memory_gb = props.total_memory / 1024 ** 3
                                    max_memory_gb = int(total_memory_gb * 0.8)
                                    max_memory_dict[gpu_id] = f'{max_memory_gb}GiB'
                            else:
                                max_memory_dict[gpu_id] = '0GiB'
                        print(f'     Set max_memory (80% VRAM): {max_memory_dict}')
                        sys.stdout.flush()
                        model, tokenizer = get_model_tokenizer(model_path, torch_dtype=torch.bfloat16, device_map='auto', max_memory=max_memory_dict if max_memory_dict else None)
                        load_duration = time.time() - load_start
                        print(f'[2/4] Model weights loaded (elapsed  {load_duration:.1f} sec）')
                        sys.stdout.flush()
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                            raise RuntimeError(f'Insufficient VRAM! Model {model_name} requires large GPU memory。\nCurrent GPUs: {gpu_count}\nSuggestion: use at least 4-8 GPUs (each >=40GB VRAM)\nor use a smaller model') from e
                        raise
                    print(f'[3/4] Get template...')
                    sys.stdout.flush()
                    template = get_template('qwen3_vl', tokenizer)
                    print(f'[4/4] Create inference engine...')
                    sys.stdout.flush()
                    engine = PtEngine.from_model_template(model, template)
                    _shared_qwen3vl = {'model': model, 'tokenizer': tokenizer, 'template': template, 'engine': engine}
                    print('✓ Qwen3VL model initialization completed')
                    print('=' * 80)
                    print()
                except ImportError as e:
                    error_msg = str(e)
                    if 'transformers' in error_msg.lower() or 'qwen_vl_utils' in error_msg.lower() or 'Qwen3VL' in error_msg or ('huggingface_hub' in error_msg.lower()) or ('is_offline_mode' in error_msg) or ('tokenizers' in error_msg.lower()):
                        print(f'❌ Qwen3VL model initialization failed: incompatible dependency versions')
                        print(f'   Error details: {error_msg}')
                        print()
                        print('=' * 80)
                        print('Solution:')
                        print('=' * 80)
                        print('Please run the following command to upgrade dependencies:')
                        print()
                        print("  pip install 'transformers>=4.57' 'qwen_vl_utils>=0.0.14' 'huggingface_hub>=0.20.0' 'tokenizers>=0.22.0,<=0.23.0' -U")
                        print()
                        print('Or run the installation script:')
                        print('  ./install_qwen3vl_deps.sh')
                        print()
                        print('If issues persist, try installing transformers from source (will install compatible deps):')
                        print('  pip install git+https://github.com/huggingface/transformers')
                        print('=' * 80)
                    else:
                        print(f'❌ Qwen3VL model initialization failed: {e}')
                        import traceback
                        traceback.print_exc()
                    sys.exit(1)
                except Exception as e:
                    print(f'❌ Qwen3VL model initialization failed: {e}')
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)
    return _shared_qwen3vl

def init_qwen_image_model():
    """Initialize Qwen-image model (for generate/edit tasks)"""
    global _shared_qwen_image
    if _shared_qwen_image is None:
        with _model_lock:
            if _shared_qwen_image is None:
                print('=' * 80)
                print('Initializing Qwen-image model...')
                print('=' * 80)
                try:
                    import torch
                    from diffusers import QwenImagePipeline, QwenImageEditPlusPipeline
                    import math
                    from diffusers import FlowMatchEulerDiscreteScheduler
                    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    qwen3vl_gpus_env = os.environ.get('VLM_SHARED_GPUS') or os.environ.get('QWEN3VL_GPUS')
                    qwen_image_gpus_env = os.environ.get('QWEN_EDIT_GPUS') or os.environ.get('QWEN_IMAGE_GPUS')
                    if qwen3vl_gpus_env:
                        qwen3vl_gpus = [int(x.strip()) for x in qwen3vl_gpus_env.split(',') if x.strip()]
                    else:
                        qwen3vl_gpus = list(range(min(7, gpu_count)))
                    if qwen_image_gpus_env:
                        qwen_image_gpus = [int(x.strip()) for x in qwen_image_gpus_env.split(',') if x.strip()]
                    else:
                        qwen_image_gpus = [len(qwen3vl_gpus)] if gpu_count > len(qwen3vl_gpus) else []
                    qwen_image_gpu_env = os.environ.get('QWEN_IMAGE_GPU', None)
                    if qwen_image_gpu_env:
                        qwen_image_gpus = [int(qwen_image_gpu_env)]
                        print(f'Using GPU specified by environment variable: {qwen_image_gpus[0]}')
                    elif qwen_image_gpus:
                        print(f'Auto-assigned GPU: {qwen_image_gpus} ({len(qwen_image_gpus)} cards)')
                    else:
                        raise RuntimeError(f'No free GPU is allocated to Qwen-image!\nQwen3VL already uses GPU {qwen3vl_gpus} ({len(qwen3vl_gpus)} cards)\nTotal GPU count: {gpu_count}\nSuggestion: add more GPUs or reduce GPUs used by Qwen3VL')

                    def _resolve_qwen_image_edit_dir() -> str:
                        env_p = os.environ.get('QWEN_IMAGE_EDIT_MODEL_PATH') or os.environ.get('QWEN_IMAGE_MODEL_PATH')
                        if env_p:
                            return env_p
                        candidates = ['Qwen/Qwen-Image-Edit-2511', 'Qwen/Qwen-Image-Edit-2511']
                        for p in candidates:
                            if Path(p).is_dir():
                                return p
                        return candidates[0]

                    def _resolve_lightning_lora_path() -> Optional[str]:
                        """Return .safetensors path if present; allow disabling via QWEN_IMAGE_EDIT_LORA_PATH=0."""
                        raw = os.environ.get('QWEN_IMAGE_EDIT_LORA_PATH')
                        if raw in ('0', 'false', 'False', ''):
                            return None
                        if raw and Path(raw).is_file():
                            return raw
                        fallback = './data/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
                        if Path(fallback).is_file():
                            return fallback
                        return None
                    edit_model_path = _resolve_qwen_image_edit_dir()
                    generate_model_path = edit_model_path
                    edit_lora_path = _resolve_lightning_lora_path()
                    print(f'Loading Qwen-image model (shared for generate/edit): {generate_model_path}')
                    print(f'Loading Qwen-image edit model: {edit_model_path}')
                    scheduler_config = {'base_image_seq_len': 256, 'base_shift': math.log(3), 'invert_sigmas': False, 'max_image_seq_len': 8192, 'max_shift': math.log(3), 'num_train_timesteps': 1000, 'shift': 1.0, 'shift_terminal': None, 'stochastic_sampling': False, 'time_shift_type': 'exponential', 'use_beta_sigmas': False, 'use_dynamic_shifting': True, 'use_exponential_sigmas': False, 'use_karras_sigmas': False}
                    pipelines = []
                    pipeline_locks = []
                    for gpu_id in qwen_image_gpus:
                        device = f'cuda:{gpu_id}'
                        print(f'Loading Qwen-image onto GPU {gpu_id}')
                        with torch.cuda.device(gpu_id):
                            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
                            edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(edit_model_path, scheduler=scheduler, torch_dtype=torch.bfloat16, trust_remote_code=True)
                            edit_pipeline = edit_pipeline.to(device)
                            if edit_lora_path:
                                print(f'Loading Lightning LoRA weights: {edit_lora_path}')
                                edit_pipeline.load_lora_weights(edit_lora_path)
                                edit_pipeline.fuse_lora()
                                edit_pipeline = edit_pipeline.to(device)
                            else:
                                print('⚠️  Lightning LoRA not found (set QWEN_IMAGE_EDIT_LORA_PATH or place the .safetensors file). Continuing without LoRA — diffusion may be slower.')
                        pipelines.append((edit_pipeline, device))
                        pipeline_locks.append(threading.Lock())
                    _shared_qwen_image = {'pipelines': pipelines, 'locks': pipeline_locks, 'current_idx': 0, 'lock': threading.Lock(), 'gpu_list': qwen_image_gpus}
                    print(f'✓ Qwen-image model initialization completed (GPU {qwen_image_gpus})')
                    print('=' * 80)
                    print()
                except Exception as e:
                    print(f'❌ Qwen-image model initialization failed: {e}')
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)
    return _shared_qwen_image

@dataclass
class TurnResult:
    """Single-turn conversation result"""
    turn: int
    user_input: str
    task_type: str
    agent_output: Dict
    all_history_images: List[str]
    conversation_history: List[Dict]

def call_qwen3vl(qwen3vl, messages: List[Dict], images: Optional[List[str]]=None) -> str:
    """Call Qwen3VL model"""
    from swift.llm import RequestConfig, InferRequest
    engine = qwen3vl['engine']
    template = qwen3vl['template']
    processed_images = None
    if images:
        processed_images = [align_image_to_model(img) for img in images]
        content = messages[0]['content']
        image_tag = '<image>'
        existing_tags = content.count(image_tag)
        if existing_tags < len(processed_images):
            messages[0]['content'] = (image_tag + '\n') * (len(processed_images) - existing_tags) + content
    request_config = RequestConfig(max_tokens=2048, temperature=0.0)
    infer_request = InferRequest(messages=messages, images=processed_images)
    with _qwen3vl_lock:
        resp_list = engine.infer([infer_request], request_config, template=template)
        return resp_list[0].choices[0].message.content

def call_qwen3vl_for_image_selection(qwen3vl, history_messages: List[Dict], history_images: List[str], history_image_turns: Optional[List[int]], user_input: str, enable_refine: bool) -> Dict[str, Optional[str]]:
    """
    Let Qwen3VL select image indices for edit/generate (multi-select) and optionally output refine instruction.
    Returns {"selected_index": List[int] | int | None, "refine_prompt": str or None}
    """
    import json as _json
    from copy import deepcopy
    image_list_lines = []
    for idx in range(len(history_images)):
        if history_image_turns and idx < len(history_image_turns):
            image_list_lines.append(f'- image_index: {idx}_turn{history_image_turns[idx]}')
        else:
            image_list_lines.append(f'- image_index: {idx}')
    system_hint = 'You are a strict multimodal assistant. Based on conversation history and user instruction,select the most relevant image indices from the given historical image list (if needed, can be multiple).and return strict JSON.You must retrieve by index (output indices only, not filenames or paths).If no image is needed, set selected_index to null.Index format: image_index: 0_turn1 means index 0 corresponds to the image generated in turn 1.'
    refine_hint = 'Also provide a refine_prompt (a clearer/more executable edit or generation instruction).' if enable_refine else 'Do not rewrite user instruction; set refine_prompt to null.'
    instruction = f'{system_hint}\n{refine_hint}\n\nHistorical image index list:\n' + ('\n'.join(image_list_lines) if image_list_lines else '(no historical images)') + '\n\nOutput must be JSON format:\n{\n  "selected_index": [0, 2],\n  "refine_prompt": "..." | null\n}\nDo not output any extra text.'
    messages = deepcopy(history_messages)
    messages.append({'role': 'user', 'content': instruction + '\n\nUser instruction: ' + user_input})
    result_text = call_qwen3vl(qwen3vl, messages, history_images if history_images else None)
    try:
        parsed = _json.loads(result_text)
        selected_index = parsed.get('selected_index', None)
        refine_prompt = parsed.get('refine_prompt', None)
        if isinstance(selected_index, list):
            selected_index = [i for i in selected_index if isinstance(i, int)]
        elif isinstance(selected_index, int):
            selected_index = [selected_index]
        else:
            selected_index = None
        return {'selected_index': selected_index, 'refine_prompt': refine_prompt}
    except Exception:
        return {'selected_index': None, 'refine_prompt': None}

def call_qwen3vl_for_understand_selection(qwen3vl, history_messages: List[Dict], history_images: List[str], history_image_turns: Optional[List[int]], user_input: str) -> Dict[str, Optional[List[int]]]:
    """
    Let Qwen3VL choose required image indices for understand tasks (multi-select).
    Returns {"selected_index": List[int] | None}
    """
    import json as _json
    from copy import deepcopy
    image_list_lines = []
    for idx in range(len(history_images)):
        if history_image_turns and idx < len(history_image_turns):
            image_list_lines.append(f'- image_index: {idx}_turn{history_image_turns[idx]}')
        else:
            image_list_lines.append(f'- image_index: {idx}')
    instruction = 'You are a strict multimodal assistant. Based on conversation history and user instruction,Select image indices needed for understand task from the given historical image list (can be multiple).If no image is needed, set selected_index to null.\n\nYou must retrieve by index (output indices only, not filenames or paths).\n\nIndex format: image_index: 0_turn1 means index 0 corresponds to the image generated in turn 1.\n\nHistorical image index list:\n' + ('\n'.join(image_list_lines) if image_list_lines else '(no historical images)') + f'\n\nOutput must be JSON format:\n{{\n  "selected_index": [0, 2]\n}}\nDo not output any extra text.\n\nUser instruction: {user_input}'
    messages = deepcopy(history_messages)
    messages.append({'role': 'user', 'content': instruction})
    result_text = call_qwen3vl(qwen3vl, messages, history_images if history_images else None)
    try:
        parsed = _json.loads(result_text)
        selected_index = parsed.get('selected_index', None)
        if isinstance(selected_index, list):
            selected_index = [i for i in selected_index if isinstance(i, int)]
        elif isinstance(selected_index, int):
            selected_index = [selected_index]
        else:
            selected_index = None
        return {'selected_index': selected_index}
    except Exception:
        return {'selected_index': None}

def call_qwen_image(qwen_image, prompt: str, images: Optional[List[str]]=None, task_type: str='generate') -> str:
    """Call Qwen-image model to generate image"""
    import torch
    import numpy as np
    from PIL import Image

    def _get_next_pipeline(qwen_image):
        with qwen_image['lock']:
            idx = qwen_image['current_idx']
            pipeline, device = qwen_image['pipelines'][idx]
            pipeline_lock = qwen_image['locks'][idx]
            qwen_image['current_idx'] = (idx + 1) % len(qwen_image['pipelines'])
            return (pipeline, device, pipeline_lock)
    edit_pipeline, device, pipeline_lock = _get_next_pipeline(qwen_image)
    generator = torch.Generator(device=device).manual_seed(42)
    input_image = None
    if images and len(images) > 0:
        img_path = images[0]
        if os.path.exists(img_path):
            input_image = Image.open(img_path).convert('RGB')
    else:
        dummy_image = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8))
        input_image = dummy_image
    positive_magic = {'en': ', Ultra HD, 4K, cinematic composition.', 'zh': ', ultra clear, 4K, cinematic composition.'}
    is_chinese = any(('一' <= char <= '鿿' for char in prompt))
    magic_suffix = positive_magic['zh'] if is_chinese else positive_magic['en']
    full_prompt = prompt + magic_suffix
    with pipeline_lock:
        with torch.inference_mode():
            try:
                output_images = edit_pipeline(image=input_image, prompt=full_prompt, height=1024, width=1024, num_inference_steps=4, generator=generator, true_cfg_scale=4.0, num_images_per_prompt=1).images
            except AttributeError as e:
                if "'NoneType' object has no attribute 'size'" in str(e):
                    print(f'⚠️  Warning: Qwen-image pipeline call failed: {e}')
                    return None
                raise
    return output_images[0] if output_images else None

def run_single_baseline(sample_id: int, output_dir: Path=None, scenario_path_override: Optional[Path]=None, scenario_label: Optional[str]=None, enable_refine: bool=False) -> Tuple[int, bool, str, float]:
    """
    Run KV Cache baseline test for a single sample
    
    Args:
        sample_id: Test sample ID (1-50)
        output_dir: Output directory
    
    Returns:
        (sample_id, success, message, duration)
    """
    start_time = time.time()
    sample_id_str = f'{sample_id:03d}'
    scenario_file = scenario_label or f'50x10-gemini/sample_{sample_id_str}.jsonl'
    scenario_path = scenario_path_override or SCENARIOS_DIR / f'sample_{sample_id_str}.jsonl'
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if not scenario_path.exists():
        return (sample_id, False, f'Scenario file does not exist: {scenario_path}', 0)
    try:
        qwen3vl = init_qwen3vl_model()
        qwen_image = init_qwen_image_model()
        turns_data = []
        with open(scenario_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    turns_data.append(json.loads(line))
        if not turns_data:
            return (sample_id, False, 'Scenario file is empty', 0)
        sample_output_dir = output_dir / f'sample_{sample_id_str}'
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        conversation_results = []
        conversation_history = []
        all_generated_images = []
        all_generated_turns = []
        print(f'\n[Sample {sample_id_str}] Starting KV Cache baseline ({len(turns_data)} turns)...')
        for turn_data in turns_data:
            turn_idx = turn_data['turn']
            user_input = turn_data['input']
            meta = turn_data.get('meta', {})
            task_type = meta.get('task_type', 'generate')
            print(f'[Sample {sample_id_str}] Turn {turn_idx}/{len(turns_data)}: {task_type}')
            history_images = all_generated_images.copy()
            history_image_turns = all_generated_turns.copy()
            history_messages = conversation_history.copy()
            if task_type in ['understand', 'pure_chat']:
                selection = call_qwen3vl_for_understand_selection(qwen3vl=qwen3vl, history_messages=history_messages, history_images=history_images, history_image_turns=history_image_turns, user_input=user_input)
                selected_index = selection.get('selected_index', None)
                messages = []
                if history_messages:
                    messages.extend(history_messages)
                messages.append({'role': 'user', 'content': user_input})
                selected_images = []
                if isinstance(selected_index, list) and selected_index:
                    for idx in selected_index:
                        if 0 <= idx < len(history_images):
                            selected_images.append(history_images[idx])
                print(f'[Sample {sample_id_str}] Qwen3VL infer start (history turns: {len(history_messages)}, images: {len(selected_images)})', flush=True)
                result_text = call_qwen3vl(qwen3vl, messages, selected_images if selected_images else None)
                full_text = result_text or ''
                print(f'[Sample {sample_id_str}] Qwen3VL infer done, output:\n{full_text}', flush=True)
                agent_output = {'text': result_text, 'image_path': None, 'qwen3vl_selected_index': selected_index}
                conversation_history.append({'role': 'user', 'content': user_input})
                conversation_history.append({'role': 'assistant', 'content': result_text})
            else:
                selection = call_qwen3vl_for_image_selection(qwen3vl=qwen3vl, history_messages=history_messages, history_images=history_images, history_image_turns=history_image_turns, user_input=user_input, enable_refine=enable_refine)
                selected_index = selection.get('selected_index', None)
                refine_prompt = selection.get('refine_prompt', None)
                prompt_parts = []
                prompt_parts.append(f'User instruction: {user_input}')
                if enable_refine and refine_prompt:
                    prompt_parts.append(f'Refine instruction: {refine_prompt}')
                full_prompt = '\n'.join(prompt_parts)
                input_images_for_generation = []
                if isinstance(selected_index, list) and selected_index:
                    for idx in selected_index:
                        if 0 <= idx < len(all_generated_images):
                            input_images_for_generation.append(all_generated_images[idx])
                else:
                    input_images_for_generation = []
                generated_image = call_qwen_image(qwen_image, full_prompt, input_images_for_generation, task_type)
                if generated_image:
                    image_filename = f'{turn_idx:02d}.png'
                    image_path = sample_output_dir / image_filename
                    generated_image.save(str(image_path))
                    all_generated_images.append(str(image_path))
                    all_generated_turns.append(turn_idx)
                    agent_output = {'text': full_prompt, 'image_path': str(image_path), 'qwen3vl_selected_index': selected_index, 'qwen3vl_refine_prompt': refine_prompt}
                else:
                    agent_output = {'text': full_prompt, 'image_path': None, 'qwen3vl_selected_index': selected_index, 'qwen3vl_refine_prompt': refine_prompt}
                conversation_history.append({'role': 'user', 'content': user_input})
                conversation_history.append({'role': 'assistant', 'content': f'Generated image: {image_filename}' if generated_image else 'Generation failed'})
            conversation_results.append(TurnResult(turn=turn_idx, user_input=user_input, task_type=task_type, agent_output=agent_output, all_history_images=history_images.copy(), conversation_history=history_messages.copy()))
        output_file = output_dir / f'sample_{sample_id_str}_kv_cache_baseline.json'
        output_data = {'sample_id': sample_id, 'scenario_file': scenario_file, 'timestamp': datetime.now().isoformat(), 'method': 'kv_cache_baseline', 'description': 'Baseline comparison using full conversation history and all images as input', 'turns': [{'turn': turn.turn, 'user_input': turn.user_input, 'task_type': turn.task_type, 'agent_output': turn.agent_output, 'history_images_count': len(turn.all_history_images), 'history_messages_count': len(turn.conversation_history)} for turn in conversation_results]}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        duration = time.time() - start_time
        return (sample_id, True, f'Success, result saved to: {output_file.name}', duration)
    except Exception as e:
        duration = time.time() - start_time
        import traceback
        error_msg = f'Exception: {str(e)}\n{traceback.format_exc()}'
        return (sample_id, False, error_msg[:200], duration)

def main():
    parser = argparse.ArgumentParser(description='KV Cache baseline comparison script (using full conversation history and all images)', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  # Run single sample\n  python run_kv_cache_baseline.py --scenario 50x10-gemini/sample_001.jsonl\n  \n  # Run multiple samples (parallel)\n  python run_kv_cache_baseline.py --start 1 --end 10 --max-workers 4\n  \n  # Specify output directory\n  python run_kv_cache_baseline.py --start 1 --end 5 --output-dir my_baseline_results\n        ')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scenario', type=str, help='Single scenario file (e.g.: 50x10-gemini/sample_001.jsonl)')
    group.add_argument('--start', type=int, help='Start sample ID (used with --end)')
    parser.add_argument('--end', type=int, help='End sample ID (used with --start)')
    parser.add_argument('--max-workers', type=int, default=1, help='Max parallel workers (default: 1, serial)')
    parser.add_argument('--enable-refine', action='store_true', help='Enable Qwen3VL refine instruction for edit/generate tasks')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: kv_cache_baseline_results/)')
    args = parser.parse_args()
    scenario_path_override = None
    scenario_label = None
    if args.scenario:
        scenario_path_override = Path(args.scenario)
        if not scenario_path_override.is_absolute():
            candidate = (BENCHMARK_DIR / scenario_path_override).resolve()
            if candidate.exists():
                scenario_path_override = candidate
            else:
                scenario_path_override = (SCENARIOS_DIR.parent / scenario_path_override).resolve()
        scenario_label = args.scenario
        scenario_name = scenario_path_override.stem
        if scenario_name.startswith('sample_'):
            sample_id = int(scenario_name.split('_')[1])
            sample_ids = [sample_id]
        else:
            print(f'Error: Failed to extract sample_id from scenario filename: {args.scenario}')
            sys.exit(1)
    else:
        if args.end is None:
            parser.error('--end must be used with --start')
        sample_ids = list(range(args.start, args.end + 1))
    valid_samples = []
    if scenario_path_override is not None:
        if not scenario_path_override.exists():
            print(f'❌ Scenario file does not exist: {scenario_path_override}')
            sys.exit(1)
        valid_samples = sample_ids
    else:
        for sample_id in sample_ids:
            sample_id_str = f'{sample_id:03d}'
            scenario_path = SCENARIOS_DIR / f'sample_{sample_id_str}.jsonl'
            if not scenario_path.exists():
                print(f'⚠ Skip sample_{sample_id_str}: Scenario file does not exist')
                continue
            valid_samples.append(sample_id)
    if not valid_samples:
        print('❌ No valid test samples to run')
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 80)
    print('KV Cache baseline comparison test')
    print('=' * 80)
    print(f'Test samples: {len(valid_samples)} (sample_{min(valid_samples):03d} to sample_{max(valid_samples):03d})')
    print(f'Parallel workers: {args.max_workers}')
    print(f'Output directory: {output_dir}')
    print('=' * 80)
    print()
    print('Pre-initializing models...')
    init_qwen3vl_model()
    init_qwen_image_model()
    print()
    start_time = time.time()
    results = []
    if args.max_workers == 1:
        for sample_id in valid_samples:
            result = run_single_baseline(sample_id, output_dir, scenario_path_override=scenario_path_override, scenario_label=scenario_label, enable_refine=args.enable_refine)
            results.append(result)
            sample_id, success, message, duration = result
            status = '✅' if success else '❌'
            print(f'{status} [{len(results)}/{len(valid_samples)}] Sample {sample_id:03d}: {message} ({duration:.1f}s)')
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_sample = {executor.submit(run_single_baseline, sample_id, output_dir, scenario_path_override=scenario_path_override, scenario_label=scenario_label, enable_refine=args.enable_refine): sample_id for sample_id in valid_samples}
            for future in as_completed(future_to_sample):
                sample_id = future_to_sample[future]
                try:
                    result = future.result()
                    results.append(result)
                    sample_id, success, message, duration = result
                    status = '✅' if success else '❌'
                    print(f'{status} [{len(results)}/{len(valid_samples)}] Sample {sample_id:03d}: {message} ({duration:.1f}s)')
                except Exception as e:
                    results.append((sample_id, False, f'Exception: {str(e)}', 0))
                    print(f'❌ [{len(results)}/{len(valid_samples)}] Sample {sample_id:03d}: Exception - {str(e)}')
    total_duration = time.time() - start_time
    print()
    print('=' * 80)
    print('Baseline test summary')
    print('=' * 80)
    print(f'Total tests: {len(valid_samples)}')
    successful = sum((1 for r in results if r[1]))
    print(f'Success: {successful}')
    print(f'Failed: {len(valid_samples) - successful}')
    print(f'Total elapsed: {total_duration:.1f} sec ({total_duration / 60:.1f} min)')
    print(f'Average per sample: {total_duration / len(valid_samples):.1f} sec')
    print()
    print('Generated result files:')
    for sample_id, success, message, duration in results:
        if success:
            sample_id_str = f'{sample_id:03d}'
            output_file = output_dir / f'sample_{sample_id_str}_kv_cache_baseline.json'
            if output_file.exists():
                print(f'  - Sample {sample_id_str}: {output_file.name}')
    print('=' * 80)
if __name__ == '__main__':
    main()
