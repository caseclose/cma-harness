"""
KV Cache baseline comparison script (50x20-gemini-hard scenarios)
Use tags + description + thumbnails (448x448) for retrieval
- For each newly generated image or user-uploaded image, call Qwen3VL-8B to generate tags and description
- Retrieval uses tags + description + thumbnails
"""
import os
import sys
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import run_kv_cache_baseline as base

def _thumb_path(image_path: str, size: int=448) -> Path:
    """Generate a stable thumbnail path for an image (without overwriting the original)"""
    cache_dir = Path('/tmp/kv_cache_thumbs_448')
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = Path(image_path).stat()
    key = f'{image_path}|{stat.st_mtime}|{stat.st_size}|{size}'
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    return cache_dir / f'thumb_{digest}_{size}.png'

def align_image_to_model_thumb(image_path: str) -> str:
    """Generate a 448x448 thumbnail and return its path"""
    try:
        src = Path(image_path)
        if not src.exists():
            return image_path
        dst = _thumb_path(image_path, size=448)
        if dst.exists():
            return str(dst)
        with Image.open(src) as img:
            img = img.convert('RGB')
            if img.size != (448, 448):
                img = img.resize((448, 448), resample=Image.LANCZOS)
            img.save(dst)
        return str(dst)
    except Exception as e:
        print(f'Warning: failed to generate thumbnail {e}')
        return image_path

class MemoryBank:
    """Memory bank storing images and their tags and description"""

    def __init__(self):
        self._memories: List[Dict] = []
        self._rank_to_image: List[str] = []

    def add(self, memory: Dict, image_path: str) -> int:
        """Add memory and return rank (index)"""
        rank = len(self._memories)
        memory = memory.copy()
        memory['id'] = rank
        memory['rank'] = rank
        self._memories.append(memory)
        self._rank_to_image.append(image_path)
        return rank

    def all(self) -> List[Dict]:
        """Get all memories"""
        return self._memories.copy()

    def get_image_by_rank(self, rank: int) -> Optional[str]:
        """Get image path by rank"""
        if 0 <= rank < len(self._rank_to_image):
            return self._rank_to_image[rank]
        return None

    def size(self) -> int:
        """Get memory count"""
        return len(self._memories)

def generate_tags_and_description(qwen3vl, image_path: str) -> Dict:
    """
    Call Qwen3VL-8B to generate tags and description for an image
    
    Returns:
        Dict with keys: tags (list), description (str)
    """
    user_message = '<image>Please analyze this image in detail and generate complete image features + description in English.Output must be valid JSON with the following fields: tags (tag list, at least 5 tags, array format), description (detailed description, at least 100 words). Please ensure the output is valid JSON.'
    messages = [{'role': 'user', 'content': user_message}]
    print(f'[Tags/Description] start generating tags and description for image: {image_path}')
    result_text = base.call_qwen3vl(qwen3vl, messages, [image_path])
    print(f'[Tags/Description] model output:\n{result_text}\n')
    try:
        memory = json.loads(result_text)
    except json.JSONDecodeError:
        match = re.search('\\{.*\\}', result_text, re.DOTALL)
        if match:
            try:
                memory = json.loads(match.group())
            except json.JSONDecodeError:
                memory = {}
        else:
            memory = {}
    tags = memory.get('tags', [])
    if not isinstance(tags, list):
        tags = []
    description = memory.get('description', '')
    if not isinstance(description, str):
        description = str(description) if description else ''
    result = {'tags': tags, 'description': description}
    print(f'[Tags/Description] parsed result: tags={len(tags)} tags, description length={len(description)}')
    return result
VALID_MEMORY_MODES = ('tags_desc', 'tags_only', 'desc_only', 'thumbs_only')

def result_json_suffix(memory_mode: str) -> str:
    if memory_mode == 'tags_desc':
        return 'kv_cache_baseline_tags_desc'
    return f'kv_cache_baseline_mem_{memory_mode}'

def call_qwen3vl_for_retrieval_with_tags_desc(qwen3vl, memory_bank: MemoryBank, user_input: str, memory_mode: str='tags_desc') -> List[int]:
    """
    Episodic retrieval with 448px thumbnails + optional text fields (ablation via memory_mode).

    memory_mode:
      - tags_desc: tags + description + thumbs (default; same as before)
      - tags_only: tags + thumbs
      - desc_only: description + thumbs
      - thumbs_only: thumbs only (ranks listed; no tag/description text in memory-bank)
    """
    if memory_mode not in VALID_MEMORY_MODES:
        raise ValueError(f'memory_mode must be one of {VALID_MEMORY_MODES}, got {memory_mode!r}')
    all_memories = memory_bank.all()
    if not all_memories:
        return []
    memory_texts = []
    image_paths = []
    rank_to_thumb_idx = {}
    thumb_idx = 0
    for mem in all_memories:
        mem_id = mem.get('id', '')
        mem_rank = mem.get('rank', mem_id)
        tags = mem.get('tags', [])
        description = mem.get('description', '')
        if isinstance(tags, list):
            tags_text = ', '.join([str(t) for t in tags if t is not None])
        else:
            tags_text = str(tags) if tags else ''
        mem_desc = f'memory[{mem_rank}]:'
        if memory_mode == 'thumbs_only':
            pass
        elif memory_mode == 'tags_only':
            if tags_text:
                mem_desc += f' tags: {tags_text}'
        elif memory_mode == 'desc_only':
            if description:
                mem_desc += f' description: {description}'
        else:
            if tags_text:
                mem_desc += f' tags: {tags_text}'
            if description:
                mem_desc += f' description: {description}'
        memory_texts.append(mem_desc)
        image_path = memory_bank.get_image_by_rank(mem_rank)
        if image_path:
            image_paths.append(image_path)
            rank_to_thumb_idx[mem_rank] = thumb_idx
            thumb_idx += 1
    if memory_mode == 'thumbs_only':
        memory_context = '(No textual memory metadata; each line is only a rank label. Select using the thumbnail images and the user query.)\n' + ('\n'.join(memory_texts) if memory_texts else '(no memories)')
    else:
        memory_context = '\n'.join(memory_texts) if memory_texts else '(currently no history memories)'
    lead = {'tags_desc': 'Based on the user query and memory-bank info (tags and description) and the thumbnails,', 'tags_only': 'Based on the user query, tag-only memory-bank text, and the thumbnails,', 'desc_only': 'Based on the user query, description-only memory-bank text, and the thumbnails,', 'thumbs_only': 'Based on the user query and the thumbnail images alone (ignore rank lines as semantic text),'}[memory_mode]
    instruction = f'You are a strict multimodal assistant. {lead} select the most relevant image indices from the provided historical image list (can be multiple).If no image is needed, set selected_index to null.\n\nYou must retrieve by index (only output rank indices, not filenames or paths).\n\nHistorical image index list:\n'
    for mem in all_memories:
        mem_rank = mem.get('rank', mem.get('id', ''))
        instruction += f'- rank: {mem_rank}\n'
    instruction += f'\nOutput must be JSON format (selected_index should be rank values, e.g. [0, 2] means rank 0 and rank 2):\n{{\n  "selected_index": [0, 2]\n}}\nDo not output any extra text.\n\nMemory-bank info:\n{memory_context}\n\nuser query: {user_input}'
    messages = [{'role': 'user', 'content': instruction}]
    thumb_paths = [align_image_to_model_thumb(img) for img in image_paths]
    print(f'[Retrieval] Start retrieval (memory count: {len(all_memories)}, thumbnail count: {len(thumb_paths)})')
    print(f'[Retrieval] User query: {user_input}')
    print(f'[Retrieval] Memory-bank info:\n{memory_context}\n')
    result_text = base.call_qwen3vl(qwen3vl, messages, thumb_paths if thumb_paths else None)
    print(f'[Retrieval] Model output:\n{result_text}\n')
    try:
        parsed = json.loads(result_text)
        selected_index = parsed.get('selected_index', None)
        if isinstance(selected_index, list):
            selected_index = [i for i in selected_index if isinstance(i, int)]
        elif isinstance(selected_index, int):
            selected_index = [selected_index]
        else:
            selected_index = None
        return selected_index if selected_index else []
    except Exception:
        numbers = re.findall('\\d+', result_text)
        if numbers:
            return [int(n) for n in numbers]
        return []

@dataclass
class TurnResult:
    """Single-turn conversation result"""
    turn: int
    user_input: str
    task_type: str
    agent_output: Dict
    all_history_images: List[str]
    conversation_history: List[Dict]
    retrieved_ranks: Optional[List[int]] = None

def run_single_baseline_tags_desc(sample_id: int, output_dir: Path=None, scenario_path_override: Optional[Path]=None, scenario_label: Optional[str]=None, memory_mode: str='tags_desc') -> Tuple[int, bool, str, float]:
    """
    Run KV Cache baseline for a single sample (thumbnail retrieval + optional text memory ablation).
    """
    import time
    start_time = time.time()
    sample_id_str = f'{sample_id:03d}'
    scenario_file = scenario_label or f'50x20-gemini-hard/sample_{sample_id_str}.jsonl'
    scenario_path = scenario_path_override or base.SCENARIOS_DIR / f'sample_{sample_id_str}.jsonl'
    if output_dir is None:
        output_dir = base.OUTPUT_DIR
    if not scenario_path.exists():
        return (sample_id, False, f'Scenario file does not exist: {scenario_path}', 0)
    try:
        qwen3vl = base.init_qwen3vl_model()
        qwen_image = base.init_qwen_image_model()
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
        memory_bank = MemoryBank()
        print(f'\n[Sample {sample_id_str}] KV Cache baseline memory_mode={memory_mode} ({len(turns_data)} turns)...')
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
                retrieved_ranks = call_qwen3vl_for_retrieval_with_tags_desc(qwen3vl=qwen3vl, memory_bank=memory_bank, user_input=user_input, memory_mode=memory_mode)
                selected_images = []
                if retrieved_ranks:
                    for rank in retrieved_ranks:
                        img_path = memory_bank.get_image_by_rank(rank)
                        if img_path and img_path in history_images:
                            selected_images.append(img_path)
                messages = []
                if history_messages:
                    messages.extend(history_messages)
                messages.append({'role': 'user', 'content': user_input})
                print(f'[Sample {sample_id_str}] Qwen3VL infer start (history turns: {len(history_messages)}, retrieved images: {len(selected_images)})', flush=True)
                result_text = base.call_qwen3vl(qwen3vl, messages, selected_images if selected_images else None)
                full_text = result_text or ''
                print(f'[Sample {sample_id_str}] Qwen3VL infer done, output:\n{full_text}', flush=True)
                agent_output = {'text': result_text, 'image_path': None, 'retrieved_ranks': retrieved_ranks}
                conversation_history.append({'role': 'user', 'content': user_input})
                conversation_history.append({'role': 'assistant', 'content': result_text})
            else:
                retrieved_ranks = call_qwen3vl_for_retrieval_with_tags_desc(qwen3vl=qwen3vl, memory_bank=memory_bank, user_input=user_input, memory_mode=memory_mode)
                input_images_for_generation = []
                if retrieved_ranks:
                    for rank in retrieved_ranks:
                        img_path = memory_bank.get_image_by_rank(rank)
                        if img_path and img_path in history_images:
                            input_images_for_generation.append(img_path)
                prompt_parts = [f'User instruction: {user_input}']
                full_prompt = '\n'.join(prompt_parts)
                generated_image = base.call_qwen_image(qwen_image, full_prompt, input_images_for_generation, task_type)
                if generated_image:
                    image_filename = f'{turn_idx:02d}.png'
                    image_path = sample_output_dir / image_filename
                    generated_image.save(str(image_path))
                    all_generated_images.append(str(image_path))
                    all_generated_turns.append(turn_idx)
                    print(f'[Sample {sample_id_str}] start generating tags and description for generated image...')
                    tags_desc = generate_tags_and_description(qwen3vl, str(image_path))
                    memory = {'tags': tags_desc.get('tags', []), 'description': tags_desc.get('description', ''), 'image_path': str(image_path)}
                    memory_bank.add(memory, str(image_path))
                    agent_output = {'text': full_prompt, 'image_path': str(image_path), 'retrieved_ranks': retrieved_ranks, 'tags': tags_desc.get('tags', []), 'description': tags_desc.get('description', '')}
                else:
                    agent_output = {'text': full_prompt, 'image_path': None, 'retrieved_ranks': retrieved_ranks, 'tags': [], 'description': ''}
                conversation_history.append({'role': 'user', 'content': user_input})
                conversation_history.append({'role': 'assistant', 'content': f'Generated image: {image_filename}' if generated_image else 'Generation failed'})
            conversation_results.append(TurnResult(turn=turn_idx, user_input=user_input, task_type=task_type, agent_output=agent_output, all_history_images=history_images.copy(), conversation_history=history_messages.copy(), retrieved_ranks=retrieved_ranks if 'retrieved_ranks' in agent_output else None))
        from datetime import datetime
        suffix = result_json_suffix(memory_mode)
        output_file = output_dir / f'sample_{sample_id_str}_{suffix}.json'
        method_label = suffix
        desc_long = {'tags_desc': 'PAE writes tags+desc; CoRE retrieval uses tags + description + 448 thumbnails', 'tags_only': 'PAE writes tags+desc; CoRE retrieval uses tags + thumbnails only', 'desc_only': 'PAE writes tags+desc; CoRE retrieval uses description + thumbnails only', 'thumbs_only': 'PAE writes tags+desc; CoRE retrieval uses thumbnails only (no tag/desc text)'}.get(memory_mode, '')
        output_data = {'sample_id': sample_id, 'scenario_file': scenario_file, 'timestamp': datetime.now().isoformat(), 'memory_mode': memory_mode, 'method': method_label, 'description': desc_long, 'turns': [{'turn': turn.turn, 'user_input': turn.user_input, 'task_type': turn.task_type, 'agent_output': turn.agent_output, 'history_images_count': len(turn.all_history_images), 'history_messages_count': len(turn.conversation_history), 'retrieved_ranks': turn.retrieved_ranks} for turn in conversation_results]}
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
    import argparse
    parser = argparse.ArgumentParser(description='KV Cache baseline comparison script (retrieval with tags + description + thumbnails)', formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scenario', type=str, help='Single scenario file (e.g. 50x20-gemini-hard/sample_001.jsonl)')
    group.add_argument('--start', type=int, help='Start sample ID (used with --end)')
    parser.add_argument('--end', type=int, help='End sample ID (used with --start)')
    parser.add_argument('--max-workers', type=int, default=1, help='Maximum parallel workers (default: 1, serial execution)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: kv_cache_baseline_tags_desc_results/)')
    parser.add_argument('--memory-mode', type=str, choices=list(VALID_MEMORY_MODES), default='tags_desc', help='Fields visible to CoRE at retrieval time (PAE still writes full tags+desc). Use tags_only / desc_only / thumbs_only for memory ablations.')
    args = parser.parse_args()
    _scenarios_dir = os.environ.get('KV_CACHE_SCENARIOS_DIR')
    base.SCENARIOS_DIR = Path(_scenarios_dir) if _scenarios_dir else base.BENCHMARK_DIR / 'scenarios' / '50x20-gemini-hard'
    _output_dir = os.environ.get('KV_CACHE_OUTPUT_DIR')
    base.OUTPUT_DIR = Path(_output_dir) if _output_dir else base.BENCHMARK_DIR / 'kv_cache_baseline_tags_desc_results_thumbs_448_hard'
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.align_image_to_model = align_image_to_model_thumb
    scenario_path_override = None
    scenario_label = None
    if args.scenario:
        scenario_path_override = Path(args.scenario)
        if not scenario_path_override.is_absolute():
            candidate = (base.BENCHMARK_DIR / scenario_path_override).resolve()
            if candidate.exists():
                scenario_path_override = candidate
            else:
                scenario_path_override = (base.SCENARIOS_DIR.parent / scenario_path_override).resolve()
        scenario_label = args.scenario
        scenario_name = scenario_path_override.stem
        if scenario_name.startswith('sample_'):
            sample_id = int(scenario_name.split('_')[1])
            sample_ids = [sample_id]
        else:
            print(f'Error: failed to extract sample_id from scenario filename: {args.scenario}')
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
            scenario_path = base.SCENARIOS_DIR / f'sample_{sample_id_str}.jsonl'
            if not scenario_path.exists():
                print(f'⚠ Skip sample_{sample_id_str}: Scenario file does not exist')
                continue
            valid_samples.append(sample_id)
    if not valid_samples:
        print('❌ No valid test samples to run')
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else base.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 80)
    print(f'KV Cache baseline (memory_mode={args.memory_mode}, suffix={result_json_suffix(args.memory_mode)})')
    print('=' * 80)
    print(f'Test samples: {len(valid_samples)} (sample_{min(valid_samples):03d} to sample_{max(valid_samples):03d})')
    print(f'Parallel workers: {args.max_workers}')
    print(f'Output directory: {output_dir}')
    print('=' * 80)
    print()
    print('Pre-initializing models...')
    base.init_qwen3vl_model()
    base.init_qwen_image_model()
    print()
    import time
    start_time = time.time()
    results = []
    if args.max_workers == 1:
        for sample_id in valid_samples:
            result = run_single_baseline_tags_desc(sample_id, output_dir, scenario_path_override=scenario_path_override, scenario_label=scenario_label, memory_mode=args.memory_mode)
            results.append(result)
            sample_id, success, message, duration = result
            status = '✅' if success else '❌'
            print(f'{status} [{len(results)}/{len(valid_samples)}] Sample {sample_id:03d}: {message} ({duration:.1f}s)')
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_sample = {executor.submit(run_single_baseline_tags_desc, sample_id, output_dir, scenario_path_override=scenario_path_override, scenario_label=scenario_label, memory_mode=args.memory_mode): sample_id for sample_id in valid_samples}
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
    sfx = result_json_suffix(args.memory_mode)
    print('Generated result files:')
    for sample_id, success, message, duration in results:
        if success:
            sample_id_str = f'{sample_id:03d}'
            output_file = output_dir / f'sample_{sample_id_str}_{sfx}.json'
            if output_file.exists():
                print(f'  - Sample {sample_id_str}: {output_file.name}')
    print('=' * 80)
if __name__ == '__main__':
    main()
