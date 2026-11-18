"""
KV Cache 评测脚本（trained_0204）
- 仅替换记忆生成(VLM A)与检索(VLM B)逻辑
- 其余流程（Qwen3VL 理解/ Qwen-image 生成）与 baseline 保持一致
- 使用 448x448 缩略图作为 Qwen3VL 的图像输入
"""
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent / 'ms-swift'
sys.path.insert(0, str(PROJECT_ROOT))
import run_kv_cache_baseline as base
from multimodal_agent_service import VLMModel
os.environ.setdefault('QWEN3VL_GPUS', '2')
os.environ.setdefault('QWEN_IMAGE_GPUS', '3,4,5,6,7')
os.environ.setdefault('VLM_A_GPUS_0204', '0')
os.environ.setdefault('VLM_B_GPUS_0204', '1')
os.environ['QWEN3VL_MODEL_PATH'] = 'Qwen/Qwen3-VL-8B-Instruct'

def _thumb_path(image_path: str, size: int=448) -> Path:
    """为图片生成稳定的缩略图路径（不覆盖原图）"""
    cache_dir = Path('/tmp/kv_cache_thumbs_448')
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = Path(image_path).stat()
    key = f'{image_path}|{stat.st_mtime}|{stat.st_size}|{size}'
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    return cache_dir / f'thumb_{digest}_{size}.png'

def align_image_to_model_thumb(image_path: str) -> str:
    """生成 448x448 缩略图并返回其路径"""
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
        print(f'警告：缩略图生成失败 {e}')
        return image_path

class MemoryBank:
    """纯文本记忆库（使用 rank 作为 id）"""

    def __init__(self):
        self._memories: List[Dict] = []
        self._rank_to_image: List[str] = []

    def add(self, memory: Dict, image_path: str) -> int:
        rank = len(self._memories)
        memory = memory.copy()
        memory['id'] = rank
        self._memories.append(memory)
        self._rank_to_image.append(image_path)
        return rank

    def all(self) -> List[Dict]:
        return self._memories.copy()

    def get_image_by_rank(self, rank: int) -> Optional[str]:
        if 0 <= rank < len(self._rank_to_image):
            return self._rank_to_image[rank]
        return None

    def size(self) -> int:
        return len(self._memories)

class VLMASummarizer:
    """VLM A：图像摘要生成器（prompt 与训练一致）"""

    def __init__(self, checkpoint_path: str, gpu_ids: Optional[List[int]]=None):
        self.model = VLMModel(checkpoint_path, 'VLM A (0204)', gpu_ids=gpu_ids)
        self.user_prompt = '<image>请为这张图片生成摘要信息，包括标题、详细描述和diff（若为编辑图，描述新图相对参考图的特点；否则为空字符串）。'

    def summarize(self, image_path: str) -> Dict:
        messages = [{'role': 'user', 'content': self.user_prompt}]
        response = self.model.infer(messages=messages, images=[image_path], max_new_tokens=2048, temperature=0.0)
        memory = self._parse_json(response)
        if not isinstance(memory, dict):
            memory = {'title': '未命名图像', 'description': response, 'diff': ''}
        memory.setdefault('title', '')
        memory.setdefault('description', '')
        memory.setdefault('diff', '')
        return memory

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict]:
        try:
            return json.loads(text)
        except Exception:
            pass
        import re
        match = re.search('\\{.*\\}', text, re.DOTALL)
        if not match:
            return None
        raw = match.group()
        try:
            return json.loads(raw)
        except Exception:
            return None

class VLMBRetriever:
    """VLM B：基于记忆库检索图像 rank（prompt 与训练一致）"""

    def __init__(self, checkpoint_path: str, gpu_ids: Optional[List[int]]=None):
        self.model = VLMModel(checkpoint_path, 'VLM B (0204)', gpu_ids=gpu_ids)

    def retrieve(self, memory_bank: MemoryBank, user_query: str) -> List[int]:
        memory_texts = []
        for mem in memory_bank.all():
            mem_id = mem.get('id', '')
            title = mem.get('title', '')
            description = mem.get('description', '')
            diff = mem.get('diff', '')
            mem_desc = f'记忆{mem_id}: {title}'
            if description:
                mem_desc += f' - {description}'
            if diff:
                mem_desc += f' | diff: {diff}'
            memory_texts.append(mem_desc)
        if memory_texts:
            memory_context = '\n'.join(memory_texts)
            user_content = f'基于以下记忆库信息：\n{memory_context}\n\n用户查询：{user_query}\n\n请根据用户查询和记忆库信息，选择最相关的图像ID。'
        else:
            user_content = f'用户查询：{user_query}\n\n请选择最相关的图像ID。'
        messages = [{'role': 'user', 'content': user_content}]
        response = self.model.infer(messages=messages, images=None, max_new_tokens=128, temperature=0.0)
        return self._parse_ids(response)

    @staticmethod
    def _parse_ids(text: str) -> List[int]:
        lower = text.lower()
        if 'null' in lower:
            return []
        import re
        nums = re.findall('\\d+', text)
        return [int(n) for n in nums] if nums else []

@dataclass
class TurnResult:
    turn: int
    user_input: str
    task_type: str
    agent_output: Dict
    all_history_images: List[str]
    conversation_history: List[Dict]
    retrieved_ranks: Optional[List[int]] = None

def _parse_gpu_list(env_name: str) -> Optional[List[int]]:
    val = os.environ.get(env_name)
    if not val:
        return None
    return [int(x.strip()) for x in val.split(',') if x.strip()]

def _select_images_by_ranks(memory_bank: MemoryBank, ranks: Optional[List[int]]) -> List[str]:
    if not ranks:
        return []
    selected = []
    for r in ranks:
        img = memory_bank.get_image_by_rank(r)
        if img:
            selected.append(img)
    return selected

def run_single_trained(sample_id: int, output_dir: Path=None, scenario_path_override: Optional[Path]=None, scenario_label: Optional[str]=None, vlm_a: Optional[VLMASummarizer]=None, vlm_b: Optional[VLMBRetriever]=None) -> Tuple[int, bool, str, float]:
    start_time = time.time()
    sample_id_str = f'{sample_id:03d}'
    scenario_file = scenario_label or f'50x20-gemini-hard/sample_{sample_id_str}.jsonl'
    scenario_path = scenario_path_override or base.SCENARIOS_DIR / f'sample_{sample_id_str}.jsonl'
    if output_dir is None:
        output_dir = base.OUTPUT_DIR
    if not scenario_path.exists():
        return (sample_id, False, f'场景文件不存在: {scenario_path}', 0)
    try:
        qwen3vl = base.init_qwen3vl_model()
        qwen_image = base.init_qwen_image_model()
        if vlm_a is None or vlm_b is None:
            vlm_a_path = os.environ.get('VLM_A_PATH_0204', './checkpoints/memory_writer')
            vlm_b_path = os.environ.get('VLM_B_PATH_0204', './checkpoints/retriever')
            vlm_a_gpus = _parse_gpu_list('VLM_A_GPUS_0204')
            vlm_b_gpus = _parse_gpu_list('VLM_B_GPUS_0204')
            vlm_a = vlm_a or VLMASummarizer(vlm_a_path, vlm_a_gpus)
            vlm_b = vlm_b or VLMBRetriever(vlm_b_path, vlm_b_gpus)
        turns_data = []
        with open(scenario_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    turns_data.append(json.loads(line))
        if not turns_data:
            return (sample_id, False, '场景文件为空', 0)
        sample_output_dir = output_dir / f'sample_{sample_id_str}'
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        conversation_results = []
        conversation_history = []
        all_generated_images = []
        all_generated_turns = []
        memory_bank = MemoryBank()
        print(f'\n[Sample {sample_id_str}] Starting trained_0204 ({len(turns_data)} turns)...')
        for turn_data in turns_data:
            turn_idx = turn_data['turn']
            user_input = turn_data['input']
            meta = turn_data.get('meta', {})
            task_type = meta.get('task_type', 'generate')
            print(f'[Sample {sample_id_str}] Turn {turn_idx}/{len(turns_data)}: {task_type}')
            history_images = all_generated_images.copy()
            history_image_turns = all_generated_turns.copy()
            history_messages = conversation_history.copy()
            retrieved_ranks = vlm_b.retrieve(memory_bank, user_input) if memory_bank.size() > 0 else []
            selected_images = _select_images_by_ranks(memory_bank, retrieved_ranks)
            if task_type in ['understand', 'pure_chat']:
                messages = []
                if history_messages:
                    messages.extend(history_messages)
                messages.append({'role': 'user', 'content': user_input})
                print(f'[Sample {sample_id_str}] Qwen3VL infer start (history turns: {len(history_messages)}, images: {len(selected_images)})', flush=True)
                result_text = base.call_qwen3vl(qwen3vl, messages, selected_images if selected_images else None)
                full_text = result_text or ''
                print(f'[Sample {sample_id_str}] Qwen3VL infer done, output:\n{full_text}', flush=True)
                agent_output = {'text': result_text, 'image_path': None, 'vlm_b_selected_ranks': retrieved_ranks}
                conversation_history.append({'role': 'user', 'content': user_input})
                conversation_history.append({'role': 'assistant', 'content': result_text})
            else:
                prompt_parts = [f'用户指令: {user_input}']
                full_prompt = '\n'.join(prompt_parts)
                input_images_for_generation = selected_images if selected_images else []
                generated_image = base.call_qwen_image(qwen_image, full_prompt, input_images_for_generation, task_type)
                if generated_image:
                    image_filename = f'{turn_idx:02d}.png'
                    image_path = sample_output_dir / image_filename
                    generated_image.save(str(image_path))
                    all_generated_images.append(str(image_path))
                    all_generated_turns.append(turn_idx)
                    memory = vlm_a.summarize(str(image_path))
                    memory_bank.add(memory, str(image_path))
                    agent_output = {'text': full_prompt, 'image_path': str(image_path), 'vlm_b_selected_ranks': retrieved_ranks, 'vlm_a_memory': memory}
                else:
                    agent_output = {'text': full_prompt, 'image_path': None, 'vlm_b_selected_ranks': retrieved_ranks, 'vlm_a_memory': None}
                conversation_history.append({'role': 'user', 'content': user_input})
                conversation_history.append({'role': 'assistant', 'content': f'已生成图片: {image_filename}' if generated_image else '生成失败'})
            conversation_results.append(TurnResult(turn=turn_idx, user_input=user_input, task_type=task_type, agent_output=agent_output, all_history_images=history_images.copy(), conversation_history=history_messages.copy(), retrieved_ranks=retrieved_ranks))
        output_file = output_dir / f'sample_{sample_id_str}_kv_cache_trained_0204.json'
        output_data = {'sample_id': sample_id, 'scenario_file': scenario_file, 'timestamp': datetime.now().isoformat(), 'method': 'kv_cache_trained_0204', 'description': '仅替换 VLM A 记忆生成 & VLM B 检索，其他流程与 baseline 一致', 'turns': [{'turn': turn.turn, 'user_input': turn.user_input, 'task_type': turn.task_type, 'agent_output': turn.agent_output, 'history_images_count': len(turn.all_history_images), 'history_messages_count': len(turn.conversation_history), 'vlm_b_selected_ranks': turn.retrieved_ranks} for turn in conversation_results]}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        duration = time.time() - start_time
        return (sample_id, True, f'成功，结果保存到: {output_file.name}', duration)
    except Exception as e:
        duration = time.time() - start_time
        import traceback
        error_msg = f'异常: {str(e)}\n{traceback.format_exc()}'
        return (sample_id, False, error_msg[:200], duration)

def main():
    parser = argparse.ArgumentParser(description='KV Cache trained_0204 评测脚本（使用 VLM A/B 替换记忆与检索）', formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scenario', type=str, help='单个场景文件（如: 50x20-gemini-hard/sample_001.jsonl）')
    group.add_argument('--start', type=int, help='起始 sample ID（与 --end 一起使用）')
    parser.add_argument('--end', type=int, help='结束 sample ID（与 --start 一起使用）')
    parser.add_argument('--max-workers', type=int, default=1, help='最大并行数（默认: 1，串行执行）')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    args = parser.parse_args()
    _scenarios_dir = os.environ.get('KV_CACHE_SCENARIOS_DIR')
    base.SCENARIOS_DIR = Path(_scenarios_dir) if _scenarios_dir else base.BENCHMARK_DIR / 'scenarios' / '50x20-gemini-hard'
    _output_dir = os.environ.get('KV_CACHE_OUTPUT_DIR')
    base.OUTPUT_DIR = Path(_output_dir) if _output_dir else base.BENCHMARK_DIR / 'kv_cache_trained_0204_results_thumbs_448_hard'
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.align_image_to_model = align_image_to_model_thumb
    scenario_path_override = None
    scenario_label = None
    if args.scenario:
        scenario_path_override = Path(args.scenario)
        if not scenario_path_override.is_absolute():
            candidate = (BENCHMARK_DIR / scenario_path_override).resolve()
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
            print(f'Error: 无法从场景文件名提取 sample_id: {args.scenario}')
            sys.exit(1)
    else:
        if args.end is None:
            parser.error('--end 必须与 --start 一起使用')
        sample_ids = list(range(args.start, args.end + 1))
    valid_samples = []
    if scenario_path_override is not None:
        if not scenario_path_override.exists():
            print(f'❌ 场景文件不存在: {scenario_path_override}')
            sys.exit(1)
        valid_samples = sample_ids
    else:
        for sample_id in sample_ids:
            sample_id_str = f'{sample_id:03d}'
            scenario_path = base.SCENARIOS_DIR / f'sample_{sample_id_str}.jsonl'
            if not scenario_path.exists():
                print(f'⚠ 跳过 sample_{sample_id_str}: 场景文件不存在')
                continue
            valid_samples.append(sample_id)
    if not valid_samples:
        print('❌ 没有有效的测试样例可运行')
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else base.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 80)
    print('KV Cache trained_0204 评测')
    print('=' * 80)
    print(f'测试样例: {len(valid_samples)} 个 (sample_{min(valid_samples):03d} 到 sample_{max(valid_samples):03d})')
    print(f'并行数: {args.max_workers}')
    print(f'输出目录: {output_dir}')
    print('=' * 80)
    print()
    print('预初始化模型...')
    base.init_qwen3vl_model()
    base.init_qwen_image_model()
    vlm_a_path = os.environ.get('VLM_A_PATH_0204', './checkpoints/memory_writer')
    vlm_b_path = os.environ.get('VLM_B_PATH_0204', './checkpoints/retriever')
    vlm_a_gpus = _parse_gpu_list('VLM_A_GPUS_0204')
    vlm_b_gpus = _parse_gpu_list('VLM_B_GPUS_0204')
    vlm_a = VLMASummarizer(vlm_a_path, vlm_a_gpus)
    vlm_b = VLMBRetriever(vlm_b_path, vlm_b_gpus)
    print()
    start_time = time.time()
    results = []
    if args.max_workers == 1:
        for sample_id in valid_samples:
            result = run_single_trained(sample_id, output_dir, scenario_path_override=scenario_path_override, scenario_label=scenario_label, vlm_a=vlm_a, vlm_b=vlm_b)
            results.append(result)
            sample_id, success, message, duration = result
            status = '✅' if success else '❌'
            print(f'{status} [{len(results)}/{len(valid_samples)}] Sample {sample_id:03d}: {message} ({duration:.1f}s)')
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_sample = {executor.submit(run_single_trained, sample_id, output_dir, scenario_path_override=scenario_path_override, scenario_label=scenario_label, vlm_a=vlm_a, vlm_b=vlm_b): sample_id for sample_id in valid_samples}
            for future in as_completed(future_to_sample):
                sample_id = future_to_sample[future]
                try:
                    result = future.result()
                    results.append(result)
                    sample_id, success, message, duration = result
                    status = '✅' if success else '❌'
                    print(f'{status} [{len(results)}/{len(valid_samples)}] Sample {sample_id:03d}: {message} ({duration:.1f}s)')
                except Exception as e:
                    results.append((sample_id, False, f'异常: {str(e)}', 0))
                    print(f'❌ [{len(results)}/{len(valid_samples)}] Sample {sample_id:03d}: 异常 - {str(e)}')
    total_duration = time.time() - start_time
    print()
    print('=' * 80)
    print('评测总结')
    print('=' * 80)
    print(f'总测试数: {len(valid_samples)}')
    successful = sum((1 for r in results if r[1]))
    print(f'成功: {successful}')
    print(f'失败: {len(valid_samples) - successful}')
    print(f'总耗时: {total_duration:.1f} 秒 ({total_duration / 60:.1f} 分钟)')
    print(f'平均每个: {total_duration / len(valid_samples):.1f} 秒')
    print()
    print('生成的结果文件:')
    for sample_id, success, message, duration in results:
        if success:
            sample_id_str = f'{sample_id:03d}'
            output_file = output_dir / f'sample_{sample_id_str}_kv_cache_trained_0204.json'
            if output_file.exists():
                print(f'  - Sample {sample_id_str}: {output_file.name}')
    print('=' * 80)
if __name__ == '__main__':
    main()
