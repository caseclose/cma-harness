"""
Build retrieval training datasets from benchmark scenarios + result files.

Outputs:
1) SFT dataset: messages(user/assistant) with target image IDs.
2) RL dataset: messages(user) + retrieval_target, filtered by error/hard rules.
"""
import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
try:
    from PIL import Image
except Exception:
    Image = None

def _normalize_indices(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str) and item.strip().lstrip('-').isdigit():
                out.append(int(item.strip()))
        return sorted(out)
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if text == '' or text.lower() in {'null', 'none'}:
            return []
        try:
            parsed = json.loads(text)
            return _normalize_indices(parsed)
        except Exception:
            nums = re.findall('-?\\d+', text)
            return sorted((int(n) for n in nums)) if nums else []
    return []

def _format_target_text(target: List[int]) -> str:
    return f'Image IDs: {json.dumps(target, ensure_ascii=False)}'

def _format_memory_line(rank: int, tags: List[str], desc: str, max_desc_chars: int) -> str:
    tags_text = ', '.join([str(t) for t in tags if t is not None])
    line = f'memory[{rank}]'
    if tags_text:
        line += f' tags: {tags_text}'
    if desc:
        if max_desc_chars > 0 and len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + '…'
        line += f' description: {desc}'
    return line

def _build_user_prompt(memory_bank: List[Dict], user_query: str, max_desc_chars: int) -> str:
    if memory_bank:
        candidates = '\n'.join([f"- rank {mem['rank']}: <image>" for mem in memory_bank])
        memory_text = '\n'.join((_format_memory_line(mem['rank'], mem.get('tags', []), mem.get('description', ''), max_desc_chars=max_desc_chars) for mem in memory_bank))
    else:
        candidates = '(currently no candidate images)'
        memory_text = '(currently no historical memory)'
    return f'Candidate image list (same order as the images field):\n{candidates}\n\nMemory-bank information (including tags + description):\n{memory_text}\n\nUser query: {user_query}\n\nBased on the user query and memory-bank information, select the most relevant image ID list.If no historical image needs to be referenced, output an empty list.'

@dataclass
class TurnRow:
    sample_id: int
    turn: int
    user_input: str
    prompt: str
    images: List[str]
    target_ranks: List[int]
    predicted_ranks: List[int]
    is_correct: bool
    task_type: str
    difficulty: str
    topic: str
    topic_switch: bool
    cross_topic_retrieval: bool
    long_term_memory: bool
    negative_sample: bool

def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows

def _load_scenario_index(scenario_path: Path) -> Dict[int, Dict]:
    turns = {}
    for row in _read_jsonl(scenario_path):
        turn = int(row['turn'])
        turns[turn] = row
    return turns

def _extract_predicted(turn_row: Dict) -> List[int]:
    val = turn_row.get('retrieved_ranks') or turn_row.get('vlm_b_selected_ranks') or turn_row.get('agent_output', {}).get('retrieved_ranks') or turn_row.get('agent_output', {}).get('vlm_b_selected_ranks')
    return _normalize_indices(val)

def _thumb_path(image_path: str, size: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        st = os.stat(image_path)
        key = f'{image_path}|{st.st_mtime}|{st.st_size}|{size}'
    except Exception:
        key = f'{image_path}|na|na|{size}'
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    return cache_dir / f'thumb_{digest}_{size}.png'

def align_image_to_thumb(image_path: str, size: int, cache_dir: Path) -> str:
    """
    Create a size×size thumbnail and return its path.
    Falls back to original path if PIL is unavailable or conversion fails.
    """
    if not image_path:
        return image_path
    if Image is None:
        return image_path
    src = Path(image_path)
    if not src.exists():
        return image_path
    dst = _thumb_path(str(src), size=size, cache_dir=cache_dir)
    if dst.exists():
        return str(dst)
    try:
        with Image.open(src) as img:
            img = img.convert('RGB')
            if img.size != (size, size):
                img = img.resize((size, size), resample=Image.LANCZOS)
            img.save(dst)
        return str(dst)
    except Exception:
        return image_path

def _update_memory_bank(memory_bank: List[Dict], turn_row: Dict) -> None:
    agent_output = turn_row.get('agent_output', {}) or {}
    image_path = agent_output.get('image_path')
    if not image_path:
        return
    tags = agent_output.get('tags', [])
    if not isinstance(tags, list):
        tags = []
    description = agent_output.get('description', '')
    if description is None:
        description = ''
    memory_bank.append({'rank': len(memory_bank), 'tags': tags, 'description': str(description), 'image_path': image_path})

def _update_memory_bank_mm(memory_bank: List[Dict], turn_row: Dict, thumb_size: int, thumb_cache_dir: Path) -> None:
    """Like _update_memory_bank, but caches thumb path once per new memory."""
    agent_output = turn_row.get('agent_output', {}) or {}
    image_path = agent_output.get('image_path')
    if not image_path:
        return
    tags = agent_output.get('tags', [])
    if not isinstance(tags, list):
        tags = []
    description = agent_output.get('description', '')
    if description is None:
        description = ''
    thumb_path = align_image_to_thumb(str(image_path), size=thumb_size, cache_dir=thumb_cache_dir)
    memory_bank.append({'rank': len(memory_bank), 'tags': tags, 'description': str(description), 'image_path': image_path, 'thumb_path': thumb_path})

def _memory_bank_images(memory_bank: List[Dict]) -> List[str]:
    sorted_mem = sorted(memory_bank, key=lambda x: int(x.get('rank', 0)))
    return [str(mem.get('thumb_path') or mem.get('image_path')) for mem in sorted_mem if mem.get('image_path')]

def _build_rows_for_sample(sample_id: int, scenario_path: Path, result_path: Path, allow_missing_result: bool, thumb_size: int, thumb_cache_dir: Path, max_desc_chars: int) -> List[TurnRow]:
    if not result_path.exists():
        if allow_missing_result:
            return []
        raise FileNotFoundError(f'Missing result file: {result_path}')
    scenario_turns = _load_scenario_index(scenario_path)
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    result_turns = sorted(result.get('turns', []), key=lambda x: int(x.get('turn', 0)))
    out: List[TurnRow] = []
    memory_bank: List[Dict] = []
    for turn_row in result_turns:
        turn = int(turn_row.get('turn'))
        sc = scenario_turns.get(turn)
        if not sc:
            _update_memory_bank_mm(memory_bank, turn_row, thumb_size=thumb_size, thumb_cache_dir=thumb_cache_dir)
            continue
        meta = sc.get('meta', {}) or {}
        target = _normalize_indices(meta.get('retrieval_target'))
        predicted = _extract_predicted(turn_row)
        user_input = turn_row.get('user_input') or sc.get('input', '')
        images = _memory_bank_images(memory_bank)
        prompt = _build_user_prompt(memory_bank, user_input, max_desc_chars=max_desc_chars)
        row = TurnRow(sample_id=sample_id, turn=turn, user_input=user_input, prompt=prompt, images=images, target_ranks=target, predicted_ranks=predicted, is_correct=target == predicted, task_type=str(meta.get('task_type', '')), difficulty=str(meta.get('difficulty', '')), topic=str(meta.get('topic', '')), topic_switch=bool(meta.get('topic_switch', False)), cross_topic_retrieval=bool(meta.get('cross_topic_retrieval', False)), long_term_memory=bool(meta.get('long_term_memory', False)), negative_sample=bool(meta.get('negative_sample', False)))
        out.append(row)
        _update_memory_bank_mm(memory_bank, turn_row, thumb_size=thumb_size, thumb_cache_dir=thumb_cache_dir)
    return out

def _iter_rows(scenarios_dir: Path, results_dir: Path, result_suffix: str, start: int, end: int, allow_missing_result: bool, thumb_size: int, thumb_cache_dir: Path, max_desc_chars: int) -> Iterable[TurnRow]:
    for sample_id in range(start, end + 1):
        sid = f'{sample_id:03d}'
        scenario_path = scenarios_dir / f'sample_{sid}.jsonl'
        result_path = results_dir / f'sample_{sid}_{result_suffix}.json'
        if not scenario_path.exists():
            raise FileNotFoundError(f'Missing scenario file: {scenario_path}')
        rows = _build_rows_for_sample(sample_id, scenario_path, result_path, allow_missing_result, thumb_size=thumb_size, thumb_cache_dir=thumb_cache_dir, max_desc_chars=max_desc_chars)
        for row in rows:
            yield row

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def write_sft(rows: List[TurnRow], output_path: Path) -> int:
    _ensure_parent(output_path)
    n = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            data = {'messages': [{'role': 'user', 'content': row.prompt}, {'role': 'assistant', 'content': _format_target_text(row.target_ranks)}], 'images': row.images, 'sample_id': row.sample_id, 'turn': row.turn, 'target_ranks': row.target_ranks, 'predicted_ranks': row.predicted_ranks, 'is_correct': row.is_correct, 'task_type': row.task_type, 'difficulty': row.difficulty, 'topic': row.topic, 'topic_switch': row.topic_switch, 'cross_topic_retrieval': row.cross_topic_retrieval, 'long_term_memory': row.long_term_memory, 'negative_sample': row.negative_sample}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            n += 1
    return n

def write_rl(rows: List[TurnRow], output_path: Path, include_error: bool, include_hard: bool, include_negative: bool) -> int:
    _ensure_parent(output_path)
    n = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            is_hard = row.difficulty in {'hard', 'very_hard'}
            if not (include_error and (not row.is_correct) or (include_hard and is_hard) or (include_negative and row.negative_sample)):
                continue
            data = {'messages': [{'role': 'user', 'content': row.prompt}], 'images': row.images, 'retrieval_target': row.target_ranks, 'sample_id': row.sample_id, 'turn': row.turn, 'predicted_ranks': row.predicted_ranks, 'is_correct': row.is_correct, 'task_type': row.task_type, 'difficulty': row.difficulty, 'topic': row.topic}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            n += 1
    return n

def _write_jsonl_row(f, row: Dict) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build retrieval SFT/RL datasets from benchmark scenarios and results.')
    parser.add_argument('--scenarios-dir', type=Path, required=True, help='Scenario directory containing sample_XXX.jsonl')
    parser.add_argument('--results-dir', type=Path, required=True, help='Result directory containing sample_XXX_<suffix>.json')
    parser.add_argument('--result-suffix', type=str, default='kv_cache_baseline_tags_desc', help='Result filename suffix (default: kv_cache_baseline_tags_desc)')
    parser.add_argument('--thumb-size', type=int, default=448, help='Thumbnail size for multimodal training images (default: 448)')
    parser.add_argument('--thumb-cache-dir', type=Path, default=Path('./datasets/retrieval_300/thumb_cache_448'), help='Thumbnail cache directory (default: benchmark/datasets/retrieval_300/thumb_cache_448)')
    parser.add_argument('--max-desc-chars', type=int, default=600, help='Max characters to keep for each description in prompt (0 = no truncation).')
    parser.add_argument('--start', type=int, default=1, help='Start sample id')
    parser.add_argument('--end', type=int, default=300, help='End sample id')
    parser.add_argument('--allow-missing-result', action='store_true', help='Skip missing result files instead of failing.')
    parser.add_argument('--sft-output', type=Path, default=None, help='Write SFT dataset jsonl to this path.')
    parser.add_argument('--rl-output', type=Path, default=None, help='Write RL dataset jsonl to this path.')
    parser.add_argument('--rl-include-error', action='store_true', help='RL set includes wrong predictions.')
    parser.add_argument('--rl-include-hard', action='store_true', help='RL set includes hard/very_hard turns.')
    parser.add_argument('--rl-include-negative', action='store_true', help='RL set includes negative_sample turns.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    include_error = args.rl_include_error
    include_hard = args.rl_include_hard
    include_negative = args.rl_include_negative
    if args.rl_output and (not (include_error or include_hard or include_negative)):
        include_error = True
        include_hard = True
    n_total = 0
    n_exact = 0
    n_sft = 0
    n_rl = 0
    sft_f = None
    rl_f = None
    try:
        if args.sft_output:
            _ensure_parent(args.sft_output)
            sft_f = open(args.sft_output, 'w', encoding='utf-8')
        if args.rl_output:
            _ensure_parent(args.rl_output)
            rl_f = open(args.rl_output, 'w', encoding='utf-8')
        for row in _iter_rows(scenarios_dir=args.scenarios_dir, results_dir=args.results_dir, result_suffix=args.result_suffix, start=args.start, end=args.end, allow_missing_result=args.allow_missing_result, thumb_size=args.thumb_size, thumb_cache_dir=args.thumb_cache_dir, max_desc_chars=args.max_desc_chars):
            n_total += 1
            if row.is_correct:
                n_exact += 1
            if sft_f is not None:
                data = {'messages': [{'role': 'user', 'content': row.prompt}, {'role': 'assistant', 'content': _format_target_text(row.target_ranks)}], 'images': row.images, 'sample_id': row.sample_id, 'turn': row.turn, 'target_ranks': row.target_ranks, 'predicted_ranks': row.predicted_ranks, 'is_correct': row.is_correct, 'task_type': row.task_type, 'difficulty': row.difficulty, 'topic': row.topic, 'topic_switch': row.topic_switch, 'cross_topic_retrieval': row.cross_topic_retrieval, 'long_term_memory': row.long_term_memory, 'negative_sample': row.negative_sample}
                _write_jsonl_row(sft_f, data)
                n_sft += 1
            if rl_f is not None:
                is_hard = row.difficulty in {'hard', 'very_hard'}
                if include_error and (not row.is_correct) or (include_hard and is_hard) or (include_negative and row.negative_sample):
                    data = {'messages': [{'role': 'user', 'content': row.prompt}], 'images': row.images, 'retrieval_target': row.target_ranks, 'sample_id': row.sample_id, 'turn': row.turn, 'predicted_ranks': row.predicted_ranks, 'is_correct': row.is_correct, 'task_type': row.task_type, 'difficulty': row.difficulty, 'topic': row.topic}
                    _write_jsonl_row(rl_f, data)
                    n_rl += 1
            if n_total % 500 == 0:
                acc = n_exact / n_total if n_total else 0.0
                print(f'Progress: rows={n_total}, exact_acc={acc:.2%}', flush=True)
    finally:
        if sft_f is not None:
            sft_f.close()
        if rl_f is not None:
            rl_f.close()
    if n_total == 0:
        raise RuntimeError('No rows generated. Please check paths/range.')
    acc = n_exact / n_total
    print(f'Loaded turn rows: {n_total}')
    print(f'Current exact-match accuracy in source results: {n_exact}/{n_total} ({acc:.2%})')
    if args.sft_output:
        print(f'Wrote SFT dataset: {n_sft} rows -> {args.sft_output}')
    if args.rl_output:
        print(f'Wrote RL dataset: {n_rl} rows -> {args.rl_output} (error={include_error}, hard={include_hard}, negative={include_negative})')
if __name__ == '__main__':
    main()
