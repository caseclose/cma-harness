"""
计算 baseline 版本（仅使用448缩略图）的检索准确率
"""
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

def normalize_indices(indices) -> Optional[List[int]]:
    """规范化索引列表"""
    if indices is None:
        return None
    if isinstance(indices, list):
        return sorted([int(i) for i in indices if isinstance(i, (int, str)) and str(i).isdigit()])
    if isinstance(indices, (int, str)):
        try:
            return [int(indices)]
        except:
            return None
    return None

def compare_sample(result_path: Path, scenario_path: Path, turn_start: Optional[int]=None, turn_end: Optional[int]=None) -> Tuple[int, int, int]:
    """
    比较单个样本的检索结果。
    约定：null 与 [] 等价，都表示「不检索」，不单独统计 missing。
    
    Returns: (total, correct, missing)
    total: 参与评估的轮次
    correct: 预测索引与 retrieval_target 完全一致
    missing: 恒为 0（保留参数兼容）
    """
    scenario_turns = {}
    with open(scenario_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                turn_data = json.loads(line)
                turn_id = turn_data.get('turn')
                scenario_turns[turn_id] = turn_data
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    total = 0
    correct = 0
    missing = 0
    for turn in result_data.get('turns', []):
        turn_id = int(turn.get('turn'))
        if turn_start is not None and turn_id < turn_start:
            continue
        if turn_end is not None and turn_id > turn_end:
            continue
        scenario_turn = scenario_turns.get(turn_id)
        if not scenario_turn:
            continue
        retrieval_target = scenario_turn.get('meta', {}).get('retrieval_target', None)
        if retrieval_target is None:
            continue
        total += 1
        expected = normalize_indices(retrieval_target) or []
        predicted = normalize_indices(turn.get('agent_output', {}).get('qwen3vl_selected_index', None))
        if predicted is None:
            predicted = []
        if sorted(expected) == sorted(predicted):
            correct += 1
    return (total, correct, missing)

def main():
    parser = argparse.ArgumentParser(description='计算 baseline 版本（仅使用448缩略图）的检索准确率')
    parser.add_argument('--results-dir', type=str, default='./results/kv_cache_baseline_20_results_qwen3vl_8b_thumbs_448_hard', help='结果文件目录')
    parser.add_argument('--scenarios-dir', type=str, default='./scenarios/50x20-gemini-hard', help='场景文件目录')
    parser.add_argument('--result-suffix', type=str, default='kv_cache_baseline', help='结果文件后缀')
    parser.add_argument('--start', type=int, default=1, help='起始 sample ID')
    parser.add_argument('--end', type=int, default=50, help='结束 sample ID')
    parser.add_argument('--turn-start', type=int, default=None, help='起始轮次（可选）')
    parser.add_argument('--turn-end', type=int, default=None, help='结束轮次（可选）')
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    scenarios_dir = Path(args.scenarios_dir)
    total_all = 0
    correct_all = 0
    missing_all = 0
    print('=' * 80)
    print('计算检索准确率（Baseline - 仅448缩略图）')
    print('=' * 80)
    print(f'结果目录: {results_dir}')
    print(f'场景目录: {scenarios_dir}')
    print(f'样本范围: {args.start} - {args.end}')
    if args.turn_start or args.turn_end:
        print(f"轮次范围: {args.turn_start or '开始'} - {args.turn_end or '结束'}")
    print('=' * 80)
    print()
    for sample_id in range(args.start, args.end + 1):
        sample_id_str = f'{sample_id:03d}'
        result_path = results_dir / f'sample_{sample_id_str}_{args.result_suffix}.json'
        scenario_path = scenarios_dir / f'sample_{sample_id_str}.jsonl'
        if not result_path.exists():
            print(f'⚠ sample_{sample_id_str}: 结果文件不存在 - {result_path}')
            continue
        if not scenario_path.exists():
            print(f'⚠ sample_{sample_id_str}: 场景文件不存在 - {scenario_path}')
            continue
        total, correct, missing = compare_sample(result_path, scenario_path, turn_start=args.turn_start, turn_end=args.turn_end)
        total_all += total
        correct_all += correct
        missing_all += missing
        if total > 0:
            acc = correct / total
            print(f'sample_{sample_id_str}: total={total:3d} correct={correct:3d} acc={acc:.3f}')
        else:
            print(f'sample_{sample_id_str}: 无检索任务')
    if total_all > 0:
        acc_all = correct_all / total_all
    else:
        acc_all = 0.0
    print()
    print('=' * 80)
    print('总体统计')
    print('=' * 80)
    print(f'总检索任务数: {total_all}')
    print(f'正确检索数: {correct_all}')
    print(f'错误检索数: {total_all - correct_all}')
    print(f'检索准确率: {acc_all:.3f} ({acc_all * 100:.2f}%)')
    print('(约定: null 与 [] 等价，均表示不检索)')
    print('=' * 80)
if __name__ == '__main__':
    main()
