"""
Split retrieval dataset by session(sample_id) to avoid leakage.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows

def _dump_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def _group_by_session(rows: List[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = {}
    for row in rows:
        sid = int(row['sample_id'])
        grouped.setdefault(sid, []).append(row)
    return grouped

def _split_ids(session_ids: List[int], train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-08:
        raise ValueError(f'Ratios must sum to 1.0, got {total_ratio}')
    n = len(session_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    train_ids = session_ids[:n_train]
    val_ids = session_ids[n_train:n_train + n_val]
    test_ids = session_ids[n_train + n_val:n_train + n_val + n_test]
    return (train_ids, val_ids, test_ids)

def _flatten(grouped: Dict[int, List[Dict]], ids: List[int]) -> List[Dict]:
    out: List[Dict] = []
    for sid in ids:
        rows = grouped.get(sid, [])
        rows_sorted = sorted(rows, key=lambda x: int(x.get('turn', 0)))
        out.extend(rows_sorted)
    return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Split retrieval JSONL by sample_id')
    parser.add_argument('--input', type=Path, required=True, help='Input JSONL path')
    parser.add_argument('--out-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--prefix', type=str, default='retrieval_300', help='Output file prefix')
    parser.add_argument('--seed', type=int, default=42, help='Shuffle seed for session ids')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--hard-difficulties', nargs='+', default=['hard', 'very_hard'], help='Difficulty values used to build train_hard subset.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.input)
    if not rows:
        raise RuntimeError(f'Input is empty: {args.input}')
    grouped = _group_by_session(rows)
    session_ids = sorted(grouped.keys())
    rng = random.Random(args.seed)
    rng.shuffle(session_ids)
    train_ids, val_ids, test_ids = _split_ids(session_ids, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    train_rows = _flatten(grouped, train_ids)
    val_rows = _flatten(grouped, val_ids)
    test_rows = _flatten(grouped, test_ids)
    hard_set = set(args.hard_difficulties)
    train_hard_rows = [r for r in train_rows if str(r.get('difficulty', '')) in hard_set]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f'{args.prefix}_train.jsonl'
    val_path = out_dir / f'{args.prefix}_val.jsonl'
    test_path = out_dir / f'{args.prefix}_test.jsonl'
    train_hard_path = out_dir / f'{args.prefix}_train_hard.jsonl'
    split_meta_path = out_dir / f'{args.prefix}_split_meta.json'
    _dump_jsonl(train_path, train_rows)
    _dump_jsonl(val_path, val_rows)
    _dump_jsonl(test_path, test_rows)
    _dump_jsonl(train_hard_path, train_hard_rows)
    split_meta = {'input': str(args.input), 'seed': args.seed, 'ratios': {'train': args.train_ratio, 'val': args.val_ratio, 'test': args.test_ratio}, 'session_counts': {'total': len(session_ids), 'train': len(train_ids), 'val': len(val_ids), 'test': len(test_ids)}, 'row_counts': {'train': len(train_rows), 'val': len(val_rows), 'test': len(test_rows), 'train_hard': len(train_hard_rows)}, 'session_ids': {'train': train_ids, 'val': val_ids, 'test': test_ids}}
    with open(split_meta_path, 'w', encoding='utf-8') as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)
    print('Split completed.')
    print(f'Train: {len(train_ids)} sessions, {len(train_rows)} rows -> {train_path}')
    print(f'Val:   {len(val_ids)} sessions, {len(val_rows)} rows -> {val_path}')
    print(f'Test:  {len(test_ids)} sessions, {len(test_rows)} rows -> {test_path}')
    print(f'Train hard rows: {len(train_hard_rows)} -> {train_hard_path}')
    print(f'Meta: {split_meta_path}')
if __name__ == '__main__':
    main()
