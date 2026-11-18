"""
Build VLM A RL dataset from retrieval RL dataset.

Supported reward modes:
1) single_regen (legacy A2):
   - One VLM A sample rewrites one candidate rank (regen_rank).
   - Reward plugin substitutes only that rank before calling frozen VLM B.

2) all_ranks:
   - One VLM A sample sees all candidate images for this turn and outputs tags+description
     for all ranks in one JSON.
   - Reward plugin substitutes as many ranks as parsed, then calls frozen VLM B.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
TAG_SPLIT_RE = re.compile('[,\\uFF0C]\\s*')

def _extract_user_query(prompt: str) -> str:
    m = re.search('用户查询：(.+?)\\n\\s*\\n', prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = prompt.rfind('用户查询：')
    if idx >= 0:
        return prompt[idx + len('用户查询：'):].strip()
    return ''

def _parse_memory_lines(prompt: str) -> Dict[int, Tuple[List[str], str]]:
    """
    Parse lines like:
      记忆[3] 标签: a, b, c 描述: xxx...
    Return: rank -> (tags_list, description)
    """
    out: Dict[int, Tuple[List[str], str]] = {}
    for m in re.finditer('记忆\\[(\\d+)\\]\\s*标签:\\s*(.*?)\\s*描述:\\s*(.*?)(?:\\n|$)', prompt):
        rank = int(m.group(1))
        tags_text = m.group(2).strip()
        desc = m.group(3).strip()
        tags = [t.strip() for t in TAG_SPLIT_RE.split(tags_text) if t.strip()]
        out[rank] = (tags, desc)
    return out

def _default_vlm_a_prompt_single() -> str:
    return '<image>Please analyze this image in detail and generate complete image features + description in English.The output must be JSON containing the following fields: tags (tag list, at least 5 tags, array format), description (detailed description, at least 100 words). Please ensure the output is valid JSON.'

def _default_vlm_a_prompt_all_ranks(n_cand: int, user_query: str) -> str:
    cand_lines = '\n'.join([f'- rank {i}: <image>' for i in range(n_cand)])
    return f'You will see multiple historical candidate images (order matches rank one by one).\nCandidate image list:\n{cand_lines}\n\nCurrent user query: {user_query}\n\nPlease generate tags and description for the image of each rank.\nOutput must be strict JSON (do not include markdown code blocks), format:\n{{\n  "memories": [\n    {{"rank": 0, "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"], "description": "an English description of at least 100 words"}},\n    {{"rank": 1, "tags": ["..."], "description": "..."}}\n  ]\n}}\nRequirements:\n1) Cover all current ranks;\n2) At least 5 tags for each rank;\n3) Description should be in English and as detailed as possible;\n4) Output JSON only.'

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('./datasets/retrieval_300/retrieval_300_rl_mm_thumbs448.jsonl'))
    ap.add_argument('--output', type=Path, default=Path('./datasets/retrieval_300/vlm_a_rl_a2_mm_thumbs448.jsonl'))
    ap.add_argument('--reward-mode', choices=['single_regen', 'all_ranks'], default='single_regen', help='single_regen: one sample per regen_rank; all_ranks: one sample per row with all candidate images')
    ap.add_argument('--max-per-row', type=int, default=2, help='Max regen_rank samples per retrieval row (single_regen only)')
    ap.add_argument('--min-candidates', type=int, default=2)
    ap.add_argument('--max-candidates', type=int, default=12)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open('w', encoding='utf-8')
    n_in = 0
    n_out = 0
    with args.input.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            messages = row.get('messages', [])
            if not messages or not isinstance(messages, list):
                continue
            prompt = messages[0].get('content', '')
            if not isinstance(prompt, str) or '记忆库信息（包含 tags + description）' not in prompt:
                continue
            images = row.get('images', [])
            if not isinstance(images, list):
                continue
            mem = _parse_memory_lines(prompt)
            if not mem:
                continue
            n_cand = len(images)
            if n_cand < args.min_candidates or n_cand > args.max_candidates:
                continue
            user_query = _extract_user_query(prompt)
            retrieval_target = row.get('retrieval_target', [])
            if not isinstance(retrieval_target, list):
                retrieval_target = []
            retrieval_target = [int(x) for x in retrieval_target if str(x).lstrip('-').isdigit()]
            candidates: List[Dict[str, Any]] = []
            for rank in range(n_cand):
                tags, desc = mem.get(rank, ([], ''))
                candidates.append({'rank': rank, 'image': images[rank], 'tags': tags, 'description': desc})
            if args.reward_mode == 'all_ranks':
                sample = {'messages': [{'role': 'user', 'content': _default_vlm_a_prompt_all_ranks(n_cand=n_cand, user_query=user_query)}], 'images': images, 'retrieval_target': retrieval_target, 'difficulty': row.get('difficulty', ''), 'task_type': row.get('task_type', ''), 'topic': row.get('topic', ''), 'sample_id': row.get('sample_id', None), 'turn': row.get('turn', None), 'reward_mode': 'all_ranks', 'vlmb_user_query': user_query, 'vlmb_candidates': candidates}
                out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                n_out += 1
                continue
            regen_ranks = []
            for r in retrieval_target:
                if 0 <= r < n_cand:
                    regen_ranks.append(r)
            regen_ranks = regen_ranks[:max(1, args.max_per_row)]
            for rr in regen_ranks:
                sample = {'messages': [{'role': 'user', 'content': _default_vlm_a_prompt_single()}], 'images': [images[rr]], 'retrieval_target': retrieval_target, 'difficulty': row.get('difficulty', ''), 'task_type': row.get('task_type', ''), 'topic': row.get('topic', ''), 'sample_id': row.get('sample_id', None), 'turn': row.get('turn', None), 'regen_rank': rr, 'reward_mode': 'single_regen', 'vlmb_user_query': user_query, 'vlmb_candidates': candidates}
                out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                n_out += 1
    out_f.close()
    print(f'done. mode={args.reward_mode} input_rows={n_in} output_rows={n_out} -> {args.output}')
if __name__ == '__main__':
    main()
