import json
import re
from typing import List
from swift.plugin import ORM, orms

def _normalize_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        text = value.strip()
        if text == '' or text.lower() in {'null', 'none'}:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        nums = re.findall('-?\\d+', text)
        return [int(n) for n in nums] if nums else []
    return []

def _normalize_ids(value) -> List[int]:
    out: List[int] = []
    for item in _normalize_list(value):
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, float):
            out.append(int(item))
        elif isinstance(item, str) and item.strip().lstrip('-').isdigit():
            out.append(int(item.strip()))
    return sorted(out)

def _jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = (set(a), set(b))
    if not sa and (not sb):
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)

class RetrievalSoftMatchReward(ORM):
    """
    Reward by overlap between predicted IDs and target IDs.
    - exact match: 1.0
    - partial match: jaccard score in [0,1)
    - hard / very_hard samples get extra weight.
    """

    def __call__(self, completions, retrieval_target, difficulty=None, **kwargs) -> List[float]:
        if difficulty is None:
            difficulty = [''] * len(completions)
        rewards = []
        for completion, target, diff in zip(completions, retrieval_target, difficulty):
            pred_ids = _normalize_ids(completion)
            gt_ids = _normalize_ids(target)
            if pred_ids == gt_ids:
                base = 1.0
            else:
                base = _jaccard(pred_ids, gt_ids)
            diff_text = str(diff).lower()
            if diff_text in {'hard', 'very_hard'}:
                base *= 1.2
            rewards.append(base)
        return rewards
orms['retrieval_soft_match_300'] = RetrievalSoftMatchReward
