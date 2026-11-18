import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler
from swift.plugin import ORM, orms

def _normalize_list(value: Any) -> List[Any]:
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
            if isinstance(parsed, dict):
                for k in ['tags', 'tag_list', 'selected_index', 'selected_indices', 'ranks']:
                    if k in parsed:
                        return _normalize_list(parsed[k])
        except Exception:
            pass
        nums = re.findall('-?\\d+', text)
        return [int(n) for n in nums] if nums else []
    return []

def _normalize_ids(value: Any) -> List[int]:
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

def _extract_json_obj(raw: str) -> Optional[Any]:
    text = (raw or '').strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search('(\\{.*\\}|\\[.*\\])', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def _normalize_tags_desc_obj(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    tags = obj.get('tags', [])
    desc = obj.get('description', '')
    if not isinstance(tags, list):
        return None
    if not isinstance(desc, str):
        desc = str(desc) if desc is not None else ''
    tags = [str(t).strip() for t in tags if t is not None and str(t).strip() != '']
    return {'tags': tags, 'description': desc}

def _parse_tags_desc_from_completion(text: str) -> Optional[Dict[str, Any]]:
    """
    Expect a JSON object with:
      - tags: list[str]
      - description: str
    """
    obj = _extract_json_obj(text)
    return _normalize_tags_desc_obj(obj)

def _parse_ranked_tags_desc_from_completion(text: str) -> Dict[int, Dict[str, Any]]:
    """
    Parse all-ranks memory output from completion.

    Supported JSON formats:
    1) {"memories":[{"rank":0,"tags":[...],"description":"..."}, ...]}
    2) [{"rank":0,"tags":[...],"description":"..."}, ...]
    3) {"0":{"tags":[...],"description":"..."}, "1":{...}}
    """
    obj = _extract_json_obj(text)
    if obj is None:
        return {}
    ranked: Dict[int, Dict[str, Any]] = {}

    def _put(rank: Any, item: Any) -> None:
        if not str(rank).lstrip('-').isdigit():
            return
        r = int(rank)
        if r < 0:
            return
        parsed = _normalize_tags_desc_obj(item)
        if parsed is None:
            return
        ranked[r] = parsed
    if isinstance(obj, dict):
        memories = obj.get('memories')
        if isinstance(memories, list):
            for idx, item in enumerate(memories):
                if not isinstance(item, dict):
                    continue
                rank = item.get('rank', idx)
                _put(rank, item)
            return ranked
        for k, v in obj.items():
            _put(k, v)
        if not ranked and 'rank' in obj:
            _put(obj.get('rank'), obj)
        return ranked
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            rank = item.get('rank', idx)
            _put(rank, item)
    return ranked

def _post_json(url: str, payload: Dict[str, Any], timeout_s: float=30.0) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = Request(url, data=data, headers={'Content-Type': 'application/json'})
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read().decode('utf-8')
        return json.loads(raw) if raw else {}
    except HTTPError as e:
        return {'ok': False, 'error': f'http_error {e.code}: {e.reason}'}
    except URLError as e:
        return {'ok': False, 'error': f'url_error: {e}'}
    except Exception as e:
        return {'ok': False, 'error': f'request_error: {e}'}

class VlmADownstreamRetrievalReward(ORM):
    """
    Downstream reward for VLM A (tags+description generator):
    1) Parse VLM A completion (single_regen or all_ranks mode). Invalid parse -> reward 0.
    2) Substitute generated tags/desc into candidate memory bank.
    3) Query local frozen VLM B service for retrieval.
    4) Reward = Jaccard(pred_ranks, retrieval_target), hard/very_hard * 1.2.
       In all_ranks mode, multiply by coverage ratio (replaced_ranks / total_ranks).
    """

    def __call__(self, completions, retrieval_target, difficulty=None, vlmb_user_query=None, vlmb_candidates=None, regen_rank=None, reward_mode=None, **kwargs) -> List[float]:
        if difficulty is None:
            difficulty = [''] * len(completions)
        if vlmb_user_query is None:
            vlmb_user_query = [''] * len(completions)
        if vlmb_candidates is None:
            vlmb_candidates = [[] for _ in completions]
        if regen_rank is None:
            regen_rank = [0] * len(completions)
        if reward_mode is None:
            reward_mode = [''] * len(completions)
        service_url = os.environ.get('VLM_B_SERVICE_URL', 'http://127.0.0.1:29640/retrieve')
        timeout_s = float(os.environ.get('VLM_B_SERVICE_TIMEOUT_S', '300'))
        max_desc_chars = int(os.environ.get('VLM_B_SERVICE_MAX_DESC_CHARS', '0'))
        req_batch_size = int(os.environ.get('VLM_B_SERVICE_BATCH_SIZE', '4'))
        req_batch_size = max(1, req_batch_size)
        max_retries = int(os.environ.get('VLM_B_SERVICE_MAX_RETRIES', '2'))
        retry_wait_s = float(os.environ.get('VLM_B_SERVICE_RETRY_WAIT_S', '2'))
        mode_env = os.environ.get('VLM_A_REWARD_MODE', 'auto').strip().lower()
        requests = []
        parsed_ok: List[bool] = []
        all_ranks_flags: List[bool] = []
        coverage_list: List[float] = []
        for comp, q, cands, r, mode_item in zip(completions, vlmb_user_query, vlmb_candidates, regen_rank, reward_mode):
            mode_row = str(mode_item).strip().lower()
            if mode_env in {'all_ranks', 'single_regen'}:
                mode = mode_env
            elif mode_row in {'all_ranks', 'single_regen'}:
                mode = mode_row
            else:
                mode = 'single_regen'
            is_all_ranks = mode == 'all_ranks'
            all_ranks_flags.append(is_all_ranks)
            if not isinstance(cands, list):
                cands = []
            new_cands: List[Dict[str, Any]] = []
            for c in cands:
                if isinstance(c, dict):
                    new_cands.append(dict(c))
            if not new_cands:
                parsed_ok.append(False)
                coverage_list.append(0.0)
                requests.append(None)
                continue
            if is_all_ranks:
                parsed_map = _parse_ranked_tags_desc_from_completion(comp)
                if not parsed_map:
                    parsed_ok.append(False)
                    coverage_list.append(0.0)
                    requests.append(None)
                    continue
                replaced = 0
                for c in new_cands:
                    try:
                        rr = int(c.get('rank', -1))
                    except Exception:
                        rr = -1
                    parsed = parsed_map.get(rr)
                    if parsed is None:
                        continue
                    c['tags'] = parsed['tags']
                    c['description'] = parsed['description']
                    replaced += 1
                coverage = replaced / max(1, len(new_cands))
                parsed_ok.append(replaced > 0)
                coverage_list.append(coverage)
            else:
                parsed = _parse_tags_desc_from_completion(comp)
                if parsed is None:
                    parsed_ok.append(False)
                    coverage_list.append(0.0)
                    requests.append(None)
                    continue
                rr = int(r) if str(r).lstrip('-').isdigit() else 0
                replaced = 0
                for c in new_cands:
                    try:
                        if int(c.get('rank', -1)) == rr:
                            c['tags'] = parsed['tags']
                            c['description'] = parsed['description']
                            replaced += 1
                    except Exception:
                        continue
                parsed_ok.append(replaced > 0)
                coverage_list.append(1.0 if replaced > 0 else 0.0)
            images = [c.get('image', '') for c in new_cands if isinstance(c, dict)]
            images = [p for p in images if isinstance(p, str) and p]
            if max_desc_chars:
                trimmed = []
                for c in new_cands:
                    cc = dict(c)
                    d = cc.get('description', '')
                    if isinstance(d, str) and len(d) > max_desc_chars:
                        cc['description'] = d[:max_desc_chars].rstrip() + '...'
                    trimmed.append(cc)
                new_cands = trimmed
            requests.append({'query': str(q), 'candidates': new_cands, 'images': images})
        idx_map = [i for i, req in enumerate(requests) if req is not None]
        service_results: Dict[int, Dict[str, Any]] = {}
        if idx_map:
            total_calls = 0
            total_elapsed = 0.0
            for start in range(0, len(idx_map), req_batch_size):
                sub_idx = idx_map[start:start + req_batch_size]
                batch = [requests[i] for i in sub_idx]
                resp: Dict[str, Any] = {}
                ok = False
                last_err = 'unknown'
                for attempt in range(max_retries + 1):
                    t0 = time.time()
                    resp = _post_json(service_url, {'requests': batch, 'max_new_tokens': 128, 'temperature': 0.0}, timeout_s=timeout_s)
                    elapsed = time.time() - t0
                    total_elapsed += elapsed
                    total_calls += 1
                    if resp.get('ok', False):
                        ok = True
                        break
                    last_err = str(resp.get('error', 'unknown'))
                    if attempt < max_retries:
                        time.sleep(retry_wait_s)
                if not ok:
                    print(f'[vlm_a_downstream_retrieval] service call failed after retries: {last_err}')
                    return [0.0] * len(completions)
                results = resp.get('results', [])
                if not isinstance(results, list) or len(results) != len(batch):
                    print('[vlm_a_downstream_retrieval] service returned invalid results shape')
                    return [0.0] * len(completions)
                for i, out in zip(sub_idx, results):
                    service_results[i] = out if isinstance(out, dict) else {}
            print(f'[vlm_a_downstream_retrieval] service ok: rows={len(idx_map)} calls={total_calls} total_latency={total_elapsed:.2f}s timeout={timeout_s}s chunk={req_batch_size}')
        rewards: List[float] = []
        for i, (target, diff) in enumerate(zip(retrieval_target, difficulty)):
            if not parsed_ok[i]:
                rewards.append(0.0)
                continue
            pred_ranks = _normalize_ids(service_results.get(i, {}).get('pred_ranks'))
            gt_ranks = _normalize_ids(target)
            base = 1.0 if pred_ranks == gt_ranks else _jaccard(pred_ranks, gt_ranks)
            if all_ranks_flags[i]:
                base *= coverage_list[i]
            if str(diff).lower() in {'hard', 'very_hard'}:
                base *= 1.2
            rewards.append(base)
        return rewards
orms['vlm_a_downstream_retrieval'] = VlmADownstreamRetrievalReward
