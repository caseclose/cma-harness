"""
Local VLM B retrieval inference service (HTTP, stdlib only).

Intended use:
- Run ONE instance per node on a reserved GPU (e.g. GPU7).
- RL reward plugin calls this service via localhost to get VLM B predicted ranks,
  so VLM A can be optimized with downstream retrieval reward.
"""
import argparse
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

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
                for k in ['selected_index', 'selected_indices', 'ranks']:
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

def _build_vlmb_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    """
    Build prompt in the same style as retrieval SFT/RL:
    - candidates list aligned with `images[]`
    - memory lines include tags + description
    """
    if not candidates:
        return f'候选图片列表（与 images 字段顺序一致）：\n（当前无候选图片）\n\n记忆库信息（包含 tags + description）：\n（当前无历史记忆）\n\n用户查询：{query}\n\n请根据用户查询和记忆库信息，选择最相关的图像ID列表。如果不需要引用历史图片，请输出空列表。'
    cand_lines = []
    mem_lines = []
    for c in candidates:
        rank = int(c.get('rank', 0))
        cand_lines.append(f'- rank {rank}: <image>')
        tags = c.get('tags', [])
        if not isinstance(tags, list):
            tags = []
        tags_text = ', '.join((str(t) for t in tags if t is not None))
        desc = c.get('description', '')
        if not isinstance(desc, str):
            desc = str(desc) if desc is not None else ''
        mem_lines.append(f'记忆[{rank}] 标签: {tags_text} 描述: {desc}')
    return '候选图片列表（与 images 字段顺序一致）：\n' + '\n'.join(cand_lines) + '\n\n记忆库信息（包含 tags + description）：\n' + '\n'.join(mem_lines) + f'\n\n用户查询：{query}\n\n请根据用户查询和记忆库信息，选择最相关的图像ID列表。如果不需要引用历史图片，请输出空列表。'

class _ServiceState:

    def __init__(self, model, max_desc_chars: int=0):
        self.model = model
        self.lock = threading.Lock()
        self.max_desc_chars = max_desc_chars
STATE: Optional[_ServiceState] = None

class Handler(BaseHTTPRequestHandler):

    def _send_json(self, code: int, obj: Dict[str, Any]) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip('/') == '/health':
            self._send_json(200, {'ok': True})
        else:
            self._send_json(404, {'ok': False, 'error': 'not found'})

    def do_POST(self):
        if self.path.rstrip('/') != '/retrieve':
            self._send_json(404, {'ok': False, 'error': 'not found'})
            return
        t_all = __import__('time').time()
        try:
            length = int(self.headers.get('Content-Length', '0'))
            body = self.rfile.read(length).decode('utf-8')
            payload = json.loads(body) if body else {}
        except Exception as e:
            self._send_json(400, {'ok': False, 'error': f'bad json: {e}'})
            return
        requests = payload.get('requests')
        if not isinstance(requests, list) or not requests:
            self._send_json(400, {'ok': False, 'error': 'payload.requests must be a non-empty list'})
            return
        max_new_tokens = int(payload.get('max_new_tokens', 128))
        temperature = float(payload.get('temperature', 0.0))
        assert STATE is not None
        results = []
        for req in requests:
            query = req.get('query', '')
            candidates = req.get('candidates', [])
            if not isinstance(candidates, list):
                candidates = []
            images = req.get('images', None)
            if images is None:
                images = [c.get('image', '') for c in candidates]
            images = [p for p in images if isinstance(p, str) and p]
            if STATE.max_desc_chars and candidates:
                trimmed = []
                for c in candidates:
                    cc = dict(c)
                    d = cc.get('description', '')
                    if isinstance(d, str) and len(d) > STATE.max_desc_chars:
                        cc['description'] = d[:STATE.max_desc_chars].rstrip() + '...'
                    trimmed.append(cc)
                candidates = trimmed
            prompt = _build_vlmb_prompt(str(query), candidates)
            messages = [{'role': 'user', 'content': prompt}]
            with STATE.lock:
                out = STATE.model.infer(messages=messages, images=images if images else None, max_new_tokens=max_new_tokens, temperature=temperature)
            pred = _normalize_ids(out)
            results.append({'pred_ranks': pred, 'raw': out})
        elapsed = __import__('time').time() - t_all
        print(f'[vlmb_service] /retrieve ok: batch={len(requests)} elapsed={elapsed:.2f}s max_new_tokens={max_new_tokens} temp={temperature}', flush=True)
        self._send_json(200, {'ok': True, 'results': results})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='VLM B LoRA checkpoint path (has args.json)')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=int(os.environ.get('VLM_B_SERVICE_PORT', '29640')))
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index (within this process)')
    parser.add_argument('--max-desc-chars', type=int, default=0, help='Optional truncate for candidate descriptions')
    args = parser.parse_args()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, this_dir)
    os.environ.setdefault('PYTHONPATH', this_dir)
    from multimodal_agent_service import VLMModel
    model = VLMModel(args.checkpoint, 'VLM B Service', gpu_ids=[args.gpu])
    global STATE
    STATE = _ServiceState(model=model, max_desc_chars=args.max_desc_chars)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'[vlmb_service] listening on http://{args.host}:{args.port}')
    print(f'[vlmb_service] checkpoint={args.checkpoint}')
    print(f'[vlmb_service] gpu={args.gpu}')
    server.serve_forever()
if __name__ == '__main__':
    main()
