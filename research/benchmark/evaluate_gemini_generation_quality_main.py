"""
Gemini generation quality evaluator for multi-turn benchmark outputs.

This script scores each evaluated turn with Gemini (0-10 integer) and
aggregates:
- Overall
- Generate
- Edit
- Understand/Chat
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from eval_utils import GeminiJudge

def _task_bucket(task_type: str) -> str:
    t = (task_type or '').strip().lower()
    if t == 'generate':
        return 'Generate'
    if t == 'edit':
        return 'Edit'
    if t in {'understand', 'pure_chat', 'chat'}:
        return 'Understand/Chat'
    return 'Other'

def _extract_assistant_text(turn: Dict) -> str:
    ao = turn.get('agent_output', {})
    text = ao.get('text', '')
    return str(text) if text is not None else ''

def _extract_assistant_image(turn: Dict) -> Optional[str]:
    ao = turn.get('agent_output', {})
    image_path = ao.get('image_path', None)
    if image_path and str(image_path).strip():
        return str(image_path)
    return None

def _build_eval_prompt(session_id: str, turn_id: int, task_type: str, history_dialogue: List[Tuple[str, str]], current_user_query: str, model_response_text: str, has_response_image: bool) -> str:
    history_lines: List[str] = []
    for role, content in history_dialogue:
        history_lines.append(f'{role}: {content}')
    history_text = '\n'.join(history_lines) if history_lines else '(empty)'
    response_desc = model_response_text if model_response_text else '(no text output)'
    if has_response_image:
        response_desc += '\n[Model also returned an image output (attached in images input if available).]'
    return f"You are given the full multi-turn dialogue history of a session, including:\n\nAll previous user and assistant turns\nAll historical images referenced in the dialogue\nThe current user query (i.e., the latest turn), which should be explicitly highlighted\nThe model's response to the current query (which may be text, image, or both)\n\nYour task is to evaluate the quality of the model's response to the latest user query, considering the full dialogue context.\n\nThe response may correspond to different task types, including but not limited to:\nUnderstand / Chat (text-based response grounded in history)\nImage Generation\nImage Editing\nImage Understanding\n\nEvaluation objectives:\n- Correctly understands the current user intent\n- Appropriately utilizes relevant historical context (text and/or images)\n- Produces an accurate and coherent response\n- Avoids hallucination or irrelevant content\n- Fulfills the requirements of the specific task type\n\nScoring:\nProvide a score from 0 to 10 (integer only), where 10 is perfect.\n\nScoring rubric:\n10/10: perfect understanding, correct/sufficient context usage, accurate/coherent, task-complete, no hallucination.\n8/10: minor imperfections.\n6/10: partially correct.\n4/10: significant errors.\n2/10: mostly incorrect.\n0/10: completely incorrect or unrelated.\n\nSession ID: {session_id}\nTurn ID: {turn_id}\nTask Type: {_task_bucket(task_type)}\n\nFull previous dialogue history:\n{history_text}\n\nCURRENT USER QUERY (latest turn, focus on this):\n{current_user_query}\n\nMODEL RESPONSE TO CURRENT QUERY:\n{response_desc}\n\nOutput format (strictly follow this exact structure; do not add extra sections):\nSession ID: <session_id>\nTurn ID: <turn_id>\nTask Type: <Generate | Edit | Understand | Chat | Other>\nScore (0-10): <integer>\nJustification:\n<concise and concrete explanation referring to intent understanding, context utilization, output correctness, and task completion quality>\n"

def _parse_score_block(text: str) -> Tuple[Optional[int], str]:
    raw = (text or '').strip()
    m = re.search('Score\\s*\\(0-?10\\)\\s*:\\s*([0-9]{1,2})', raw, flags=re.IGNORECASE)
    if not m:
        m = re.search('Score\\s*:\\s*([0-9]{1,2})', raw, flags=re.IGNORECASE)
    score = None
    if m:
        try:
            val = int(m.group(1))
            if 0 <= val <= 10:
                score = val
        except Exception:
            score = None
    return (score, raw)

def _detect_suffix(results_dir: Path) -> str:
    files = sorted(results_dir.glob('sample_*_*.json'))
    if not files:
        raise FileNotFoundError(f'No result files found in {results_dir}')
    m = re.match('sample_\\d+_(.*)\\.json$', files[0].name)
    if not m:
        raise ValueError(f'Unexpected result filename: {files[0].name}')
    return m.group(1)

def _load_turns(result_file: Path) -> List[Dict]:
    data = json.loads(result_file.read_text(encoding='utf-8'))
    return data.get('turns', [])

def _init_stats() -> Dict[str, Dict[str, float]]:
    return {'Overall': {'sum': 0.0, 'cnt': 0}, 'Generate': {'sum': 0.0, 'cnt': 0}, 'Edit': {'sum': 0.0, 'cnt': 0}, 'Understand/Chat': {'sum': 0.0, 'cnt': 0}, 'Other': {'sum': 0.0, 'cnt': 0}}

def _evaluate_one_session(sid: int, results_dir: Path, scenarios_dir: Path, suffix: str, judge_kwargs: Dict, max_retries: int, retry_sleep_s: float, log_progress: bool, log_fn=None) -> Dict:
    sid_str = f'{sid:03d}'
    result_file = results_dir / f'sample_{sid_str}_{suffix}.json'
    scenario_file = scenarios_dir / f'sample_{sid_str}.jsonl'
    if not result_file.exists() or not scenario_file.exists():
        return {'session_id': sid_str, 'ok': False, 'skipped': True, 'error': f'missing file: result_exists={result_file.exists()} scenario_exists={scenario_file.exists()}', 'records': [], 'stats': _init_stats()}
    turns = _load_turns(result_file)
    judge = GeminiJudge(**judge_kwargs)
    if log_progress and log_fn is not None:
        log_fn(f'[session {sid_str}] start turns={len(turns)}')
    session_stats = _init_stats()
    records: List[Dict] = []
    dialogue_history: List[Tuple[str, str]] = []
    history_images: List[str] = []
    for turn in turns:
        turn_id = int(turn.get('turn', -1))
        if turn_id <= 0:
            continue
        task_type = str(turn.get('task_type', 'Other'))
        bucket = _task_bucket(task_type)
        user_query = str(turn.get('user_input', ''))
        assistant_text = _extract_assistant_text(turn)
        assistant_img = _extract_assistant_image(turn)
        judge_images = history_images.copy()
        if assistant_img and Path(assistant_img).exists():
            judge_images.append(assistant_img)
        prompt = _build_eval_prompt(session_id=f'sample_{sid_str}', turn_id=turn_id, task_type=task_type, history_dialogue=dialogue_history, current_user_query=user_query, model_response_text=assistant_text, has_response_image=assistant_img is not None)
        score = None
        raw_judge_output = ''
        err = None
        attempts = 0
        turn_start = time.time()
        if log_progress and log_fn is not None:
            log_fn(f'  [session {sid_str}] turn={turn_id} task={task_type} bucket={bucket} start')
        for retry_idx in range(max_retries):
            attempts = retry_idx + 1
            try:
                raw = judge.chat(prompt=prompt, images=judge_images if judge_images else None, temperature=0.0)
                parsed_score, parsed_raw = _parse_score_block(raw)
                raw_judge_output = parsed_raw
                if parsed_score is not None:
                    score = parsed_score
                    break
                err = 'Failed to parse score from judge output'
            except Exception as e:
                err = str(e)
            if log_progress and log_fn is not None:
                log_fn(f'    [session {sid_str} turn {turn_id}] retry {retry_idx + 1}/{max_retries} failed: {err}')
            time.sleep(retry_sleep_s)
        record = {'session_id': f'sample_{sid_str}', 'turn_id': turn_id, 'task_type': task_type, 'bucket': bucket, 'score_0_10': score, 'judge_output': raw_judge_output, 'error': err if score is None else None, 'attempts': attempts, 'elapsed_s': round(time.time() - turn_start, 4)}
        records.append(record)
        if score is not None:
            session_stats['Overall']['sum'] += score
            session_stats['Overall']['cnt'] += 1
            session_stats[bucket]['sum'] += score
            session_stats[bucket]['cnt'] += 1
            if log_progress and log_fn is not None:
                log_fn(f"  [session {sid_str}] turn={turn_id} score={score} attempts={attempts} elapsed={record['elapsed_s']}s")
        elif log_progress and log_fn is not None:
            log_fn(f"  [session {sid_str}] turn={turn_id} score=NA attempts={attempts} elapsed={record['elapsed_s']}s err={err}")
        dialogue_history.append(('user', user_query))
        dialogue_history.append(('assistant', assistant_text if assistant_text else '[no text]'))
        if assistant_img and Path(assistant_img).exists():
            history_images.append(assistant_img)
    return {'session_id': sid_str, 'ok': True, 'skipped': False, 'error': None, 'records': records, 'stats': session_stats}

def evaluate_method(args):
    results_dir = Path(args.results_dir)
    scenarios_dir = Path(args.scenarios_dir)
    suffix = args.result_suffix or _detect_suffix(results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / 'gemini_quality_eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / f'{args.method_name}_turn_scores.jsonl'
    summary_path = out_dir / f'{args.method_name}_summary.json'
    stats = _init_stats()

    def _avg(k: str) -> Optional[float]:
        c = stats[k]['cnt']
        return round(stats[k]['sum'] / c, 4) if c > 0 else None

    def _build_summary() -> Dict:
        return {'method': args.method_name, 'results_dir': str(results_dir), 'scenarios_dir': str(scenarios_dir), 'result_suffix': suffix, 'range': {'start': args.start, 'end': args.end}, 'scores_0_10': {'Overall': _avg('Overall'), 'Generate': _avg('Generate'), 'Edit': _avg('Edit'), 'Understand/Chat': _avg('Understand/Chat')}, 'counts': {'Overall': stats['Overall']['cnt'], 'Generate': stats['Generate']['cnt'], 'Edit': stats['Edit']['cnt'], 'Understand/Chat': stats['Understand/Chat']['cnt'], 'Other': stats['Other']['cnt']}, 'details_jsonl': str(details_path)}

    def _merge_stats(dst: Dict[str, Dict[str, float]], src: Dict[str, Dict[str, float]]) -> None:
        for k in dst.keys():
            dst[k]['sum'] += src[k]['sum']
            dst[k]['cnt'] += src[k]['cnt']
    judge_kwargs = {'api_base_url': args.api_base_url, 'app_id': args.app_id, 'app_secret': args.app_secret, 'source': args.source, 'model': args.model}
    total_sessions = args.end - args.start + 1
    log_lock = threading.Lock()

    def _log(msg: str) -> None:
        with log_lock:
            print(msg, flush=True)
    print('=' * 80, flush=True)
    print(f'Start Gemini quality evaluation: method={args.method_name}', flush=True)
    print(f'Results dir: {results_dir}', flush=True)
    print(f'Scenarios dir: {scenarios_dir}', flush=True)
    print(f'Session range: {args.start}-{args.end} (total={total_sessions})', flush=True)
    print(f'Num workers: {args.num_workers}', flush=True)
    print(f'Summary file: {summary_path}', flush=True)
    print('=' * 80, flush=True)
    completed = 0
    failed = 0
    skipped = 0
    with details_path.open('w', encoding='utf-8') as wf:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_sid = {executor.submit(_evaluate_one_session, sid, results_dir, scenarios_dir, suffix, judge_kwargs, args.max_retries, args.retry_sleep_s, args.log_progress, _log): sid for sid in range(args.start, args.end + 1)}
            for future in as_completed(future_to_sid):
                sid = future_to_sid[future]
                sid_str = f'{sid:03d}'
                try:
                    sess_out = future.result()
                except Exception as e:
                    failed += 1
                    print(f'[session {sid_str}] failed with exception: {e}', flush=True)
                    continue
                if sess_out.get('skipped', False):
                    skipped += 1
                    print(f"[session {sid_str}] skipped: {sess_out.get('error')}", flush=True)
                    continue
                if not sess_out.get('ok', False):
                    failed += 1
                    print(f"[session {sid_str}] failed: {sess_out.get('error')}", flush=True)
                    continue
                for r in sess_out['records']:
                    r['method'] = args.method_name
                    wf.write(json.dumps(r, ensure_ascii=False) + '\n')
                _merge_stats(stats, sess_out['stats'])
                completed += 1
                summary_path.write_text(json.dumps(_build_summary(), ensure_ascii=False, indent=2), encoding='utf-8')
                print(f"[session {sid_str}] done ({completed}/{total_sessions}, failed={failed}, skipped={skipped}) rolling_overall={_avg('Overall')} count={stats['Overall']['cnt']}", flush=True)
    summary = _build_summary()
    summary['session_status'] = {'completed': completed, 'failed': failed, 'skipped': skipped, 'total_requested': total_sessions}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('=' * 80)
    print(f'Method: {args.method_name}')
    print(f'Summary JSON: {summary_path}')
    print(f'Details JSONL: {details_path}')
    print('-' * 80)
    print('Gemini Generation Quality (0-10):')
    print(f"Overall         : {summary['scores_0_10']['Overall']}")
    print(f"Generate        : {summary['scores_0_10']['Generate']}")
    print(f"Edit            : {summary['scores_0_10']['Edit']}")
    print(f"Understand/Chat : {summary['scores_0_10']['Understand/Chat']}")
    print('=' * 80)

def main():
    parser = argparse.ArgumentParser(description='Gemini generation quality evaluator for benchmark result directories.')
    parser.add_argument('--method-name', type=str, required=True, help='Method name for output files.')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing sample_XXX_*.json result files.')
    parser.add_argument('--scenarios-dir', type=str, required=True, help='Scenario directory (sample_XXX.jsonl).')
    parser.add_argument('--result-suffix', type=str, default=None, help='Result suffix; auto-detected if omitted.')
    parser.add_argument('--start', type=int, default=1, help='Start sample id.')
    parser.add_argument('--end', type=int, default=50, help='End sample id.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output dir for score files.')
    parser.add_argument('--api-base-url', type=str, default=None)
    parser.add_argument('--app-id', type=str, default=None)
    parser.add_argument('--api-key', '--app-secret', dest='app_secret', type=str, default=None)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--model', type=str, default='gemini-3-pro-preview')
    parser.add_argument('--max-retries', type=int, default=3, help='Retries per turn for Gemini call.')
    parser.add_argument('--retry-sleep-s', type=float, default=2.0, help='Sleep between retries.')
    parser.add_argument('--num-workers', type=int, default=8, help='Max concurrent sessions.')
    parser.add_argument('--log-progress', action=argparse.BooleanOptionalAction, default=True, help='Print per-session and per-turn realtime progress logs.')
    args = parser.parse_args()
    evaluate_method(args)
if __name__ == '__main__':
    main()
