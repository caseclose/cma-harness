"""
运行真实 Agent 完成对话，不进行评估
只执行对话并保存结果，供后续批量评估使用

用法:
    python run_agent_conversation.py --scenario 50x10-gemini/sample_001.jsonl
    python run_agent_conversation.py --scenario 50x10-gemini/sample_001.jsonl --output-dir agent_conversations
    python run_agent_conversation.py --start 1 --end 50 --max-workers 4
"""
import argparse
import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent / 'ms-swift'
sys.path.insert(0, str(PROJECT_ROOT))
SCENARIOS_DIR = BENCHMARK_DIR / 'scenarios' / '50x10-gemini'
OUTPUT_DIR = BENCHMARK_DIR / 'agent_conversations'
OUTPUT_DIR.mkdir(exist_ok=True)
_shared_agent = None
_agent_lock = threading.Lock()

def init_shared_agent():
    """在主进程中初始化共享的 Agent 实例"""
    global _shared_agent
    if _shared_agent is None:
        with _agent_lock:
            if _shared_agent is None:
                print('=' * 80)
                print('初始化共享 Agent 实例...')
                print('=' * 80)
                try:
                    from multimodal_agent_service import MultimodalAgentService
                except ImportError as e:
                    print(f'Failed to import MultimodalAgentService: {e}')
                    sys.exit(1)
                vlm_a_gpus = os.environ.get('VLM_A_GPUS', '0')
                vlm_b_gpus = os.environ.get('VLM_B_GPUS', '1')
                vlm_c_gpus = os.environ.get('VLM_C_GPUS', '2')
                _shared_agent = MultimodalAgentService(vlm_a_gpus=[int(x) for x in vlm_a_gpus.split(',')], vlm_b_gpus=[int(x) for x in vlm_b_gpus.split(',')], vlm_c_gpus=[int(x) for x in vlm_c_gpus.split(',')])
                print('✓ Agent 初始化完成')
                print('=' * 80)
                print()
    return _shared_agent

@dataclass
class TurnConversation:
    """单轮对话结果"""
    turn: int
    user_input: str
    task_type: str
    agent_output: Dict
    history_images_count: int
    history_messages_count: int

def run_single_conversation(sample_id: int, output_dir: Path=None, scenario_path_override: Optional[Path]=None, scenario_label: Optional[str]=None) -> Tuple[int, bool, str, float]:
    """
    运行单个 sample 的对话（不进行评估）
    
    Args:
        sample_id: 测试样例 ID (1-50)
        output_dir: 输出目录
    
    Returns:
        (sample_id, success, message, duration)
    """
    start_time = time.time()
    sample_id_str = f'{sample_id:03d}'
    scenario_file = scenario_label or f'50x10-gemini/sample_{sample_id_str}.jsonl'
    scenario_path = scenario_path_override or SCENARIOS_DIR / f'sample_{sample_id_str}.jsonl'
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if not scenario_path.exists():
        return (sample_id, False, f'场景文件不存在: {scenario_path}', 0)
    try:
        agent = init_shared_agent()
        session_id = agent.create_session()
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
        history_messages_count = 0
        generated_images = []
        path_mapping = {}
        print(f'\n[Sample {sample_id_str}] Starting conversation ({len(turns_data)} turns)...')
        for turn_data in turns_data:
            turn_idx = turn_data['turn']
            user_input = turn_data['input']
            meta = turn_data.get('meta', {})
            task_type = meta.get('task_type', 'generate')
            print(f'[Sample {sample_id_str}] Turn {turn_idx}/{len(turns_data)}: {task_type}')
            history_images_count = len(generated_images)
            history_messages_count_before = history_messages_count
            if task_type == 'understand':
                result = agent.understand_task(session_id, user_input, None)
                agent_output = {'text': result.get('answer', ''), 'image_path': None, 'qwen3vl_selected_index': result.get('retrieved_image_ids', []) or None, 'qwen3vl_refine_prompt': None}
            else:
                action_type = 'edit' if task_type == 'edit' else 'generate'
                result = agent.generate_image(session_id=session_id, user_instruction=user_input, action_type=action_type, use_vlm_b_retrieval=True)
                agent_original_path = result.get('image_path') or result.get('data', {}).get('local_path')
                benchmark_path = None
                if agent_original_path and os.path.exists(agent_original_path):
                    image_filename = f'{turn_idx:02d}.png'
                    benchmark_path = sample_output_dir / image_filename
                    shutil.copy(agent_original_path, str(benchmark_path))
                    agent_abs_path = os.path.abspath(agent_original_path)
                    benchmark_abs_path = os.path.abspath(str(benchmark_path))
                    path_mapping[agent_abs_path] = benchmark_abs_path
                    path_mapping[agent_original_path] = benchmark_abs_path
                    generated_images.append({'turn': turn_idx, 'rank': len(generated_images), 'agent_original_path': agent_original_path, 'benchmark_path': str(benchmark_path)})
                else:
                    print(f'  [Warning] Agent did not return valid image path: {agent_original_path}')
                image_path_rel = None
                if benchmark_path:
                    image_path_rel = os.path.relpath(str(benchmark_path), output_dir)
                agent_output = {'text': f'用户指令: {user_input}', 'image_path': str(benchmark_path) if benchmark_path else None, 'qwen3vl_selected_index': result.get('retrieved_image_ids', []) or None, 'qwen3vl_refine_prompt': None}
            history_messages_count = history_messages_count_before + 2
            conversation_results.append(TurnConversation(turn=turn_idx, user_input=user_input, task_type=task_type, agent_output=agent_output, history_images_count=history_images_count, history_messages_count=history_messages_count_before))
        output_file = output_dir / f'sample_{sample_id_str}_conversation.json'
        output_data = {'scenario_file': scenario_file, 'sample_id': sample_id, 'timestamp': datetime.now().isoformat(), 'method': 'agent_conversation', 'description': '真实 Agent 对话输出（不评估）', 'turns': [{'turn': turn.turn, 'user_input': turn.user_input, 'task_type': turn.task_type, 'agent_output': turn.agent_output, 'history_images_count': turn.history_images_count, 'history_messages_count': turn.history_messages_count} for turn in conversation_results]}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        duration = time.time() - start_time
        return (sample_id, True, f'成功，结果保存到: {output_file.name}', duration)
    except Exception as e:
        duration = time.time() - start_time
        import traceback
        error_msg = f'异常: {str(e)}\n{traceback.format_exc()}'
        return (sample_id, False, error_msg, duration)

def main():
    parser = argparse.ArgumentParser(description='运行真实 Agent 完成对话（不进行评估）', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\n示例:\n  # 运行单个 sample\n  python run_agent_conversation.py --scenario 50x10-gemini/sample_001.jsonl\n  \n  # 运行多个 sample（并行）\n  python run_agent_conversation.py --start 1 --end 10 --max-workers 4\n  \n  # 指定输出目录\n  python run_agent_conversation.py --start 1 --end 5 --output-dir my_conversations\n        ')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scenario', type=str, help='单个场景文件（如: 50x10-gemini/sample_001.jsonl）')
    group.add_argument('--start', type=int, help='起始 sample ID（与 --end 一起使用）')
    parser.add_argument('--end', type=int, help='结束 sample ID（与 --start 一起使用）')
    parser.add_argument('--max-workers', type=int, default=1, help='最大并行数（默认: 1，串行执行）')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录（默认: agent_conversations/）')
    parser.add_argument('--scenarios-dir', type=str, default=None, help='场景目录（默认: scenarios/50x10-gemini）')
    args = parser.parse_args()
    global SCENARIOS_DIR
    if args.scenarios_dir:
        scenarios_dir = Path(args.scenarios_dir)
        if not scenarios_dir.is_absolute():
            scenarios_dir = (BENCHMARK_DIR / scenarios_dir).resolve()
        SCENARIOS_DIR = scenarios_dir
    scenario_path_override = None
    scenario_label = None
    if args.scenario:
        scenario_path_override = Path(args.scenario)
        if not scenario_path_override.is_absolute():
            scenario_path_override = (BENCHMARK_DIR / scenario_path_override).resolve()
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
            scenario_path = SCENARIOS_DIR / f'sample_{sample_id_str}.jsonl'
            if not scenario_path.exists():
                print(f'⚠ 跳过 sample_{sample_id_str}: 场景文件不存在')
                continue
            valid_samples.append(sample_id)
    if not valid_samples:
        print('❌ 没有有效的测试样例可运行')
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 80)
    print('运行 Agent 对话（不进行评估）')
    print('=' * 80)
    print(f'测试样例: {len(valid_samples)} 个 (sample_{min(valid_samples):03d} 到 sample_{max(valid_samples):03d})')
    print(f'并行数: {args.max_workers}')
    print(f'输出目录: {output_dir}')
    print('=' * 80)
    print()
    print('预初始化 Agent...')
    init_shared_agent()
    print()
    start_time = time.time()
    results = []
    if args.max_workers == 1:
        for sample_id in valid_samples:
            result = run_single_conversation(sample_id, output_dir, scenario_path_override=scenario_path_override, scenario_label=scenario_label)
            results.append(result)
            sample_id, success, message, duration = result
            status = '✅' if success else '❌'
            print(f'{status} [{len(results)}/{len(valid_samples)}] Sample {sample_id:03d}: {message} ({duration:.1f}s)')
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_sample = {executor.submit(run_single_conversation, sample_id, output_dir, scenario_path_override=scenario_path_override, scenario_label=scenario_label): sample_id for sample_id in valid_samples}
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
    print('对话总结')
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
            output_file = output_dir / f'sample_{sample_id_str}_conversation.json'
            if output_file.exists():
                print(f'  - Sample {sample_id_str}: {output_file.name}')
    print('=' * 80)
if __name__ == '__main__':
    main()
