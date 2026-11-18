"""
KV Cache 基准对比脚本（50x20-gemini 场景）
复用 run_kv_cache_baseline.py，改默认场景目录与输出目录
"""
import os
import sys
from pathlib import Path
import run_kv_cache_baseline as base
base.SCENARIOS_DIR = base.BENCHMARK_DIR / 'scenarios' / '50x20-gemini'
_output_dir = os.environ.get('KV_CACHE_OUTPUT_DIR')
base.OUTPUT_DIR = Path(_output_dir) if _output_dir else base.BENCHMARK_DIR / 'kv_cache_baseline_20_results'
base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if __name__ == '__main__':
    sys.exit(base.main())
