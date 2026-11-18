"""CPU-only release checks. No real models, benchmark data or HTTP calls."""
import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark'))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


api = load('cma_api_client', 'ms-swift/cma_api_client.py')
metric = load('retrieval_metric', 'benchmark/calculate_retrieval_accuracy_tags_desc.py')
judge = load('generation_judge', 'benchmark/evaluate_gemini_generation_quality_main.py')
eval_utils = load('public_eval_utils', 'benchmark/eval_utils.py')


class RetrievalRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only the plugin registration interface is needed for reward arithmetic.
        plugin = types.ModuleType('swift.plugin')
        plugin.ORM = object
        plugin.orms = {}
        with patch.dict(sys.modules, {'swift': types.ModuleType('swift'), 'swift.plugin': plugin}):
            cls.reward = load('retrieval_reward', 'ms-swift/examples/train/grpo/plugin/retrieval_reward_300.py')

    def test_empty_and_exact(self):
        fn = self.reward.RetrievalSoftMatchReward()
        self.assertEqual(fn(['null', '[2, 1]'], [[], [1, 2]]), [1.0, 1.0])

    def test_partial_and_hard_weight(self):
        fn = self.reward.RetrievalSoftMatchReward()
        values = fn(['[1]', '[1]', '[3]'], [[1, 2], [1, 2], [1]], ['easy', 'very_hard', 'hard'])
        self.assertAlmostEqual(values[0], 0.5)
        self.assertAlmostEqual(values[1], 0.6)
        self.assertEqual(values[2], 0.0)

    def test_normalize_text_and_empty_overlap(self):
        self.assertEqual(self.reward._normalize_ids('IDs: 3, 1'), [1, 3])
        self.assertEqual(self.reward._jaccard([], []), 1.0)


class RetrievalMetricTests(unittest.TestCase):
    def compare(self, turns, scenarios, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            results, scenario = Path(d) / 'results.json', Path(d) / 'scenario.jsonl'
            results.write_text(json.dumps({'turns': turns}))
            scenario.write_text('\n'.join(json.dumps(x) for x in scenarios))
            return metric.compare_sample(results, scenario, **kwargs)

    def test_null_prediction_and_skipped_target(self):
        turns = [{'turn': 1, 'retrieved_ranks': None}, {'turn': 2}, {'turn': 3}]
        scenarios = [{'turn': 1, 'meta': {'retrieval_target': []}}, {'turn': 2, 'meta': {'retrieval_target': None}}, {'turn': 3, 'meta': {}}]
        self.assertEqual(self.compare(turns, scenarios), (1, 1, 0))

    def test_sorted_exact_match_and_range(self):
        turns = [{'turn': 10, 'retrieved_ranks': [2, 1]}, {'turn': 11, 'agent_output': {'retrieved_ranks': [2, 1]}}, {'turn': 20, 'retrieved_ranks': [1]}]
        scenarios = [{'turn': t, 'meta': {'retrieval_target': [1, 2]}} for t in [10, 11, 20]]
        self.assertEqual(self.compare(turns, scenarios, turn_start=11, turn_end=20), (2, 1, 0))


class PublicAdapterTests(unittest.TestCase):
    def test_mocked_request_and_text_response(self):
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': 'answer'}}]}
        client = api.CompatibleChatClient(api_base_url='https://provider.example/v1', api_key='unit-test-key', model='judge-model')
        with patch.object(api.requests, 'post', return_value=response) as post:
            result = client.chat('question', images=['https://images.example/test.png'])
        self.assertEqual(client.extract_response_text(result), 'answer')
        self.assertEqual(post.call_args.args[0], 'https://provider.example/v1/chat/completions')
        self.assertEqual(post.call_args.kwargs['json']['messages'][0]['content'][1]['type'], 'image_url')
        self.assertEqual(post.call_args.kwargs['headers']['Authorization'], 'Bearer unit-test-key')
        response.raise_for_status.assert_called_once()

    def test_missing_endpoint_fails_before_network(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(api.requests, 'post') as post:
            with self.assertRaises(ValueError):
                api.CompatibleChatClient().chat('question')
            post.assert_not_called()

    def test_response_blocks_and_image_inputs(self):
        from PIL import Image
        data = {'choices': [{'message': {'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}}]}
        self.assertEqual(api.CompatibleChatClient.extract_response_text(data), 'ab')
        image = Image.new('RGB', (2, 2), 'red')
        self.assertTrue(api.CompatibleChatClient.image_url(image).startswith('data:image/png;base64,'))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'image.png'
            image.save(p)
            self.assertEqual(api.CompatibleChatClient.image_url(p), api.CompatibleChatClient.image_url('file://' + str(p)))

    def test_judge_returns_string(self):
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': 'Score: 8'}}]}
        with patch.object(api.requests, 'post', return_value=response):
            self.assertEqual(api.GeminiJudge(api_base_url='https://provider.example/v1').chat('q'), 'Score: 8')

    def test_scenario_image_adapter_reads_generated_file(self):
        clients = load('public_scenario_clients', 'gemini_multi_agent/api_clients.py')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'result.png'
            p.write_bytes(b'synthetic-image')
            generator = Mock()
            generator.generate.return_value = {'data': {'local_path': str(p)}}
            fake = types.ModuleType('qwen_lightning_generator')
            fake.QwenImageGenerator = Mock(return_value=generator)
            fake.QwenImageEditGenerator = Mock(return_value=generator)
            with patch.dict(sys.modules, {'qwen_lightning_generator': fake}):
                client = clients.GeminiImageGenerator()
                self.assertEqual(client.generate('new image'), b'synthetic-image')
                self.assertEqual(client.generate('edit image', images=['input.png']), b'synthetic-image')


class EvaluationAndWiringTests(unittest.TestCase):
    def test_score_parser_boundaries(self):
        self.assertEqual(judge._parse_score_block('Score (0-10): 10')[0], 10)
        self.assertEqual(judge._parse_score_block('Score: 0')[0], 0)
        self.assertIsNone(judge._parse_score_block('Score: 11')[0])
        self.assertIsNone(judge._parse_score_block('not a score')[0])

    def test_legacy_evaluation_json_parser(self):
        client = Mock()
        client.chat.return_value = '{"total_score": 4}'
        self.assertEqual(eval_utils.evaluate_chat(client, 'q', 'a'), {'total_score': 4})

    def test_api_constructor_keywords_match_service(self):
        service_tree = ast.parse((ROOT / 'ms-swift/multimodal_agent_service.py').read_text())
        cls = next(n for n in service_tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultimodalAgentService')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        accepted = {a.arg for a in init.args.args}
        api_tree = ast.parse((ROOT / 'ms-swift/agent_api_server.py').read_text())
        calls = [n for n in ast.walk(api_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == cls.name]
        self.assertTrue(calls)
        for call in calls:
            self.assertLessEqual({k.arg for k in call.keywords}, accepted)

    def test_baseline_discovers_exported_service(self):
        baseline = load('public_baseline', 'benchmark/run_kv_cache_baseline.py')
        self.assertEqual(baseline._find_ms_swift_root(), (ROOT / 'ms-swift').resolve())


if __name__ == '__main__':
    unittest.main()
