"""Offline budget integration checks; fixture responses never call a model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import async_jobs as a
import client
import host_bridge as bridge
import project_support as p
import server as s
import state_store as state
from test_server import response


class BudgetSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(self.temp.name)
        self.runs = self.root/'runs'
        patch.object(s, 'RUNS_DIR', self.runs).start()
        patch.object(s, 'load_api_key', side_effect=AssertionError('No credentials')).start()
        self.api = patch.object(s, 'call_responses', side_effect=AssertionError('No paid request')).start()

    def call(self, **args):
        return s.handle_tools_call({'name': 'deepseek_subagent', 'arguments': args})

    def start(self, **args):
        return self.call(**{'task_id': 'budget-settings', 'task': 'bounded fixture', 'cwd': str(self.root), **args})

    def receipt(self, result):
        return state.read(result['structuredContent']['receipt_path'])

    def test_default_budget_allows_long_reasoning_then_write_and_report(self):
        captured = []
        first = response(calls=[('write_file', {'path': 'artifact.py', 'content': 'value = 1\n'})], tokens=60000)
        first['usage']['output_tokens_details']['reasoning_tokens'] = 59000
        first['output'].insert(0, {'type': 'reasoning', 'content': [
            {'type': 'reasoning_text', 'text': 'synthetic fixture ' * 13000}]})
        first['usage']['input_tokens'] = 1000
        replies = [first, response('written fixture; controller review required', tokens=1000)]
        def fake(payload, timeout):
            captured.append((copy.deepcopy(payload), timeout))
            return replies.pop(0)
        self.api.side_effect = fake
        result = self.start(allow_write=True)
        self.assertFalse(result['isError'], result)
        self.assertEqual((self.root/'artifact.py').read_text(), 'value = 1\n')
        receipt = self.receipt(result)
        self.assertEqual(receipt['budgets']['input_budget'], 500000)
        self.assertEqual(receipt['budgets']['output_budget'], 128000)
        self.assertEqual(receipt['budgets']['max_output_tokens'], 65536)
        self.assertGreater(receipt['peak_request_bytes'], 192000)
        self.assertEqual([r[0]['max_output_tokens'] for r in captured], [65536, 65536])
        self.assertTrue(all(r[0]['reasoning']['effort'] == 'max' for r in captured))
        self.assertGreater(captured[0][1], 600)
        self.assertLessEqual(captured[0][1], 1200)

    def test_explicit_long_task_reaches_payload_with_matching_total_budget(self):
        self.api.side_effect = [response('fixture complete', tokens=130000)]
        result = self.start(max_output_tokens=131072, output_budget=500000, timeout_sec=3600)
        self.assertFalse(result['isError'], result)
        self.assertEqual(self.api.call_args.args[0]['max_output_tokens'], 131072)
        self.assertEqual(self.receipt(result)['budgets']['output_budget'], 500000)
        self.assertGreater(client.CLIENT_TIMEOUT, s.HARD_TIMEOUT)

    def test_remaining_total_output_still_clamps_long_rounds(self):
        self.api.side_effect = [response(calls=[('list_dir', {})], tokens=100000), response('done')]
        result = self.start(max_output_tokens=131072, output_budget=150000)
        self.assertFalse(result['isError'], result)
        self.assertEqual([x.args[0]['max_output_tokens'] for x in self.api.call_args_list], [131072, 50000])

    def test_cumulative_input_over_old_default_can_complete(self):
        first = response(calls=[('list_dir', {})])
        first['usage']['input_tokens'] = 100000
        second = response('done'); second['usage']['input_tokens'] = 120000
        self.api.side_effect = [first, second]
        result = self.start()
        self.assertFalse(result['isError'], result)
        self.assertEqual(self.receipt(result)['input_tokens'], 220000)

    def test_invalid_large_round_or_task_budget_rejected_before_request(self):
        for args in ({'max_output_tokens': 500000}, {'max_output_tokens': 131073},
                     {'output_budget': 500001}, {'input_budget': 500001}, {'timeout_sec': 3601}):
            with self.subTest(args=args):
                self.assertTrue(self.start(**args)['isError'])
        self.api.assert_not_called()

    def test_context_projection_reserves_output_before_network(self):
        with patch.object(bridge, 'request_sizes', return_value=(880000, 880000)):
            result = self.start(max_output_tokens=131072, output_budget=500000)
        receipt = self.receipt(result)
        self.assertEqual(receipt['status'], 'context_budget')
        projection = receipt['context_projection']
        self.assertEqual(projection['reserved_output_tokens'], 131072)
        self.assertGreater(projection['projected_tokens'], 1000000)
        self.assertTrue(projection['is_estimate'])
        self.api.assert_not_called()

    def test_text_and_image_wire_caps_have_distinct_diagnostics(self):
        for ident, sizes, message in [('text', (s.MAX_REQUEST_BYTES+1, s.MAX_REQUEST_BYTES+1), '文本请求'),
                                      ('image', (1000, bridge.MAX_VISION_REQUEST_BYTES+1), '含图片')]:
            with self.subTest(ident=ident), patch.object(bridge, 'request_sizes', return_value=sizes):
                result = self.start(task_id=ident)
                self.assertEqual(result['structuredContent']['status'], 'context_budget')
                self.assertIn(message, result['content'][0]['text'])
        self.api.assert_not_called()

    def test_large_explicit_conversation_packet_is_accepted(self):
        packet = self.root/'host.json'
        packet.write_text(json.dumps({'schema': 'DEEPSEEK_HOST_V1', 'conversation': [
            {'role': 'user', 'text': 'synthetic context ' * 13000}]}))
        self.assertGreater(packet.stat().st_size, 192000)
        self.api.side_effect = [response('fixture inspected')]
        result = self.start(host_context_path=str(packet))
        self.assertFalse(result['isError'], result)
        self.assertGreater(len(self.api.call_args.args[0]['input'][0]['content']), 192000)

    def test_project_overrides_and_explicit_budget_precedence(self):
        config = self.root/p.CONFIG_NAME
        config.write_text(json.dumps({'schema': 'DEEPSEEK_PROJECT_V1', 'defaults': {
            'max_output_tokens': 100000, 'output_budget': 250000, 'input_budget': 400000,
            'timeout_sec': 1800}}))
        self.api.side_effect = [response('fixture complete')]
        result = self.start(max_output_tokens=65536)
        self.assertFalse(result['isError'], result)
        budgets = self.receipt(result)['budgets']
        self.assertEqual(budgets['max_output_tokens'], 65536)
        self.assertEqual(budgets['output_budget'], 250000)
        self.assertEqual(budgets['input_budget'], 400000)
        self.assertEqual(budgets['timeout_sec'], 1800)
        self.assertNotIn('write_file', [t['name'] for t in self.api.call_args.args[0]['tools']])

    def test_followup_keeps_original_smaller_budgets_after_upgrade(self):
        self.api.side_effect = [response('fixture completed'), response('fixture reviewed')]
        with patch.object(s, 'DEFAULT_MAX_OUTPUT_TOKENS', 8192), patch.object(s, 'DEFAULT_OUTPUT_BUDGET', 32000), \
             patch.object(s, 'DEFAULT_INPUT_BUDGET', 120000), patch.object(s, 'DEFAULT_TIMEOUT', 360):
            first = self.start()  # Original arguments do not contain any budget overrides.
        self.assertFalse(first['isError'], first)
        follow = self.call(action='followup', task_id='budget-settings', followup_id='review-1', instruction='Check fixture')
        self.assertFalse(follow['isError'], follow)
        self.assertEqual(follow['structuredContent']['budgets'], first['structuredContent']['budgets'])
        self.assertEqual(self.api.call_args.args[0]['max_output_tokens'], 8192)
        self.assertEqual(self.receipt(follow)['output_tokens'], 40)

    def test_host_resume_keeps_old_implicit_budgets_after_upgrade(self):
        packet = self.root/'host.json'
        packet.write_text(json.dumps({'schema': 'DEEPSEEK_HOST_V1', 'capabilities': [
            {'id': 'read', 'permission': 'read', 'description': 'Read fixture evidence'}]}))
        self.api.side_effect = [response(calls=[('request_host', {
            'capability': 'read', 'arguments': {'query': 'fixture'}, 'reason': 'need evidence'})]), response('done')]
        with patch.object(s, 'DEFAULT_MAX_OUTPUT_TOKENS', 8192), patch.object(s, 'DEFAULT_OUTPUT_BUDGET', 32000), \
             patch.object(s, 'DEFAULT_INPUT_BUDGET', 120000), patch.object(s, 'DEFAULT_TIMEOUT', 360):
            first = self.start(host_context_path=str(packet))
        self.assertEqual(first['structuredContent']['status'], 'needs_host', first)
        request_id = first['structuredContent']['host_request']['request_id']
        claimed = self.call(action='claim', task_id='budget-settings', request_id=request_id)
        self.assertFalse(claimed['isError'], claimed)
        result_path = self.root/'host-result.json'
        result_path.write_text(json.dumps({'request_id': request_id, 'status': 'completed',
                                           'text': 'actual fixture evidence', 'sources': [], 'images': []}))
        resumed = self.call(action='resume', task_id='budget-settings', host_result_path=str(result_path))
        self.assertFalse(resumed['isError'], resumed)
        self.assertEqual(resumed['structuredContent']['budgets'], first['structuredContent']['budgets'])
        self.assertEqual(self.api.call_args.args[0]['max_output_tokens'], 8192)
        self.assertLess(self.api.call_args.kwargs['timeout'], 360)
        self.assertEqual(self.receipt(resumed)['input_tokens'], 200)

    def test_queue_freezes_global_defaults_before_execution(self):
        prepared = a.prepare({'task_id': 'queued', 'task': 'fixture', 'cwd': str(self.root)}, self.runs)
        with patch.object(s, 'DEFAULT_OUTPUT_BUDGET', 500000), patch.object(s, 'DEFAULT_MAX_OUTPUT_TOKENS', 131072):
            self.api.side_effect = [response('fixture complete')]
            result = self.call(**prepared['arguments'])
        self.assertFalse(result['isError'], result)
        self.assertEqual(self.receipt(result)['budgets']['output_budget'], 128000)
        self.assertEqual(self.api.call_args.args[0]['max_output_tokens'], 65536)


if __name__ == '__main__':
    unittest.main()
