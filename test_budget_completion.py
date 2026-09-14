"""Regressions for completed artifacts at input stop and reasoning-only overflow.

All model replies are fixtures. The one validation command is a local synthetic
Python script; no CAD runtime, project data, credentials or live APIs are used.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import project_support as p
import server as s
import state_store as state
from test_server import response


class BudgetCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(self.temp.name)
        patch.object(s, 'RUNS_DIR', self.root/'runs').start()
        patch.object(s, 'load_api_key', side_effect=AssertionError('No credentials')).start()
        self.api = patch.object(s, 'call_responses', side_effect=AssertionError('No extra model request')).start()

    def write(self, name, value):
        path = self.root/name
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    def contract(self, validation=False):
        value = {'schema': 'DEEPSEEK_TASK_V2', 'objective': 'bounded output', 'acceptance': ['review actual results'],
                 'read_paths': [str(self.root)+'/'], 'write_paths': ['artifact.py'], 'commands': [],
                 'deliverables': [{'path': 'artifact.py', 'must_change': True}]}
        if validation:
            script = self.write('validate.py', 'from pathlib import Path\nimport json\n'
                                'count=Path("counter")\ncount.write_text(str(int(count.read_text())+1) if count.exists() else "1")\n'
                                'Path("validation.json").write_text(json.dumps({"passed":Path("artifact.py").read_text()=="value=1"}))\n')
            value['commands'] = [{'id': 'validate', 'argv': [sys.executable, str(script)], 'cwd': str(self.root),
                                  'timeout_sec': 3, 'inputs': [p.pin(script)]}]
            value['write_paths'] += ['validation.json']
            value['checks'] = [{'id': 'validated', 'path': 'validation.json', 'pointer': '/passed', 'expected': True}]
            value['deliverables'] += [{'path': 'validation.json', 'must_change': True}]
        return str(self.write('contract.json', value))

    def call(self, **kwargs):
        return s.handle_tools_call({'name': 'deepseek_subagent', 'arguments': kwargs})

    def start(self, **kwargs):
        return self.call(task_id='budget-task', task='Implement only the fixed interface and validate', cwd=str(self.root),
                         **{'max_steps': 8, 'max_output_tokens': 32768, 'output_budget': 40000, **kwargs})

    def test_input_prediction_runs_local_checks_without_summary_or_command_replay(self):
        contract = self.contract(validation=True)
        first = response(calls=[('write_file', {'path': 'artifact.py', 'content': 'value=1'})])
        second = response(calls=[('run_command', {'id': 'validate'})])
        first['usage']['input_tokens'] = second['usage']['input_tokens'] = 10000
        self.api.side_effect = [first, second]
        with patch.object(s.bridge, 'request_sizes', return_value=(10000, 10000)):
            result = self.start(contract_path=contract, allow_write=True, allow_shell=True, input_budget=25000)
        meta = result['structuredContent']
        self.assertEqual(meta['status'], 'input_budget_prediction', result)
        self.assertTrue(result['isError'])  # Not scientific acceptance or dependency completion.
        self.assertEqual(meta['completion']['verification'], 'passed')
        self.assertEqual(meta['completion']['model_report'], 'not_received')
        self.assertEqual(meta['completion']['controller_acceptance'], 'pending')
        self.assertEqual(meta['local_verification']['checked'], 3)
        self.assertEqual(meta['local_verification']['commands_run'], 0)
        self.assertEqual(self.api.call_count, 2)
        self.assertEqual((self.root/'counter').read_text(), '1')
        self.assertTrue(all(x['passed'] for x in meta['acceptance_checks']))
        self.assertIn('3 项全部通过', result['content'][0]['text'])
        handoff = state.read(meta['handoff_path'])
        self.assertEqual(handoff['completion'], meta['completion'])
        self.assertEqual(handoff['local_verification'], meta['local_verification'])

    def test_actual_input_and_output_budget_stops_also_check_outputs(self):
        contract = self.contract()
        for kind in ('input', 'output'):
            with self.subTest(kind=kind):
                out = self.root/'artifact.py'
                out.unlink(missing_ok=True)
                reply = response(calls=[('write_file', {'path': 'artifact.py', 'content': 'value=1'})], tokens=128)
                reply['usage']['input_tokens'] = 1000
                self.api.side_effect = [reply]
                args = {'output_budget': 128} if kind == 'output' else {'input_budget': 1000}
                result = self.call(task_id='task-'+kind, task='create', cwd=str(self.root), contract_path=contract,
                                   allow_write=True, **args)
                self.assertEqual(result['structuredContent']['status'], kind+'_budget')
                self.assertEqual(result['structuredContent']['completion']['verification'], 'passed')

    def test_missing_deliverable_fails_checks_but_keeps_budget_stop_reason(self):
        reply = response(calls=[('list_dir', {'path': str(self.root)})])
        reply['usage']['input_tokens'] = 1000
        self.api.side_effect = [reply]
        result = self.start(contract_path=self.contract(), allow_write=True, input_budget=1000)
        self.assertEqual(result['structuredContent']['status'], 'input_budget')
        self.assertEqual(result['structuredContent']['completion']['verification'], 'failed')
        self.assertFalse(result['structuredContent']['acceptance_checks'][0]['passed'])

    def test_no_declared_checks_is_not_a_vacuous_pass(self):
        reply = response(calls=[('list_dir', {})]); reply['usage']['input_tokens'] = 1000
        self.api.side_effect = [reply]
        result = self.start(input_budget=1000)
        self.assertEqual(result['structuredContent']['completion']['verification'], 'not_configured')
        self.assertEqual(result['structuredContent']['acceptance_checks'], [])

    def test_unpaired_tools_and_unknown_writes_defer_checks(self):
        contract = self.contract()
        self.api.side_effect = [response(calls=[('write_file', {'path': 'artifact.py', 'content': 'bad'})])]
        with patch.object(p, 'acceptance_results', side_effect=AssertionError('Must not verify unresolved tools')) as check:
            result = self.start(contract_path=contract, allow_write=True, max_steps=1)
            self.assertEqual(result['structuredContent']['completion']['verification'], 'deferred')
            self.assertEqual(result['structuredContent']['local_verification']['reason'], 'unresolved_tool_or_host_action')
            check.assert_not_called()
        self.assertFalse((self.root/'artifact.py').exists())
        self.api.side_effect = [response(calls=[('write_file', {'path': 'artifact.py', 'content': 'bad'})])]
        with patch.object(s, 'dispatch', side_effect=s.DeadlineExceeded('during write')), \
             patch.object(p, 'acceptance_results') as check:
            result = self.call(task_id='unknown', task='create', cwd=str(self.root), allow_write=True, contract_path=contract)
            self.assertEqual(result['structuredContent']['status'], 'timeout')
            self.assertEqual(result['structuredContent']['completion']['verification'], 'deferred')
            check.assert_not_called()

    def test_local_check_timeout_does_not_mark_completed_or_retry_model(self):
        contract = self.contract()
        self.api.side_effect = [response('done')]
        with patch.object(p, 'acceptance_results', side_effect=s.DeadlineExceeded('local check limit')):
            result = self.start(contract_path=contract, allow_write=True)
        self.assertEqual(result['structuredContent']['status'], 'acceptance_deferred')
        self.assertEqual(result['structuredContent']['local_verification']['reason'], 'local_check_time_limit')
        self.assertTrue(result['isError'])
        self.assertEqual(self.api.call_count, 1)

    def incomplete(self, output=32768, reasoning=32768, reason='max_output_tokens'):
        reply = response(status='incomplete', tokens=output)
        reply['usage']['output_tokens_details']['reasoning_tokens'] = reasoning
        reply['incomplete_details'] = {'reason': reason}
        # A truncated write must never be parsed or dispatched, even with a
        # valid prefix. Hidden reasoning text must not enter the handoff either.
        reply['output'] = [{'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'private scratch'}]},
                           {'type': 'function_call', 'call_id': 'bad', 'name': 'write_file',
                            'arguments': '{"path":"artifact.py","content":'}]
        return reply

    def test_reasoning_only_overflow_reports_exact_usage_without_writes_or_retry(self):
        self.write('source.py', 'value=1')
        first = response(calls=[('read_file', {'path': 'source.py'})], tokens=492)
        self.api.side_effect = [first, self.incomplete()]
        result = self.start(contract_path=self.contract(), allow_write=True)
        meta = result['structuredContent']; diag = meta['diagnostic']
        self.assertEqual(meta['status'], 'api_incomplete')
        self.assertEqual(diag['code'], 'reasoning_output_exhausted')
        self.assertEqual(diag['reported_reasoning_tokens'], 32768)
        self.assertEqual(diag['non_reasoning_output_tokens'], 0)
        self.assertEqual(diag['discarded_tool_calls'], 1)
        self.assertEqual(diag['prior_write_or_command_attempts'], 0)
        self.assertEqual(diag['remaining_output_budget'], 6740)
        self.assertEqual(self.api.call_count, 2)
        self.assertFalse((self.root/'artifact.py').exists())
        self.assertIsNone(meta['followup_checkpoint'])
        handoff = state.read(meta['handoff_path'])
        self.assertEqual(handoff['inspection_index'][0]['tool'], 'read_file')
        self.assertTrue(Path(handoff['inspection_index'][0]['result_file']['path']).exists())
        self.assertNotIn('private scratch', json.dumps(handoff))
        self.assertNotIn('private scratch', json.dumps(result))
        self.assertIn('32768 token 全部计入推理', result['content'][0]['text'])

    def test_mixed_output_and_remaining_total_budget_are_distinguished(self):
        first = response(calls=[('list_dir', {})], tokens=492)
        self.api.side_effect = [first, self.incomplete(output=508, reasoning=400)]
        result = self.start(output_budget=1000)
        diag = result['structuredContent']['diagnostic']
        self.assertEqual(diag['code'], 'output_limit_exhausted')
        self.assertEqual(diag['non_reasoning_output_tokens'], 108)
        self.assertEqual(diag['limit_source'], 'remaining_task_output_budget')
        self.assertEqual(diag['request_output_limit'], 508)
        self.assertEqual(diag['remaining_output_budget'], 0)

    def test_overflow_after_prior_write_does_not_claim_task_had_no_effects(self):
        self.api.side_effect = [response(calls=[('write_file', {'path': 'artifact.py', 'content': 'partial'})]),
                               self.incomplete()]
        result = self.start(allow_write=True)
        self.assertEqual(result['structuredContent']['diagnostic']['prior_write_or_command_attempts'], 1)
        self.assertIn('此前已有写入', result['content'][0]['text'])
        self.assertEqual((self.root/'artifact.py').read_text(), 'partial')

    def test_other_incomplete_reason_is_not_labeled_output_exhaustion(self):
        self.api.side_effect = [self.incomplete(reason='other')]
        result = self.start()
        self.assertIsNone(result['structuredContent']['diagnostic'])
        self.assertEqual(result['structuredContent']['status'], 'api_incomplete')

    def test_followup_does_not_reuse_old_pass_after_incomplete_generation(self):
        contract = self.contract()
        self.api.side_effect = [response(calls=[('write_file', {'path': 'artifact.py', 'content': 'value=1'})]),
                               response('done'), self.incomplete()]
        first = self.start(contract_path=contract, allow_write=True)
        self.assertEqual(first['structuredContent']['completion']['verification'], 'passed')
        follow = self.call(action='followup', task_id='budget-task', followup_id='feedback', instruction='inspect feedback')
        self.assertEqual(follow['structuredContent']['completion']['verification'], 'deferred')
        self.assertEqual(follow['structuredContent']['acceptance_checks'], [])

    def test_second_round_nudge_keeps_max_and_names_new_targets(self):
        captured = []
        self.write('source.py', 'known input')
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload))
            return response(calls=[('read_file', {'path': 'source.py'})]) if len(captured) == 1 else response('needs controller split')
        self.api.side_effect = fake
        self.start(contract_path=self.contract(), allow_write=True)
        note = captured[1]['input'][-1]['content']
        self.assertIn('可独立验证', note)
        self.assertIn('artifact.py', note)
        self.assertIn('尚不存在', note)
        self.assertTrue(all(x['reasoning']['effort'] == 'max' for x in captured))


if __name__ == '__main__': unittest.main()
