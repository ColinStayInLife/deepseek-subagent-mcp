"""Offline regression of observed no-output/budget failures; never use an API key."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import execution_support as e
import prepare
import project_support as p
import server as s
from test_server import response


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cwd = str(self.root)
        self.api = patch.object(s, 'call_responses', side_effect=AssertionError('Unexpected API request')).start()
        self.addCleanup(patch.stopall)
        self.kw = dict(task='Implement the specified change', context='', cwd=self.cwd,
                       model='deepseek-flash', effort='max', max_steps=8,
                       allow_write=True, allow_shell=False, timeout_sec=60,
                       record_dir=self.root/'record')

    def write(self, name, text):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def contract(self, **updates):
        value = dict(schema='DEEPSEEK_TASK_V2', objective='bounded fix', acceptance=['actual output'],
                     read_paths=['source.py'], write_paths=['candidate.py'], commands=[])
        value.update(updates)
        path = self.write('contract.json', json.dumps(value))
        return p.load_contract(path, self.cwd)

    def clone_args(self):
        self.write('source.py', 'SCHEMA = "V1"\r\nvalue = 3\r\n')
        return dict(source_path='source.py', path='candidate.py',
                    source_sha256=p.pin(self.root/'source.py')['sha256'],
                    edits=[dict(old_text='"V1"', new_text='"V2"'),
                           dict(old_text='value = 3', new_text='value = 4')])

    def dispatch_clone(self, args, contract=None, allow_write=True):
        return s.dispatch('clone_file', json.dumps(args), self.cwd, allow_write, False, contract)

    def run_mock(self, responses, **kwargs):
        captured = []
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload))
            return responses.pop(0)
        self.api.side_effect = fake
        result = s.run_subagent(**(self.kw | kwargs))
        return result, captured

    def test_clone_exact_output_and_crlf_source_unchanged_without_shell(self):
        args = self.clone_args()
        original = (self.root/'source.py').read_bytes()
        result = json.loads(self.dispatch_clone(args, self.contract()))
        self.assertEqual((self.root/'candidate.py').read_bytes(), original.replace(b'V1', b'V2').replace(b'= 3', b'= 4'))
        self.assertEqual((self.root/'source.py').read_bytes(), original)
        self.assertEqual(result['output'], p.pin(self.root/'candidate.py'))
        self.assertEqual(result['source']['sha256'], args['source_sha256'])
        self.api.assert_not_called()

    def test_clone_permission_source_and_destination_independent(self):
        args = self.clone_args()
        for contract in (self.contract(read_paths=['different.py']), self.contract(write_paths=[])):
            with self.subTest(contract=contract['value']):
                self.assertTrue(self.dispatch_clone(args, contract).startswith('错误：'))
                self.assertFalse((self.root/'candidate.py').exists())

    def test_clone_not_exposed_or_executed_readonly(self):
        args = self.clone_args()
        self.assertNotIn('clone_file', [x['name'] for x in s.tool_schemas(False, False)])
        self.assertTrue(self.dispatch_clone(args, allow_write=False).startswith('错误：'))
        self.assertFalse((self.root/'candidate.py').exists())

    def test_clone_stale_sha_or_pinned_source_denied(self):
        args = self.clone_args()
        contract = self.contract(read_pins=[p.pin(self.root/'source.py')])
        self.write('source.py', 'source changed')
        for c in (None, contract):
            self.assertTrue(self.dispatch_clone(args, c).startswith('错误：'))
        self.assertFalse((self.root/'candidate.py').exists())

    def test_clone_all_edits_validate_before_publishing(self):
        args = self.clone_args()
        args['edits'].append(dict(old_text='missing', new_text='x'))
        self.assertTrue(self.dispatch_clone(args).startswith('错误：'))
        self.assertFalse((self.root/'candidate.py').exists())
        args['edits'] = [dict(old_text='\r\n', new_text='\n')]
        self.assertTrue(self.dispatch_clone(args).startswith('错误：'))
        self.assertFalse((self.root/'candidate.py').exists())

    def test_clone_never_overwrites_or_follows_dangling_destination(self):
        args = self.clone_args()
        target = self.write('candidate.py', 'keep')
        self.assertTrue(self.dispatch_clone(args).startswith('错误：'))
        self.assertEqual(target.read_text(), 'keep')
        target.unlink(); target.symlink_to(self.root/'outside.py')
        self.assertTrue(self.dispatch_clone(args).startswith('错误：'))
        self.assertFalse((self.root/'outside.py').exists())

    def test_clone_atomic_publish_loses_race_without_overwrite_or_temp_leak(self):
        args = self.clone_args()
        real_link = e.os.link
        def race(source, target):
            Path(target).write_text('another writer')
            real_link(source, target)
        with patch.object(e.os, 'link', side_effect=race):
            self.assertTrue(self.dispatch_clone(args).startswith('错误：'))
        self.assertEqual((self.root/'candidate.py').read_text(), 'another writer')
        self.assertEqual(list(self.root.glob('.deepseek-clone-*')), [])

    def test_source_change_during_clone_is_not_published(self):
        args = self.clone_args()
        with patch.object(e.project, 'pin', return_value={'sha256': 'changed'}):
            self.assertTrue(self.dispatch_clone(args).startswith('错误：'))
        self.assertFalse((self.root/'candidate.py').exists())

    def test_source_symlink_outside_contract_is_rejected(self):
        args = self.clone_args()
        original = self.root/'source.py'
        original.rename(self.root/'outside.py')
        self.write('source.py', 'permitted')
        contract = self.contract()
        original.unlink(); original.symlink_to(self.root/'outside.py')
        self.assertTrue(self.dispatch_clone(args, contract).startswith('错误：'))
        self.assertFalse((self.root/'candidate.py').exists())

    def test_clone_incomplete_api_does_not_execute_even_valid_call(self):
        args = self.clone_args()
        (_, stats), captured = self.run_mock([
            response('partial', [('clone_file', args)], status='incomplete', tokens=8192)])
        self.assertEqual(stats['status'], 'api_incomplete')
        self.assertEqual(stats['progress']['write_attempts'], 0)
        self.assertFalse((self.root/'candidate.py').exists())
        self.assertEqual(len(captured), 1)
        self.assertEqual(stats['round_records'][0]['status'], 'incomplete')

    def test_missing_clone_source_is_correctable_tool_error_not_task_crash(self):
        args = self.clone_args()
        c = self.contract(read_paths=['source.py', 'missing.py'],
                          deliverables=[dict(path='candidate.py', must_change=True)])
        (_, stats), _ = self.run_mock([
            response(calls=[('clone_file', {**args, 'source_path': 'missing.py'})]),
            response(calls=[('clone_file', args)]), response('done')], contract=c)
        self.assertEqual(stats['status'], 'completed')
        self.assertEqual(stats['tool_errors'], 1)
        self.assertEqual(stats['progress']['writes_returned'], 1)
        self.assertEqual(stats['action_records'][0]['status'], 'tool_error')

    def test_progress_pins_actual_clone_and_deliverable_acceptance(self):
        args = self.clone_args()
        contract = self.contract(deliverables=[dict(path='candidate.py', must_change=True)])
        (_, stats), captured = self.run_mock([
            response(calls=[('clone_file', args)]), response('done')], contract=contract)
        self.assertEqual(stats['status'], 'completed')
        self.assertEqual(stats['progress']['writes_returned'], 1)
        self.assertEqual(stats['action_records'][0]['output_file'], p.pin(self.root/'candidate.py'))
        self.assertTrue(stats['acceptance_checks'][0]['passed'])
        self.assertTrue(all(x['reasoning'] == {'effort': 'max'} for x in captured))

    def test_missing_or_empty_deliverable_cannot_claim_completed(self):
        contract = self.contract(deliverables=[dict(path='candidate.py', must_change=True)])
        for content in (None, ''):
            if content is not None: self.write('candidate.py', content)
            (_, stats), _ = self.run_mock([response('all done')], contract=contract)
            self.assertEqual(stats['status'], 'acceptance_failed')
            self.assertFalse(stats['acceptance_checks'][0]['passed'])

    def test_unchanged_file_rejected_unless_contract_explicitly_allows_it(self):
        self.write('candidate.py', 'existing output')
        for must_change, expected in ((True, 'acceptance_failed'), (False, 'completed')):
            c = self.contract(deliverables=[dict(path='candidate.py', must_change=must_change)])
            (_, stats), _ = self.run_mock([response('done')], contract=c)
            self.assertEqual(stats['status'], expected)

    def test_changed_deliverable_passes_without_loosening_other_checks(self):
        self.write('candidate.py', 'old')
        self.write('report.json', '{"native_ok":false}')
        c = self.contract(read_paths=['candidate.py', 'report.json'],
                          deliverables=[dict(path='candidate.py', must_change=True)],
                          checks=[dict(id='native', path='report.json', pointer='/native_ok', expected=True)])
        (_, stats), _ = self.run_mock([response(calls=[('edit_file', dict(path='candidate.py', old_text='old', new_text='new'))]), response('done')], contract=c)
        self.assertEqual(stats['status'], 'acceptance_failed')
        self.assertFalse(stats['acceptance_checks'][0]['passed'])
        self.assertTrue(stats['acceptance_checks'][1]['passed'])

    def test_invalid_deliverables_reject_before_api(self):
        for item in ({'path': 'outside.py', 'must_change': True},
                     {'path': 'candidate.py', 'must_change': 'true'},
                     {'path': 'candidate.py/' , 'must_change': True}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.contract(deliverables=[item])
        self.api.assert_not_called()

    def test_prepare_normalizes_deliverables_without_model(self):
        value = prepare.build_contract(dict(objective='fix', acceptance=['actual file'],
            read_paths=['source.py'], write_paths=['candidate.py'],
            deliverables=[dict(path='candidate.py', must_change=True)]), self.cwd)
        self.assertEqual(value['deliverables'][0]['path'], str(self.root/'candidate.py'))
        self.api.assert_not_called()

    def test_input_budget_does_not_force_report_when_one_useful_request_fits(self):
        first = response(calls=[('list_dir', {})]); first['usage']['input_tokens'] = 30000
        second = response(calls=[('write_file', dict(path='candidate.py', content='actual'))])
        second['usage']['input_tokens'] = 30000
        # Old heuristic forced report at 50k remaining because it reserved >77k.
        with patch.object(s.bridge, 'request_sizes', return_value=(10000, 10000)):
            (_, stats), captured = self.run_mock([first, second], input_budget=80000)
        self.assertEqual(captured[1]['tool_choice'], 'auto')
        self.assertEqual((self.root/'candidate.py').read_text(), 'actual')
        self.assertEqual(stats['status'], 'input_budget_prediction')
        self.assertEqual(len(captured), 2)  # no extra paid report
        self.assertEqual(stats['input_tokens'], 60000)
        self.assertEqual(stats['next_input_estimate'], 34756)

    def test_estimate_tracks_actual_request_growth_and_includes_cached_input(self):
        first = response(calls=[('list_dir', {})]); first['usage']['input_tokens'] = 10000
        first['usage']['input_tokens_details']['cached_tokens'] = 9000
        with patch.object(s.bridge, 'request_sizes', side_effect=[(10000, 10000), (30000, 30000)]):
            (_, stats), captured = self.run_mock([first], input_budget=40000)
        self.assertEqual(stats['status'], 'input_budget_prediction')
        self.assertEqual(stats['input_tokens'], 10000)
        self.assertEqual(stats['next_input_estimate'], 34756)
        self.assertEqual(len(captured), 1)
        self.assertIsNone(e.next_input_estimate(100, 0, 50))
        self.assertEqual(e.next_input_estimate(100, 1000, 10000, True), 11756)

    def test_budget_and_no_write_nudge_are_ephemeral_and_do_not_change_effort(self):
        self.write('source.py', 'evidence')
        (_, stats), captured = self.run_mock([
            response(calls=[('read_file', {'path':'source.py'})]),
            response(calls=[('file_info', {'path':'source.py'})]), response('cannot proceed')])
        note = captured[2]['input'][-1]['content']
        self.assertIn('尚无成功写入', note)
        self.assertIn('不猜参数', note)
        self.assertTrue(all(x['reasoning']['effort'] == 'max' for x in captured))
        self.assertEqual(sum('执行预算：' in str(x) for x in captured[2]['input']), 1)
        self.assertEqual(stats['progress']['write_attempts'], 0)
        self.assertEqual(len(stats['round_records']), 3)

    def test_duplicate_read_revalidates_and_preserves_first_full_result(self):
        self.write('source.py', 'evidence line\n' * 100)
        real_dispatch = s.dispatch
        with patch.object(s, 'dispatch', wraps=real_dispatch) as dispatch:
            (_, stats), captured = self.run_mock([
                response(calls=[('read_file', {'path':'source.py'})]),
                response(calls=[('read_file', {'path':'source.py'})]), response('done')])
        self.assertEqual(dispatch.call_count, 2)
        outputs = [x['output'] for x in captured[2]['input'] if x.get('type') == 'function_call_output']
        self.assertIn('evidence line', outputs[0])
        self.assertTrue(outputs[1].startswith('UNCHANGED_READ'))
        self.assertGreater(stats['duplicate_read_chars_avoided'], 1000)
        self.assertEqual(stats['action_records'][0]['result_sha256'], stats['action_records'][1]['result_sha256'])

    def test_changed_read_is_never_compacted(self):
        self.write('source.py', 'evidence\n' * 100)
        (_, stats), captured = self.run_mock([
            response(calls=[('read_file', {'path':'source.py'})]),
            response(calls=[('write_file', dict(path='source.py', content='NEW\n' * 100))]),
            response(calls=[('read_file', {'path':'source.py'})]), response('done')])
        outputs = [x['output'] for x in captured[3]['input'] if x.get('type') == 'function_call_output']
        self.assertIn('NEW', outputs[-1])
        self.assertNotIn('UNCHANGED_READ', outputs[-1])
        self.assertEqual(stats['duplicate_read_chars_avoided'], 0)

    def test_receipt_saved_before_unknown_mutation_and_no_retry(self):
        captured_receipt = {}
        def interrupted(*args, **kwargs):
            captured_receipt.update(json.loads((self.root/'record/receipt.json').read_text()))
            raise RuntimeError('interrupted')
        with patch.object(s, 'dispatch', side_effect=interrupted):
            (_, stats), captured = self.run_mock([
                response(calls=[('write_file', dict(path='candidate.py', content='value'))])])
        self.assertEqual(captured_receipt['progress']['writes_unknown'], 1)
        self.assertEqual(stats['status'], 'error')
        self.assertEqual(stats['progress']['writes_unknown'], 1)
        self.assertEqual(len(captured), 1)

    def test_resume_keeps_deliverable_baseline_and_input_calibration(self):
        self.write('candidate.py', 'before')
        c = self.contract(deliverables=[dict(path='candidate.py', must_change=True)])
        packet = {'messages': [], 'pin': None, 'images': [],
                  'capabilities': [{'id':'read', 'description':'read real evidence', 'permission':'read'}]}
        (_, first), _ = self.run_mock([response(calls=[('request_host', dict(capability='read', arguments={}, reason='need evidence'))])], contract=c, host_packet=packet)
        self.assertEqual(first['status'], 'needs_host')
        continuation = json.loads((self.root/'record/continuation.json').read_text())
        self.assertGreater(continuation['last_request_bytes'], 0)
        initial_sha = continuation['stats']['deliverable_baseline']['candidate.py']['sha256']
        self.write('candidate.py', 'after host work')
        (_, stats), _ = self.run_mock([response('done')], contract=c, host_packet=packet,
            resume_state=continuation, resume_items=[{'type':'function_call_output','call_id':'call_0','output':'actual evidence'}])
        self.assertEqual(stats['status'], 'completed')
        self.assertEqual(stats['deliverable_baseline']['candidate.py']['sha256'], initial_sha)
        self.assertEqual(stats['input_tokens'], 200)
        self.assertEqual(len(stats['round_records']), 2)
        self.assertTrue(stats['acceptance_checks'][0]['changed'])


if __name__ == '__main__':
    unittest.main()
