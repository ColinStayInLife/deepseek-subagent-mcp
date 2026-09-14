"""Offline regression of large reports, scoped writes, pinned execution and receipts."""
import copy
import hashlib
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server as s
import project_support as p
from test_server import response


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cwd = str(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def contract(self, commands=None):
        value = {'schema': 'DEEPSEEK_TASK_V1', 'objective': 'Inspect pinned evidence',
                 'acceptance': ['check real result'], 'read_paths': ['input.json', 'src/'],
                 'write_paths': ['new.py', 'out/'], 'commands': commands or []}
        path = self.root / 'contract.json'; path.write_text(json.dumps(value))
        return p.load_contract(path, self.cwd)

    def command(self, text='print("done")'):
        path = self.root / 'job.py'; path.write_text(text)
        return {'id': 'test', 'argv': [sys.executable, str(path)], 'cwd': self.cwd,
                'timeout_sec': 3, 'inputs': [p.pin(path)]}

    def test_large_single_line_json_field_extraction_and_hash(self):
        path = self.root / 'input.json'
        path.write_text(json.dumps({'error': 'physical tolerance failed', 'huge': ['x' * 1000] * 800,
                                    'nested': {'a/b': {'~key': 42}}}))
        value = json.loads(p.tool_json_query({'path': 'input.json', 'pointers': ['/error', '/nested/a~1b/~0key']}, self.cwd))
        self.assertEqual(value['selection']['/error'], 'physical tolerance failed')
        self.assertEqual(value['selection']['/nested/a~1b/~0key'], 42)
        self.assertLess(len(json.dumps(value)), 300)
        self.assertEqual(json.loads(p.tool_file_info({'path': 'input.json'}, self.cwd))['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIn('json_query', s.tool_read_file({'path': 'input.json'}, self.cwd))

    def test_json_pagination_and_missing_fields_are_explicit(self):
        (self.root / 'input.json').write_text(json.dumps({'nodes': list(range(100))}))
        v = json.loads(p.tool_json_query({'path': 'input.json', 'pointers': ['/nodes', '/absent'], 'offset': 7, 'limit': 3}, self.cwd))['selection']
        self.assertEqual(v['/nodes']['items'], [7, 8, 9])
        self.assertEqual(v['/nodes']['count'], 100)
        self.assertEqual(v['/nodes']['next_offset'], 10)
        self.assertEqual(v['/absent'], {'error': 'pointer not found'})
        with self.assertRaises(ValueError): p.json_pointer({}, '/bad~2escape')
        with self.assertRaises(ValueError): p.json_pointer([1], '/-1')
        with self.assertRaises(ValueError): p.read_json(self.root / 'input.json', 10)

    def test_contract_blocks_traversal_symlinks_and_unlisted_writes(self):
        contract = self.contract()
        (self.root / 'src').mkdir()
        outside = self.root / 'secret'; outside.write_text('untouched')
        (self.root / 'src' / 'link').symlink_to(outside)
        for name, args in [('read_file', {'path': 'src/link'}), ('read_file', {'path': 'src/../secret'}),
                           ('write_file', {'path': 'secret', 'content': 'wrong'}),
                           ('run_shell', {'command': 'touch secret'})]:
            result = s.dispatch(name, json.dumps(args), self.cwd, True, True, contract)
            self.assertTrue(result.startswith('错误：'), result)
        self.assertEqual(outside.read_text(), 'untouched')
        result = s.dispatch('write_file', json.dumps({'path': 'new.py', 'content': 'x=1'}), self.cwd, True, False, contract)
        self.assertTrue(result.startswith('已写入'))
        self.assertEqual((self.root / 'new.py').read_text(), 'x=1')

    def test_changed_contract_and_source_prevent_execution(self):
        command = self.command('from pathlib import Path; Path("marker").touch()')
        contract = self.contract([command])
        (self.root / 'job.py').write_text('changed')
        with self.assertRaises(ValueError): p.run_command({'id': 'test'}, contract, set(), self.root / 'logs')
        self.assertFalse((self.root / 'marker').exists())
        (self.root / 'contract.json').write_text('{}')
        with self.assertRaises(ValueError): p.check_access('read_file', {'path': 'input.json'}, self.cwd, contract)

    def test_approved_command_logs_and_no_retry_after_failure(self):
        contract = self.contract([self.command('import sys; print("diagnostic evidence"); sys.exit(3)')])
        attempted = set()
        result = p.run_command({'id': 'test'}, contract, attempted, self.root / 'logs')
        self.assertTrue(result.startswith('错误：'))
        value = json.loads(result.split('\n', 1)[1])
        self.assertEqual(value['exit_code'], 3)
        self.assertIn('diagnostic evidence', Path(value['stdout']['path']).read_text())
        with self.assertRaises(ValueError): p.run_command({'id': 'test'}, contract, attempted, self.root / 'retry')
        self.assertFalse((self.root / 'retry').exists())

    def test_profile_budget_defaults_do_not_grant_permissions(self):
        config = {'schema': 'DEEPSEEK_PROJECT_V1', 'instructions': 'retain original geometry',
                  'require_contract_for_shell': True, 'require_contract_for_write': True,
                  'tool_output_chars': 3000, 'defaults': {'max_steps': 20, 'input_budget': 240000}}
        (self.root / p.CONFIG_NAME).write_text(json.dumps(config))
        sub = self.root / 'nested'; sub.mkdir()
        captured = []
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload)); return response('done')
        with patch.object(s, 'RUNS_DIR', self.root / 'runs'), patch.object(s, 'call_responses', side_effect=fake):
            denied = s.handle_tools_call({'name': 'deepseek_subagent', 'arguments': {'task': 'x', 'cwd': str(sub), 'allow_shell': True}})
            self.assertTrue(denied['isError']); self.assertEqual(captured, [])
            denied = s.handle_tools_call({'name': 'deepseek_subagent', 'arguments': {'task': 'x', 'cwd': str(sub), 'allow_write': True}})
            self.assertTrue(denied['isError']); self.assertEqual(captured, [])
            good = s.handle_tools_call({'name': 'deepseek_subagent', 'arguments': {'task': 'x', 'cwd': str(sub), 'max_steps': 1}})
        self.assertFalse(good['isError'])
        names = {t['name'] for t in captured[0]['tools']}
        self.assertNotIn('run_shell', names); self.assertNotIn('write_file', names)
        self.assertIn('json_query', names)
        self.assertIn('retain original geometry', captured[0]['instructions'])

    def test_predicted_budget_stop_preserves_evidence_without_paid_report(self):
        (self.root / 'input.json').write_text('{"error":"known failure"}')
        replies = [response(calls=[('json_query', {'path': 'input.json', 'pointers': ['/error']})])]
        replies[0]['usage']['input_tokens'] = 1200
        captured = []
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload)); return replies.pop(0)
        with patch.object(s, 'call_responses', side_effect=fake):
            report, stats = s.run_subagent('inspect', '', self.cwd, 'deepseek-flash', 'max', 20, False, False, 30,
                                         input_budget=2000, record_dir=self.root / 'run')
        self.assertEqual(stats['status'], 'input_budget_prediction')
        self.assertEqual(len(captured), 1)
        receipt = json.loads((self.root / 'run/receipt.json').read_text())
        self.assertEqual(receipt['action_records'][0]['status'], 'returned')
        evidence = receipt['action_records'][0]['result_file']
        self.assertEqual(p.pin(evidence['path']), evidence)
        self.assertIn('known failure', Path(evidence['path']).read_text())
        self.assertNotIn('reasoning', receipt)

    def test_command_contract_hides_shell_and_records_interruption(self):
        contract = self.contract([self.command()])
        replies = [response(calls=[('run_command', {'id': 'test'})])]
        captured = []
        def fake(payload, timeout):
            captured.append(payload); return replies.pop(0)
        with patch.object(s, 'call_responses', side_effect=fake), patch.object(p, 'run_command', side_effect=s.DeadlineExceeded('interrupted')):
            report, stats = s.run_subagent('run', '', self.cwd, 'deepseek-flash', 'max', 5, False, True, 30,
                                         contract=contract, record_dir=self.root / 'run')
        self.assertEqual(stats['status'], 'timeout')
        names = {t['name'] for t in captured[0]['tools']}
        self.assertIn('run_command', names); self.assertNotIn('run_shell', names)
        receipt = json.loads((self.root / 'run/receipt.json').read_text())
        self.assertEqual(receipt['action_records'][0]['status'], 'started_outcome_unknown')


if __name__ == '__main__': unittest.main()
