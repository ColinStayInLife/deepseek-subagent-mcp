"""Real directory/grep, budget and protocol regressions; model calls are fixtures.

No project writes, CAD commands, paid model requests or credentials are used.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import async_jobs
import prepare
import project_support as project
import server
import state_store as state
import workflow_support as workflow
import platform_support as platform
from test_platform_support import directory_link
from test_server import response


class ContractEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(self.temp.name).resolve()
        self.cwd = str(self.root)
        patch.object(server, 'RUNS_DIR', self.root/'runs').start()
        patch.object(server, 'load_api_key', side_effect=AssertionError('No credentials')).start()
        self.api = patch.object(server, 'call_responses', side_effect=AssertionError('No live model')).start()

    def write(self, name, value):
        p = self.root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value if isinstance(value, str) else json.dumps(value))
        return p

    def contract(self, **updates):
        value = {'schema': 'DEEPSEEK_TASK_V2', 'objective': 'bounded audit',
                 'acceptance': ['controller checks evidence'], 'read_paths': ['allowed/'],
                 'write_paths': [], 'commands': [], **updates}
        return self.write('contract.json', value)

    def call(self, **kw):
        return server.handle_tools_call({'name': 'deepseek_subagent', 'arguments': {
            'task': 'bounded fixture', 'cwd': self.cwd, 'max_steps': 8, **kw}})

    def dispatch(self, tool, args, contract):
        return server.dispatch(tool, json.dumps(args), self.cwd, False, False, contract)

    def test_ambiguous_directory_rejected_before_synchronous_model_call(self):
        (self.root/'allowed').mkdir()
        path = self.contract(read_paths=['allowed'])
        result = self.call(contract_path=str(path))
        self.assertTrue(result['isError'])
        self.assertIn('directory scope must end with /', result['content'][0]['text'])
        self.api.assert_not_called()

    def test_async_rejects_bad_scope_before_enqueuing(self):
        (self.root/'allowed').mkdir()
        path = self.contract(read_paths=['allowed'])
        with self.assertRaisesRegex(ValueError, 'directory scope must end with /'):
            async_jobs.prepare({'task_id': 'bad', 'task': 'fixture', 'cwd': self.cwd,
                                'contract_path': str(path)}, self.root/'runs')
        self.assertFalse((self.root/'runs/async/jobs').exists())
        self.api.assert_not_called()

    def test_write_directory_and_file_with_slash_are_rejected(self):
        (self.root/'allowed').mkdir()
        self.write('file.txt', 'data')
        for args in [{'write_paths': ['allowed']}, {'read_paths': ['file.txt/']}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                project.load_contract(self.contract(**args), self.cwd)

    def test_explicit_directory_draft_fields_do_not_expand_file_scopes(self):
        (self.root/'allowed').mkdir()
        self.write('one.txt', 'single')
        value = prepare.build_contract({'objective': 'audit', 'acceptance': ['evidence'],
            'read_paths': ['one.txt'], 'read_dirs': ['allowed'], 'write_dirs': ['new output']}, self.cwd)
        self.assertEqual(value['read_paths'], [str(self.root/'one.txt'), str(self.root/'allowed')+'/'])
        self.assertEqual(value['write_paths'], [str(self.root/'new output')+'/'])
        with self.assertRaisesRegex(ValueError, 'directory scope must end with /'):
            prepare.build_contract({'objective': 'audit', 'acceptance': ['evidence'],
                                    'read_paths': ['allowed']}, self.cwd)

    def test_file_scope_cannot_become_recursive_after_load(self):
        p = self.write('allowed', 'file')
        contract = project.load_contract(self.contract(read_paths=['allowed']), self.cwd)
        p.unlink(); p.mkdir(); self.write('allowed/child.txt', 'DO_NOT_READ')
        with patch.object(server.subprocess, 'Popen', side_effect=AssertionError('No recursive search')):
            for tool, args in [('grep', {'path': 'allowed', 'pattern': 'DO_NOT_READ'}),
                               ('list_dir', {'path': 'allowed'}), ('read_file', {'path': 'allowed/child.txt'})]:
                result = self.dispatch(tool, args, contract)
                self.assertTrue(result.startswith('错误：'), result)
                self.assertNotIn('DO_NOT_READ', result)

    def test_recursive_scope_can_list_read_and_search_consistently(self):
        self.write('allowed/child.txt', 'MATCH_ME')
        contract = project.load_contract(self.contract(), self.cwd)
        for tool, args, expect in [('list_dir', {'path': 'allowed'}, 'child.txt'),
                                  ('read_file', {'path': 'allowed/child.txt'}, 'MATCH_ME'),
                                  ('grep', {'path': 'allowed', 'pattern': 'MATCH_ME'}, 'MATCH_ME')]:
            result = self.dispatch(tool, args, contract)
            self.assertFalse(result.startswith('错误：'), result)
            self.assertIn(expect, result)

    def test_rg_and_fallback_never_bypass_changed_pin(self):
        p = self.write('allowed/child.txt', 'original')
        contract = project.load_contract(self.contract(read_pins=[project.pin(p)]), self.cwd)
        p.write_text('CHANGED_CONTENT')
        real_rg = shutil.which('rg')
        for binary in [None, real_rg]:
            with self.subTest(binary=binary), patch.object(server.shutil, 'which', return_value=binary):
                result = self.dispatch('grep', {'path': 'allowed', 'pattern': 'CHANGED_CONTENT'}, contract)
                self.assertTrue(result.startswith('错误：'), result)
                self.assertIn('evidence changed', result)
                self.assertNotIn('CHANGED_CONTENT', result)

    def test_search_pin_retargeted_symlink_is_denied_before_hashing_outside(self):
        p = self.write('allowed/child.txt', 'original')
        outside = self.write('outside/secret.txt', 'PRIVATE')
        contract = project.load_contract(self.contract(read_pins=[project.pin(p)]), self.cwd)
        p.unlink()
        if platform.WINDOWS:
            p.parent.rmdir()
            directory_link(p.parent, outside.parent)
        else:
            p.symlink_to(outside)
        original_pin = project.pin
        def guarded_pin(path):
            if Path(path).resolve() == outside:
                raise AssertionError('Must deny before hashing an outside target')
            return original_pin(path)
        with patch.object(project, 'pin', side_effect=guarded_pin):
            result = self.dispatch('grep', {'path': 'allowed', 'pattern': 'PRIVATE'}, contract)
        self.assertTrue(result.startswith('错误：'), result)
        self.assertIn('outside explicit read_paths', result)
        self.assertNotIn('PRIVATE', result)

    def test_search_revalidates_pins_after_rg(self):
        if not shutil.which('rg'):
            self.skipTest('rg unavailable')
        p = self.write('allowed/child.txt', 'ORIGINAL')
        contract = project.load_contract(self.contract(read_pins=[project.pin(p)]), self.cwd)
        real_check = project.verify_search_pins
        calls = []
        def check(root, contract):
            calls.append(1)
            if len(calls) == 2: p.write_text('CHANGED')
            return real_check(root, contract)
        with patch.object(project, 'verify_search_pins', side_effect=check):
            result = self.dispatch('grep', {'path': 'allowed', 'pattern': 'ORIGINAL'}, contract)
        self.assertEqual(len(calls), 2)
        self.assertTrue(result.startswith('错误：'), result)

    def test_single_file_fallback_search_and_symlink_isolation(self):
        self.write('allowed/source.txt', 'MARKER\nother')
        secret = self.write('outside/secret.txt', 'SECRET_ONLY')
        if platform.WINDOWS:
            directory_link(self.root/'allowed/link', secret.parent)
        else:
            (self.root/'allowed/link').symlink_to(secret)
        directory_link(self.root/'allowed/outside_dir', secret.parent)
        contract = project.load_contract(self.contract(), self.cwd)
        for binary in [None, shutil.which('rg')]:
            with self.subTest(binary=binary), patch.object(server.shutil, 'which', return_value=binary):
                result = self.dispatch('grep', {'path': 'allowed/source.txt', 'pattern': 'MARKER'}, contract)
                self.assertIn('MARKER', result)
                result = self.dispatch('grep', {'path': 'allowed', 'pattern': 'SECRET_ONLY'}, contract)
                self.assertNotIn('SECRET_ONLY', result)

    def test_directory_pagination_preserves_every_entry_with_output_cap(self):
        for i in range(140): self.write(f'allowed/{i:03d}_long_filename.txt', 'x')
        offset, names = 0, []
        while offset is not None:
            text = server.tool_list_dir({'path': 'allowed', 'offset': offset}, self.cwd, output_chars=1000)
            self.assertLessEqual(len(text), 1000)
            self.assertNotIn('已截断', text)
            names += [line.split('\t')[0] for line in text.splitlines() if '\t' in line]
            next_value = re.search(r'next_offset=(\d+|None)', text).group(1)
            offset = None if next_value == 'None' else int(next_value)
        self.assertEqual(len(names), 140)
        self.assertEqual(len(set(names)), 140)

    def test_check_contract_cli_reports_invalid_and_missing_without_models(self):
        (self.root/'allowed').mkdir()
        for paths, code, expected in [(['allowed'], 2, 'invalid'), (['allowed/', 'future.txt'], 0, 'valid')]:
            contract = self.contract(read_paths=paths)
            run = subprocess.run([sys.executable, str(Path(prepare.__file__)), 'check-contract',
                '--contract', str(contract), '--cwd', self.cwd], capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, code, run.stderr)
            value = json.loads(run.stdout)
            self.assertEqual(value['status'], expected)
            self.assertEqual(value['model_calls'], 0)
            self.assertEqual(value['commands_run'], 0)
            if code == 0: self.assertEqual(value['missing_read_paths'], [str(self.root/'future.txt')])

    def test_tool_budget_reserves_report_and_keeps_incomplete_status(self):
        calls = [('read_file', {'path': f'allowed/{i}.txt'}) for i in range(38)]
        for i in range(38): self.write(f'allowed/{i}.txt', 'evidence')
        self.api.side_effect = [response(calls=calls), response('Evidence complete enough for controller; unresolved items remain')]
        result = self.call(contract_path=str(self.contract()))
        meta = result['structuredContent']
        self.assertEqual(meta['status'], 'tool_limit')
        self.assertTrue(result['isError'])
        self.assertEqual(meta['completion']['model_report'], 'received')
        self.assertEqual(meta['tool_budget']['remaining'], 2)
        self.assertEqual(self.api.call_args_list[-1].args[0]['tool_choice'], 'none')
        self.assertEqual(self.api.call_args_list[-1].args[0]['reasoning']['effort'], 'max')

    def test_tool_overflow_pairs_only_unexecuted_tail_and_checks_real_artifact(self):
        calls = [('write_file', {'path': f'out/{i}.txt', 'content': 'actual'}) for i in range(41)]
        self.api.side_effect = [response(calls=calls)]
        path = self.contract(write_paths=['out/'], read_paths=['out/'],
                             deliverables=[{'path': 'out/0.txt', 'must_change': True}])
        result = self.call(contract_path=str(path), allow_write=True)
        meta = result['structuredContent']
        self.assertEqual(meta['status'], 'tool_limit')
        self.assertEqual(meta['completion']['verification'], 'passed')
        self.assertEqual(meta['completion']['model_report'], 'not_received')
        self.assertEqual(meta['completion']['controller_acceptance'], 'pending')
        self.assertEqual(len(meta['not_executed_calls']), 1)
        self.assertFalse((self.root/'out/40.txt').exists())
        self.assertEqual(len(list((self.root/'out').iterdir())), 40)
        handoff = state.read(meta['handoff_path'])
        self.assertEqual(handoff['not_executed_calls'], meta['not_executed_calls'])
        self.assertEqual(self.api.call_count, 1)  # No paid summary just to hide a limit.

    def test_missing_artifact_at_tool_overflow_is_not_a_pass(self):
        calls = [('read_file', {'path': f'allowed/{i}.txt'}) for i in range(41)]
        for i in range(41): self.write(f'allowed/{i}.txt', 'evidence')
        self.api.side_effect = [response(calls=calls)]
        path = self.contract(write_paths=['missing.txt'],
                             deliverables=[{'path': 'missing.txt', 'must_change': True}])
        result = self.call(contract_path=str(path), allow_write=True)
        self.assertEqual(result['structuredContent']['completion']['verification'], 'failed')
        self.assertEqual(result['structuredContent']['status'], 'tool_limit')
        self.assertTrue(result['isError'])

    def test_unexecuted_pairing_does_not_hide_unknown_side_effects(self):
        call = {'type': 'function_call', 'call_id': 'pending', 'name': 'write_file'}
        items = [call]
        stats = {'steps': 1, 'usage_complete': True, 'action_records': [
            {'tool': 'write_file', 'status': 'started_outcome_unknown'}]}
        workflow.skip_unexecuted_calls(items, [call], stats, 'not dispatched tail')
        self.assertTrue(workflow.paired(items))
        gate = workflow.verification_gate('tool_limit', items, stats, None)
        self.assertEqual(gate['status'], 'deferred')
        self.assertEqual(gate['reason'], 'unresolved_tool_or_host_action')


if __name__ == '__main__':
    unittest.main()
