"""Offline behavior tests for evidence, recovery, host handoff and native plans."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import client
import host_bridge as b
import project_support as p
import server as s
import state_store as store
import prepare
from test_server import response


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.cwd = str(self.root)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(s, 'RUNS_DIR', self.runs).start()
        self.api = patch.object(s, 'call_responses', side_effect=AssertionError('Unexpected model request')).start()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    def contract(self, **updates):
        value = {'schema': 'DEEPSEEK_TASK_V2', 'objective': 'bounded test', 'acceptance': ['actual evidence'],
                 'read_paths': [self.cwd + '/'], 'write_paths': [self.cwd + '/'], 'commands': []}
        value.update(updates)
        path = self.write('contract.json', value)
        return p.load_contract(path, self.cwd)

    def call(self, **args):
        return s.handle_tools_call({'name': 'deepseek_subagent', 'arguments': args})

    def host_start(self, reply=None, **kwargs):
        packet = self.write('host.json', {'schema': 'DEEPSEEK_HOST_V1',
            'conversation': [{'role': 'user', 'text': 'Example project context'}],
            'capabilities': [{'id': 'browser_read', 'description': 'Read a page in host browser', 'permission': 'read'}]})
        self.api.side_effect = None
        self.api.return_value = reply or response(calls=[('request_host', {'capability': 'browser_read', 'arguments': {'url': 'https://example.test/paper'}, 'reason': 'Read evidence'})])
        result = self.call(task='inspect evidence', cwd=self.cwd, task_id='host-task', host_context_path=str(packet), max_steps=6, **kwargs)
        self.assertEqual(result['structuredContent']['status'], 'needs_host', result)
        return result

    def claim(self, start):
        meta = start['structuredContent']
        request = meta.get('host_request') or meta
        return self.call(action='claim', task_id=meta.get('task_id') or 'native', request_id=request['request_id'])

    def host_result(self, start, status='completed'):
        request_id = start['structuredContent']['host_request']['request_id']
        return self.write('host-result.json', {'request_id': request_id, 'status': status,
            'text': 'Actual host result: source explicitly leaves temperature unknown.',
            'sources': [{'url': 'https://example.test/paper', 'title': 'Evidence'}]})

    def test_json_is_valid_with_source_and_bounded_nested_output(self):
        path = self.write('large.json', {'nodes': [{'text': 'x'*1000} for _ in range(30)]})
        for cap in (1000, 3000, 6000):
            text = p.tool_json_query({'path': str(path), 'pointers': ['/nodes'], 'depth': 3, 'limit': 30}, self.cwd, cap)
            self.assertLessEqual(len(text), cap)
            result = json.loads(text)
            self.assertEqual(result['source'], p.pin(path))
            self.assertEqual(result['selection']['/nodes']['count'], 30)
            self.assertIsNotNone(result['selection']['/nodes']['next_offset'])

    def test_json_scalar_pages_reconstruct_original(self):
        original = '示例\\"\n'*500
        path = self.write('text.json', {'s': original})
        parts, offset = [], 0
        while offset is not None:
            result = json.loads(p.tool_json_query({'path': str(path), 'pointers': ['/s'], 'offset': offset}, self.cwd, 1000))
            value = result['selection']['/s']
            parts.append(value['preview'])
            self.assertNotEqual(offset, value['next_offset'])
            offset = value['next_offset']
        self.assertEqual(''.join(parts), original)

    def test_query_schema_exposes_actual_bounds(self):
        properties = p.data_tool_schemas()[0]['parameters']['properties']
        self.assertEqual(properties['pointers']['maxItems'], 12)
        self.assertEqual(properties['depth']['maximum'], 3)
        self.assertEqual(properties['limit']['maximum'], 30)

    def test_snapshot_detects_mid_read_mutation(self):
        path = self.write('file.txt', 'original')
        real = p.os.fstat
        count = 0
        def mutated(fd):
            nonlocal count
            count += 1
            if count == 2: path.write_text('changed to a longer version')
            return real(fd)
        with patch.object(p.os, 'fstat', side_effect=mutated), self.assertRaisesRegex(ValueError, 'changed'):
            p.snapshot(path)

    def test_read_file_returns_complete_lines_source_and_cursor(self):
        path = self.write('source.py', '\n'.join('x'*100 for _ in range(100)))
        text = s.tool_read_file({'path': str(path), 'start_line': 1, 'end_line': 100}, self.cwd)
        self.assertLessEqual(len(text), 6000)
        identity = json.loads(text.splitlines()[0].removeprefix('SOURCE '))
        self.assertEqual(identity, p.pin(path))
        numbered = [line for line in text.splitlines() if '\t' in line]
        self.assertTrue(all(line.split('\t')[1] == 'x'*100 for line in numbered))
        self.assertIn(f'next_line={len(numbered)+1}', text)

    def test_read_pins_and_dependencies_reject_before_api(self):
        source = self.write('source.json', {'value': 1})
        contract = self.contract(read_pins=[p.pin(source)])
        source.write_text('{"value":2}')
        result = self.call(task='read', cwd=self.cwd, contract_path=contract['pin']['path'])
        self.assertTrue(result['isError'])
        self.api.assert_not_called()
        value = contract['value']; value.pop('read_pins'); value['dependencies'] = ['missing.py']
        self.write('contract.json', value)
        result = self.call(task='read', cwd=self.cwd, contract_path=str(self.root/'contract.json'))
        self.assertTrue(result['isError'])
        self.api.assert_not_called()

    def command(self, candidate=False):
        script = self.write('driver.py', 'from pathlib import Path\np=Path("counter")\np.write_text(str(int(p.read_text())+1) if p.exists() else "1")\nprint("validation result")\n')
        cmd = {'id': 'validate', 'argv': [sys.executable, str(script)], 'cwd': self.cwd, 'timeout_sec': 3, 'inputs': [p.pin(script)]}
        if candidate:
            self.write('candidate.py', 'v1')
            cmd.update(mode='validation', candidate_paths=['candidate.py'], max_attempts=2)
        return cmd

    def test_validation_new_candidate_allowed_same_candidate_reused(self):
        contract = self.contract(commands=[self.command(True)])
        ledger = self.root / 'ledger'
        first = p.run_command({'id': 'validate'}, contract, set(), self.root/'a', ledger)
        same = p.run_command({'id': 'validate'}, contract, set(), self.root/'b', ledger)
        self.assertEqual(first, same)
        self.assertFalse((self.root/'b').exists())
        self.write('candidate.py', 'v2')
        p.run_command({'id': 'validate'}, contract, set(), self.root/'c', ledger)
        self.assertEqual((self.root/'counter').read_text(), '2')
        self.write('candidate.py', 'v3')
        with self.assertRaisesRegex(ValueError, 'budget'):
            p.run_command({'id': 'validate'}, contract, set(), self.root/'d', ledger)

    def test_unknown_command_outcome_blocks_new_runs(self):
        contract = self.contract(commands=[self.command()])
        ledger = self.root/'ledger'
        with patch.object(p, '_execute_command', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError): p.run_command({'id':'validate'}, contract, set(), self.root/'a', ledger)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            p.run_command({'id':'validate'}, contract, set(), self.root/'b', ledger)
        self.assertFalse((self.root/'counter').exists())

    def test_once_command_is_not_executed_again_in_new_run(self):
        contract = self.contract(commands=[self.command()])
        for attempt in ('a', 'b'):
            p.run_command({'id':'validate'}, contract, set(), self.root/attempt, self.root/'ledger')
        self.assertEqual((self.root/'counter').read_text(), '1')

    def test_command_saved_evidence_mutation_rejects_reuse(self):
        contract = self.contract(commands=[self.command()])
        result = json.loads(p.run_command({'id':'validate'}, contract, set(), self.root/'a', self.root/'ledger'))
        Path(result['stdout']['path']).write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'changed'):
            p.run_command({'id':'validate'}, contract, set(), self.root/'b', self.root/'ledger')

    def test_read_after_two_edits_is_not_false_loop(self):
        self.write('source.py', 'one')
        self.api.side_effect = [response(calls=[('read_file', {'path':'source.py'})]),
            response(calls=[('edit_file', {'path':'source.py','old_text':'one','new_text':'two'})]),
            response(calls=[('read_file', {'path':'source.py'})]),
            response(calls=[('edit_file', {'path':'source.py','old_text':'two','new_text':'three'})]),
            response(calls=[('read_file', {'path':'source.py'})]), response('done')]
        result = self.call(task='bounded fix', cwd=self.cwd, allow_write=True, max_steps=10)
        self.assertFalse(result['isError'], result)

    def test_grep_fallback_does_not_follow_file_symlink(self):
        outside = self.write('outside', 'MARKER')
        (self.root/'allowed').mkdir()
        (self.root/'allowed/link').symlink_to(outside)
        contract = self.contract(read_paths=['allowed/'])
        with patch.object(s.shutil, 'which', return_value=None):
            result = s.dispatch('grep', json.dumps({'path':'allowed','pattern':'MARKER'}), self.cwd, False, False, contract)
        self.assertNotIn('MARKER', result)

    def test_contract_scope_does_not_move_when_symlink_is_retargeted(self):
        (self.root/'original').mkdir(); (self.root/'other').mkdir()
        link = self.root/'alias'; link.symlink_to(self.root/'original', target_is_directory=True)
        contract = self.contract(read_paths=['alias/'])
        link.unlink(); link.symlink_to(self.root/'other', target_is_directory=True)
        with self.assertRaises(ValueError): p.check_access('read_file', {'path':'alias/file'}, self.cwd, contract)

    def test_acceptance_check_prevents_false_completed(self):
        self.write('report.json', {'native_ok': False})
        contract = self.contract(checks=[{'id':'native','path':'report.json','pointer':'/native_ok','expected':True}])
        self.api.side_effect = None; self.api.return_value = response('PASS')
        result = self.call(task='inspect', cwd=self.cwd, contract_path=contract['pin']['path'])
        self.assertEqual(result['structuredContent']['status'], 'acceptance_failed')
        self.assertFalse(result['structuredContent']['acceptance_checks'][0]['passed'])

    def test_task_result_replayed_without_api_and_conflicts_rejected(self):
        self.api.side_effect = None; self.api.return_value = response('done')
        args = {'task':'inspect','cwd':self.cwd,'task_id':'stable'}
        first = self.call(**args); second = self.call(**args)
        self.assertFalse(first['isError'])
        self.assertTrue(second['structuredContent']['reused_result'])
        self.assertEqual(self.api.call_count, 1)
        self.assertEqual(self.call(**(args | {'task':'different'}))['structuredContent']['status'], 'task_conflict')
        self.assertEqual(self.call(action='status', task_id='stable')['structuredContent']['status'], 'completed')

    def test_task_lock_excludes_other_processes(self):
        path = self.root/'claim.lock'
        with store.lock(path):
            script = 'import sys;sys.path.insert(0,sys.argv[1]);import state_store;\nwith state_store.lock(sys.argv[2]): print("entered")'
            proc = subprocess.run([sys.executable, '-c', script, str(Path(s.__file__).parent), str(path)], text=True, capture_output=True, timeout=3)
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn('entered', proc.stdout)

    def test_host_handoff_preserves_history_budgets_and_tool_pairing(self):
        initial = response('checking', calls=[('request_host', {'capability':'browser_read','arguments':{'url':'https://example.test'},'reason':'evidence'}), ('write_file', {'path':'not-executed','content':'bad'})])
        initial['output'].insert(0, {'type':'reasoning','content':[{'type':'reasoning_text','text':'flash protocol fixture'}]})
        start = self.host_start(initial, allow_write=True)
        self.assertFalse((self.root/'not-executed').exists())
        self.assertFalse(self.claim(start)['isError'])
        self.api.return_value = response('Verified supplied evidence; temperature remains unknown')
        result = self.call(action='resume', task_id='host-task', host_result_path=str(self.host_result(start)))
        self.assertFalse(result['isError'], result)
        payload = self.api.call_args.args[0]
        self.assertTrue(any(x.get('type')=='reasoning' for x in payload['input']))
        outputs = [x for x in payload['input'] if x.get('type')=='function_call_output']
        self.assertEqual(len(outputs), 2)
        self.assertTrue(any('Actual host result' in x['output'] for x in outputs))
        receipt = json.loads(Path(result['structuredContent']['receipt_path']).read_text())
        self.assertEqual(receipt['input_tokens'], 200)
        self.assertEqual(receipt['steps'], 2)
        self.assertFalse((Path(result['structuredContent']['receipt_path']).parent/'continuation.json').exists())
        metrics = [json.loads(x) for x in (self.runs/'usage.jsonl').read_text().splitlines()]
        self.assertEqual(sum(x['input_tokens'] for x in metrics), 200)

    def test_host_claim_cannot_be_repeated_and_unknown_does_not_resume(self):
        start = self.host_start()
        self.assertFalse(self.claim(start)['isError'])
        self.assertTrue(self.claim(start)['isError'])
        result = self.call(action='resume', task_id='host-task', host_result_path=str(self.host_result(start, 'unknown')))
        self.assertEqual(result['structuredContent']['status'], 'host_outcome_unknown')
        self.assertEqual(self.api.call_count, 1)

    def test_host_resume_cannot_change_id_or_budget(self):
        start = self.host_start(); self.claim(start)
        path = self.host_result(start)
        value = json.loads(path.read_text()); value['request_id'] = 'wrong'; path.write_text(json.dumps(value))
        result = self.call(action='resume', task_id='host-task', host_result_path=str(path))
        self.assertTrue(result['isError']); self.assertEqual(self.api.call_count, 1)
        result = self.call(action='resume', task_id='host-task', host_result_path=str(path), output_budget=128000)
        self.assertTrue(result['isError']); self.assertEqual(self.api.call_count, 1)

    def test_host_resume_output_budget_is_cumulative(self):
        initial = response(calls=[('request_host', {'capability':'browser_read','arguments':{},'reason':'evidence'})], tokens=450)
        start = self.host_start(initial, output_budget=512)
        self.claim(start)
        result = self.call(action='resume', task_id='host-task', host_result_path=str(self.host_result(start)))
        self.assertEqual(result['structuredContent']['status'], 'output_budget')
        self.assertEqual(self.api.call_count, 1)

    def test_host_context_and_images_are_pinned_real_response_parts(self):
        image = self.root/'pixel.png'
        image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j3ioAAAAASUVORK5CYII='))
        packet = self.write('images.json', {'schema':'DEEPSEEK_HOST_V1', 'conversation':[{'role':'user','text':'Inspect this figure'}], 'images':[p.pin(image)]})
        self.api.side_effect = None; self.api.return_value = response('image inspected')
        result = self.call(task='inspect', cwd=self.cwd, host_context_path=str(packet))
        self.assertFalse(result['isError'], result)
        payload = self.api.call_args.args[0]
        images = [part for item in payload['input'] if isinstance(item.get('content'), list) for part in item['content'] if part['type']=='input_image']
        self.assertEqual(base64.b64decode(images[0]['image_url'].split(',',1)[1]), image.read_bytes())
        image.write_bytes(b'changed')
        result = self.call(task='inspect', cwd=self.cwd, host_context_path=str(packet))
        self.assertTrue(result['isError']); self.assertEqual(self.api.call_count, 1)

    def test_wire_budget_keeps_image_bytes_separate_from_text_budget(self):
        payload = {'input':[{'role':'user','content':[{'type':'input_image','image_url':'data:image/png;base64,'+'A'*300000}]}]}
        text, wire = b.request_sizes(payload)
        self.assertLess(text, 1000); self.assertGreater(wire, 300000)

    def test_native_mode_only_prepares_claimed_host_delegation(self):
        result = self.call(action='native_request', task_id='native', task='read bounded files', cwd=self.cwd)
        meta = result['structuredContent']
        self.assertEqual(meta['status'], 'needs_native_agent')
        self.assertEqual(meta['spawn_arguments']['model'], 'deepseek-flash')
        self.assertEqual(meta['spawn_arguments']['reasoning_effort'], 'max')
        self.assertEqual(meta['spawn_arguments']['fork_turns'], '100')
        self.assertIn('unverified', meta['billing'])
        self.assertFalse(self.claim(result)['isError'])
        self.assertTrue(self.claim(result)['isError'])
        report = self.write('native-report.json', {'agent_id':'mock-child','text':'Actual child report','evidence':[]})
        completed = self.call(action='native_result', task_id='native', native_result_path=str(report))
        self.assertEqual(completed['structuredContent']['status'], 'native_reported')
        self.assertTrue(completed['structuredContent']['controller_acceptance_required'])
        self.api.assert_not_called()

    def test_private_continuation_not_in_status_response(self):
        start = self.host_start()
        continuation = Path(start['structuredContent']['receipt_path']).parent/'continuation.json'
        self.assertEqual(continuation.stat().st_mode & 0o777, 0o600)
        status = self.call(action='status', task_id='host-task')
        self.assertNotIn('Example project context', json.dumps(status))

    def test_native_failure_is_not_reported_as_success(self):
        start = self.call(action='native_request',task_id='native',task='bounded task',cwd=self.cwd)
        self.claim(start)
        report = self.write('native-failure.json',{'agent_id':'mock-child','text':'Tool failed; work remains','execution_status':'failed','evidence':[]})
        result = self.call(action='native_result',task_id='native',native_result_path=str(report))
        self.assertTrue(result['isError'])
        self.assertEqual(result['structuredContent']['status'],'native_failed')
        self.assertEqual(result['structuredContent']['report_source'],p.pin(report))
        self.api.assert_not_called()

    def test_client_reserves_result_before_process_and_rejects_second_client(self):
        arguments = self.write('client-args.json', {'task':'mock only'})
        result_path = self.root/'client-result.json'
        messages = [{'id':1,'result':{'serverInfo':{'version':'mock'}}}, {'id':2,'result':{'tools':[]}}, {'id':3,'result':{'content':[],'isError':False}}]
        def process(*args, **kwargs):
            self.assertTrue(result_path.exists())
            self.assertEqual(json.loads(result_path.read_text())['status'], 'client_started_outcome_unknown')
            fake = Mock(returncode=0)
            fake.communicate.return_value = ('\n'.join(map(json.dumps,messages)), '')
            return fake
        with patch.object(sys, 'argv', ['client.py','--arguments',str(arguments),'--result',str(result_path)]), patch.object(client.subprocess, 'Popen', side_effect=process) as launch:
            self.assertEqual(client.main(), 0)
            with self.assertRaises(SystemExit): client.main()
            self.assertEqual(launch.call_count, 1)

    def test_nested_deadlines_do_not_reset_outer_limit(self):
        start = time.monotonic()
        with self.assertRaises(s.DeadlineExceeded):
            with s.wall_deadline(.03):
                with s.wall_deadline(10): time.sleep(.1)
        self.assertLess(time.monotonic()-start, .09)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_stale_report_for_previous_candidate_fails_acceptance(self):
        candidate = self.write('candidate.py', 'original')
        self.write('report.json', {'status':'PASS','candidate_sha256':p.pin(candidate)['sha256']})
        contract = self.contract(checks=[{'id':'identity','path':'report.json','pointer':'/candidate_sha256','expected_file_sha256':'candidate.py'}])
        self.assertTrue(p.acceptance_results(contract,self.cwd)[0]['passed'])
        candidate.write_text('modified')
        self.assertFalse(p.acceptance_results(contract,self.cwd)[0]['passed'])

    def test_prepare_contract_fills_pins_and_keeps_candidates_mutable(self):
        self.write('spec.md','scientific spec')
        self.write('candidate.py','candidate')
        draft = {'objective':'fix','acceptance':['validate'], 'read_paths':['spec.md','candidate.py','out/'],
                 'write_paths':['candidate.py'],'pinned_read_paths':['spec.md'],'dependencies':['spec.md'],
                 'checks':[{'id':'source','path':'out/REPORT.json','pointer':'/source','expected_file_sha256':'candidate.py'}]}
        value = prepare.build_contract(draft,self.cwd)
        self.assertEqual(value['read_pins'],[p.pin(self.root/'spec.md')])
        self.assertEqual(value['checks'][0]['expected_file_sha256'],str(self.root/'candidate.py'))
        self.assertTrue(value['read_paths'][-1].endswith('/'))

    def test_resumed_request_must_not_auto_repeat_after_lost_api_response(self):
        start = self.host_start(); self.claim(start)
        self.api.side_effect = RuntimeError('response lost')
        first = self.call(action='resume',task_id='host-task',host_result_path=str(self.host_result(start)))
        self.assertTrue(first['isError'])
        second = self.call(action='resume',task_id='host-task',host_result_path=str(self.root/'host-result.json'))
        self.assertEqual(second['structuredContent']['status'],'not_resumable')
        self.assertEqual(self.api.call_count,2)

    def test_changed_project_contract_blocks_resume_before_model(self):
        contract = self.contract()
        start = self.host_start(contract_path=contract['pin']['path']); self.claim(start)
        value = contract['value']; value['objective'] = 'changed'
        self.write('contract.json',value)
        result = self.call(action='resume',task_id='host-task',host_result_path=str(self.host_result(start)))
        self.assertTrue(result['isError']); self.assertEqual(self.api.call_count,1)

    def test_durable_saved_tool_output_can_be_read_under_contract(self):
        contract = self.contract(read_paths=['allowed/'])
        path = self.write('outside/tool_result.txt','actual tool evidence')
        result = s.dispatch('read_file',json.dumps({'path':str(path)}),self.cwd,False,False,contract,evidence=[p.pin(path)])
        self.assertIn('actual tool evidence',result)
        self.assertFalse(result.startswith('错误：'))
        denied = s.dispatch('read_file',json.dumps({'path':str(path)}),self.cwd,False,False,contract)
        self.assertTrue(denied.startswith('错误：'))

    def test_stdio_native_planning_handshake_never_uses_api(self):
        import os
        env = dict(os.environ,DEEPSEEK_SUBAGENT_RUNS_DIR=str(self.runs),DEEPSEEK_API_BASE='http://127.0.0.1:9',DEEPSEEK_API_KEY='offline-test-only')
        messages = [{'jsonrpc':'2.0','id':1,'method':'initialize','params':{}},
                    {'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}},
                    {'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'deepseek_subagent','arguments':{'action':'native_request','task_id':'offline-plan','task':'mock plan only','cwd':self.cwd}}}]
        proc = subprocess.run([sys.executable,s.__file__],input='\n'.join(map(json.dumps,messages))+'\n',text=True,capture_output=True,timeout=5,env=env)
        rows = [json.loads(x) for x in proc.stdout.splitlines()]
        self.assertEqual(proc.returncode,0)
        self.assertEqual(rows[0]['result']['serverInfo']['version'],s.SERVER_VERSION)
        self.assertIn('host_context_path',rows[1]['result']['tools'][0]['inputSchema']['properties'])
        self.assertEqual(rows[-1]['result']['structuredContent']['status'],'needs_native_agent')


if __name__ == '__main__': unittest.main()
