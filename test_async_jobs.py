"""Offline integration tests: real subprocesses and locks, mocked model calls."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import async_jobs as a
import project_support as p
import server as s
import state_store as state
from test_server import response


def fixture_worker(root, task_id):
    root = Path(root)
    calls = 0
    def fake(payload, timeout):
        nonlocal calls
        calls += 1
        job = state.read(a.job_path(root, task_id))
        task = job['arguments']['task']
        event = {'task_id': task_id, 'start': time.time(), 'model': payload['model'],
                 'effort': payload['reasoning']['effort'], 'output_limit': payload['max_output_tokens']}
        try:
            if task == 'crash': os._exit(9)
            time.sleep(float(job['arguments'].get('context') or .1))
            if task == 'fail': return response(status='incomplete')
            if task == 'host' and not any(x.get('type') == 'function_call_output' for x in payload['input']):
                return response(calls=[('request_host', {'capability': 'read', 'arguments': {'query': 'evidence'}, 'reason': 'need source'})])
            if task == 'shell' and not any(x.get('type') == 'function_call_output' for x in payload['input']):
                return response(calls=[('run_command', {'id': 'sleep'})])
            if task == 'clone' and not any(x.get('type') == 'function_call_output' for x in payload['input']):
                source = Path(job['arguments']['cwd'])/'source.py'
                return response(calls=[('clone_file', {'source_path': str(source), 'path': 'candidate.py',
                    'source_sha256': p.pin(source)['sha256'],
                    'edits': [{'old_text': 'old', 'new_text': 'new'}]})])
            return response('verified fixture evidence')
        finally:
            event['end'] = time.time()
            fd = os.open(root / 'events.jsonl', os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try: os.write(fd, (json.dumps(event)+'\n').encode())
            finally: os.close(fd)
    with patch.object(s, 'call_responses', side_effect=fake), patch.object(s, 'load_api_key', side_effect=AssertionError('No live credentials in tests')):
        a.worker(root, task_id)


class AsyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.procs = []
        self.addCleanup(self.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(s, 'RUNS_DIR', self.runs).start()
        patch.object(a, 'worker_command', side_effect=lambda root, task_id:
            [sys.executable, str(Path(__file__).resolve()), '--fixture-worker', str(root), task_id]).start()
        real_popen = subprocess.Popen
        def tracked(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            self.procs.append(proc)
            return proc
        patch.object(a.subprocess, 'Popen', side_effect=tracked).start()
        self.api = patch.object(s, 'call_responses', side_effect=AssertionError('No models in controller')).start()

    def cleanup(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=2)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=2)
            else: proc.wait()
        self.temp.cleanup()

    def write(self, name, value):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value if isinstance(value, str) else json.dumps(value))
        return str(target)

    def task(self, task_id, **updates):
        return {'task_id': task_id, 'task': 'ok', 'cwd': str(self.root), 'max_steps': 4,
                'context': '.3', **updates}

    def call(self, **args):
        return s.handle_tools_call({'name': 'deepseek_subagent', 'arguments': args})

    def batch(self, tasks, batch_id='batch'):
        path = self.write(batch_id+'.json', {'schema': 'DEEPSEEK_BATCH_V1', 'batch_id': batch_id, 'tasks': tasks})
        return self.call(action='batch', batch_path=path)

    def until(self, task_id=None, batch_id=None, predicate=None, timeout=8):
        selector = {'task_id': task_id} if task_id else {'batch_id': batch_id}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.call(action='status', **selector)
            self.assertFalse(result['isError'], result)
            meta = result['structuredContent']
            if predicate(meta) if predicate else not any(x['status'] in a.ACTIVE for x in meta['tasks']):
                return result
            time.sleep(.03)
        self.fail('Timed out: '+str(result))

    def events(self):
        path = self.runs / 'events.jsonl'
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def contract(self, name, read_paths, write_paths=()):
        return self.write(name+'.json', {'schema':'DEEPSEEK_TASK_V2', 'objective':'bounded fixture',
            'acceptance':['actual fixture report'], 'read_paths':read_paths,
            'write_paths':list(write_paths), 'commands':[]})

    def test_submit_returns_before_work_and_exposes_result(self):
        started = time.monotonic()
        result = self.call(action='submit', **self.task('one', context='1'))
        self.assertFalse(result['isError'], result)
        self.assertLess(time.monotonic()-started, .8)
        self.assertIn(result['structuredContent']['status'], a.ACTIVE)
        observed = self.until(task_id='one')
        done = observed['structuredContent']['tasks'][0]
        self.assertEqual(done['status'], 'completed')
        self.assertEqual(done['execution']['steps'], 1)
        self.assertEqual(done['execution']['progress']['writes_returned'], 0)
        self.assertNotIn('action_records', done['execution'])
        self.assertTrue(Path(done['result']['structuredContent']['report_path']).is_file())
        self.assertEqual(done['report_path'],done['result']['structuredContent']['report_path'])
        self.assertIn('verified fixture evidence', '\n'.join(x['text'] for x in observed['content']))
        self.assertEqual(self.events()[0]['model'], 'deepseek-flash')
        self.assertEqual(self.events()[0]['effort'], 'max')
        self.api.assert_not_called()

    def test_six_jobs_really_overlap_but_never_exceed_three(self):
        result = self.batch([self.task(str(i), context='.6') for i in range(6)])
        self.assertFalse(result['isError'], result)
        done = self.until(batch_id='batch')
        self.assertTrue(done['structuredContent']['all_completed'])
        self.assertTrue(all(x['report_path'] for x in done['structuredContent']['tasks']))
        self.assertNotIn('verified fixture evidence',json.dumps(done['content']))
        timeline = sorted((t, change) for row in self.events() for t, change in ((row['start'],1),(row['end'],-1)))
        count = peak = 0
        for _, change in timeline: count += change; peak = max(peak,count)
        self.assertEqual(peak,3)
        self.assertEqual(len(self.events()),6)

    def test_repeat_batch_and_submit_never_launch_twice(self):
        tasks = [self.task('one'), self.task('two')]
        self.batch(tasks)
        self.batch(tasks)
        self.until(batch_id='batch')
        self.batch(tasks)
        result = self.call(action='submit', **tasks[0])
        self.assertEqual(result['structuredContent']['status'],'completed')
        self.assertEqual(len(self.procs),2)
        self.assertEqual(len(self.events()),2)

    def test_id_conflict_and_execution_mode_switch_are_rejected(self):
        self.call(action='submit', **self.task('one'))
        result = self.call(action='submit', **self.task('one', task='changed'))
        self.assertTrue(result['isError'])
        self.assertTrue(self.call(action='native_request', **self.task('one'))['isError'])
        self.call(action='run', **self.task('one'))
        self.until(task_id='one')
        self.assertEqual(len(self.procs),1)

    def test_dependencies_enforce_order(self):
        self.batch([self.task('a'), self.task('b',depends_on=['a'])])
        self.until(batch_id='batch')
        rows={x['task_id']:x for x in self.events()}
        self.assertGreaterEqual(rows['b']['start'],rows['a']['end'])

    def test_failed_dependency_prevents_model_call(self):
        self.batch([self.task('a',task='fail'), self.task('b',depends_on=['a'])])
        result=self.until(batch_id='batch')
        self.assertEqual(result['structuredContent']['tasks'][1]['status'],'blocked_dependency')
        self.assertEqual([x['task_id'] for x in self.events()],['a'])

    def test_cycles_unknown_dependencies_and_malformed_batch_rejected(self):
        cases = [[self.task('a',depends_on=['b']),self.task('b',depends_on=['a'])],
                 [self.task('a',depends_on=['absent'])], [self.task('a'),self.task('a')],
                 [self.task('a',depends_on='b')], [self.task('a',allow_shell='yes')]]
        for tasks in cases: self.assertTrue(self.batch(tasks)['isError'])
        self.assertFalse(self.procs)
        self.assertFalse(a.jobs(self.runs))

    def test_writing_requires_contract(self):
        for flag in ('allow_write','allow_shell'):
            self.assertTrue(self.call(action='submit',**self.task(flag,**{flag:True}))['isError'])
        self.assertFalse(self.procs)

    def test_conflicting_reader_and_writer_are_serialized(self):
        source = self.write('source.txt','fixture')
        read = self.contract('read',[source])
        write = self.contract('write',[source],[source])
        self.batch([self.task('reader',contract_path=read), self.task('writer',contract_path=write,allow_write=True)])
        self.until(batch_id='batch')
        rows=sorted(self.events(),key=lambda x:x['start'])
        self.assertGreaterEqual(rows[1]['start'],rows[0]['end'])

    def test_disjoint_write_scopes_can_overlap(self):
        tasks=[]
        for name in ('a','b'):
            source=self.write(name+'.txt','fixture')
            contract=self.contract(name+'contract',[source],[source])
            tasks.append(self.task(name,contract_path=contract,allow_write=True,context='.6'))
        self.batch(tasks); self.until(batch_id='batch')
        rows=self.events()
        self.assertLess(max(x['start'] for x in rows),min(x['end'] for x in rows))

    def test_queue_capacity_rejects_entire_batch(self):
        with patch.object(a,'config',return_value={'max_workers':3,'max_pending':1,'queue_timeout_sec':3600}):
            self.assertTrue(self.batch([self.task('a'),self.task('b')])['isError'])
        self.assertFalse(self.procs)
        self.assertFalse(a.jobs(self.runs))

    def test_cancel_running_stops_worker_and_keeps_partial_receipt(self):
        self.call(action='submit',**self.task('a',context='4'))
        self.until(task_id='a',predicate=lambda m:m['status']=='running')
        time.sleep(.1)
        self.call(action='cancel',task_id='a')
        result=self.until(task_id='a')
        self.assertEqual(result['structuredContent']['status'],'cancelled')
        self.assertEqual(len(self.procs),1)
        self.assertIsNotNone(result['structuredContent']['tasks'][0]['result'])

    def test_cancel_queued_does_not_call_model(self):
        self.batch([self.task('a',context='1'),self.task('b',depends_on=['a'])])
        self.call(action='cancel',task_id='b')
        self.until(batch_id='batch')
        self.assertEqual([x['task_id'] for x in self.events()],['a'])

    def test_live_clone_progress_and_deliverable_survive_real_worker(self):
        self.write('source.py', 'old\n')
        contract = self.write('delivery-contract.json', {'schema': 'DEEPSEEK_TASK_V2',
            'objective': 'bounded clone', 'acceptance': ['new output'], 'read_paths': ['source.py'],
            'write_paths': ['candidate.py'], 'commands': [],
            'deliverables': [{'path': 'candidate.py', 'must_change': True}]})
        self.call(action='submit', **self.task('delivery', task='clone', context='.6',
                                             contract_path=contract, allow_write=True))
        live = self.until(task_id='delivery', predicate=lambda m:
            m['status'] == 'running' and m['tasks'][0].get('execution', {}).get('progress', {}).get('writes_returned') == 1)
        self.assertFalse(live['structuredContent']['all_completed'])
        self.assertNotIn('old', json.dumps(live['structuredContent']['tasks'][0]['execution']))
        done = self.until(task_id='delivery')
        self.assertEqual(done['structuredContent']['status'], 'completed')
        self.assertEqual((self.root/'candidate.py').read_text(), 'new\n')
        self.assertTrue(done['structuredContent']['tasks'][0]['result']['structuredContent']['acceptance_checks'][0]['passed'])
        self.call(action='submit', **self.task('delivery', task='clone', context='.6',
                                             contract_path=contract, allow_write=True))
        self.assertEqual(len(self.procs), 1)
        self.api.assert_not_called()

    def test_crashed_worker_is_unknown_and_never_restarted(self):
        task=self.task('a',task='crash')
        self.call(action='submit',**task)
        result=self.until(task_id='a')
        self.assertEqual(result['structuredContent']['status'],'outcome_unknown')
        self.call(action='submit',**task)
        self.assertEqual(len(self.procs),1)

    def test_wait_returns_at_state_change(self):
        self.call(action='submit',**self.task('a',context='.6'))
        self.until(task_id='a',predicate=lambda m:m['status']=='running')
        started=time.monotonic()
        result=self.call(action='wait',task_id='a',wait_sec=3)
        self.assertLess(time.monotonic()-started,2)
        self.assertEqual(result['structuredContent']['status'],'completed')

    def test_changed_queued_contract_is_rejected_before_model(self):
        contract=self.contract('contract',[str(self.root)+'/'])
        self.batch([self.task('a',context='.6'),self.task('b',contract_path=contract,depends_on=['a'])])
        Path(contract).write_text(Path(contract).read_text()+'\n')
        result=self.until(batch_id='batch')
        self.assertEqual(result['structuredContent']['tasks'][1]['status'],'preflight_failed')
        self.assertEqual([x['task_id'] for x in self.events()],['a'])

    def test_async_host_claim_and_resume_preserve_budget_and_identity(self):
        host=self.write('host.json',{'schema':'DEEPSEEK_HOST_V1','capabilities':[
            {'id':'read','description':'read source','permission':'read'}]})
        self.call(action='submit',**self.task('h',task='host',host_context_path=host,max_output_tokens=1024,output_budget=1500))
        result=self.until(task_id='h')['structuredContent']['tasks'][0]['result']
        request_id=result['structuredContent']['host_request']['request_id']
        output=self.write('host-result.json',{'request_id':request_id,'status':'completed','text':'actual fixture source','sources':[]})
        self.assertTrue(self.call(action='resume',task_id='h',host_result_path=output)['isError'])
        self.assertFalse(self.call(action='claim',task_id='h',request_id=request_id)['isError'])
        self.assertTrue(self.call(action='resume',task_id='h',host_result_path=output,output_budget=9999)['isError'])
        self.assertFalse(self.call(action='resume',task_id='h',host_result_path=output)['isError'])
        self.call(action='resume',task_id='h',host_result_path=output)
        done=self.until(task_id='h')
        self.assertEqual(done['structuredContent']['status'],'completed',done)
        self.assertEqual(len(self.procs),2)
        rows=self.events(); self.assertEqual(len(rows),2)
        self.assertTrue(all(x['output_limit']<=1024 for x in rows))
        self.assertTrue(all(x['effort']=='max' for x in rows))

    def test_missing_absolute_cwd_invalid_wait_and_async_parameter_types(self):
        self.assertTrue(self.call(action='submit',**self.task('a',cwd='.'))['isError'])
        self.assertTrue(self.call(action='wait',task_id='a',wait_sec=21)['isError'])
        self.assertTrue(self.call(action='submit',**self.task('a',allow_write=1))['isError'])
        self.assertFalse(self.procs)

    def test_status_without_submission_never_starts_process(self):
        self.assertEqual(self.call(action='status',task_id='absent')['structuredContent']['status'],'not_found')
        self.assertFalse(self.procs)
        self.api.assert_not_called()

    def test_native_stdio_disconnect_leaves_background_job_queryable(self):
        env={**os.environ,'DEEPSEEK_SUBAGENT_RUNS_DIR':str(self.runs)}
        message={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
            'name':'deepseek_subagent','arguments':{'action':'submit',**self.task('detached',context='.8')}}}
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fixture-server'],
                            input=json.dumps(message)+'\n',text=True,capture_output=True,env=env,timeout=3)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertFalse(json.loads(proc.stdout)['result']['isError'])
        self.assertEqual(self.until(task_id='detached')['structuredContent']['status'],'completed')
        self.assertEqual(len(self.events()),1)

    def test_same_task_submitted_by_two_mcp_processes_executes_once(self):
        from concurrent.futures import ThreadPoolExecutor
        env={**os.environ,'DEEPSEEK_SUBAGENT_RUNS_DIR':str(self.runs)}
        message={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
            'name':'deepseek_subagent','arguments':{'action':'submit',**self.task('same',context='.6')}}}
        def invoke():
            return subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fixture-server'],
                                  input=json.dumps(message)+'\n',text=True,capture_output=True,env=env,timeout=3)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(invoke) for _ in range(2)]
            for future in futures:
                result=future.result()
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertFalse(json.loads(result.stdout)['result']['isError'])
        self.until(task_id='same')
        self.assertEqual(len(self.events()),1)

    def test_client_does_not_invent_task_id_for_batch_or_batch_status(self):
        import client
        for arguments in ({'action':'batch','batch_path':self.write('invalid.json',{'schema':'invalid'})},
                          {'action':'status','batch_id':'absent'}):
            input_path=self.write('client-args.json',arguments)
            output_path=str(self.root/('client-'+arguments['action']+'.json'))
            with patch.object(sys,'argv',['client.py','--arguments',input_path,'--result',output_path]):
                self.assertEqual(client.main(),1)
            stored=json.loads(Path(output_path).read_text())
            self.assertIsNone(stored['task_id'])
            message=stored['response']['result']['content'][0]['text']
            self.assertNotIn('only accepts batch_path',message)
            self.assertNotIn('exactly one',message)

    def test_cancel_claimed_host_retains_unknown_scope(self):
        host=self.write('host.json',{'schema':'DEEPSEEK_HOST_V1','capabilities':[
            {'id':'read','description':'read source','permission':'read'}]})
        self.call(action='submit',**self.task('h',task='host',host_context_path=host))
        result=self.until(task_id='h')['structuredContent']['tasks'][0]['result']
        request_id=result['structuredContent']['host_request']['request_id']
        self.call(action='claim',task_id='h',request_id=request_id)
        result=self.call(action='cancel',task_id='h')
        self.assertEqual(result['structuredContent']['status'],'outcome_unknown')

    def test_queue_timeout_does_not_call_model(self):
        self.batch([self.task('a',context='.8'),self.task('b',depends_on=['a'])])
        with a.queue_lock(self.runs):
            target=a.job_path(self.runs,'b'); job=state.read(target)
            job['queued_at']=time.time()-3700; state.atomic_json(target,job)
        result=self.until(batch_id='batch')
        self.assertEqual(result['structuredContent']['tasks'][1]['status'],'queue_timeout')
        self.assertEqual([x['task_id'] for x in self.events()],['a'])

    def test_cancelled_host_cannot_be_claimed_later(self):
        host=self.write('host.json',{'schema':'DEEPSEEK_HOST_V1','capabilities':[
            {'id':'read','description':'read source','permission':'read'}]})
        self.call(action='submit',**self.task('h',task='host',host_context_path=host))
        result=self.until(task_id='h')['structuredContent']['tasks'][0]['result']
        request_id=result['structuredContent']['host_request']['request_id']
        self.call(action='cancel',task_id='h')
        self.assertTrue(self.call(action='claim',task_id='h',request_id=request_id)['isError'])

    def test_cancel_cleans_approved_command_process_group(self):
        pid_path=self.root/'command.pid'
        code='import os,time,pathlib\npathlib.Path('+repr(str(pid_path))+').write_text(str(os.getpid()))\ntime.sleep(30)\n'
        script=self.write('sleeper.py',code)
        contract=self.contract('contract',[str(self.root)+'/'],[str(self.root)+'/'])
        value=json.loads(Path(contract).read_text())
        value['commands']=[{'id':'sleep','argv':[sys.executable,script],'cwd':str(self.root),
                            'timeout_sec':40,'inputs':[p.pin(script)]}]
        Path(contract).write_text(json.dumps(value))
        self.call(action='submit',**self.task('sleep',task='shell',allow_shell=True,contract_path=contract))
        deadline=time.monotonic()+3
        while not pid_path.exists() and time.monotonic()<deadline: time.sleep(.03)
        self.assertTrue(pid_path.exists())
        pid=int(pid_path.read_text())
        self.call(action='cancel',task_id='sleep')
        result=self.until(task_id='sleep')
        self.assertEqual(result['structuredContent']['status'],'cancelled',result)
        self.assertIsNone(a.process_identity(pid))


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--fixture-worker': fixture_worker(sys.argv[2],sys.argv[3])
    elif len(sys.argv)==2 and sys.argv[1]=='--fixture-server':
        with patch.object(a,'worker_command',side_effect=lambda root,task_id:
                [sys.executable,str(Path(__file__).resolve()),'--fixture-worker',str(root),task_id]), \
             patch.object(s,'call_responses',side_effect=AssertionError('No live model in fixture MCP')):
            s.main()
    else: unittest.main()
