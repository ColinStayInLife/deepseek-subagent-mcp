"""Durable, bounded background jobs for the direct DeepSeek MCP.

No hidden model or host-tool calls: a worker runs the existing server loop.
The stdio process only submits/observes. Locks and receipts survive disconnects.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import project_support as project
import state_store as state

ACTIVE = {'queued', 'running', 'cancel_requested'}
RESERVED = ACTIVE | {'needs_host', 'outcome_unknown'}
RUN_KEYS = {'task_id', 'task', 'context', 'cwd', 'contract_path', 'host_context_path',
            'model', 'effort', 'max_steps', 'timeout_sec', 'max_output_tokens',
            'output_budget', 'input_budget', 'allow_write', 'allow_shell'}
CONFIG_PATH = Path(__file__).with_name('scheduler.json')


def config():
    value = state.read(CONFIG_PATH, {'max_workers': 3, 'max_pending': 24, 'queue_timeout_sec': 3600})
    bounds = {'max_workers': (1, 8), 'max_pending': (1, 64), 'queue_timeout_sec': (30, 86400)}
    if set(value) != set(bounds) or any(type(value[k]) is not int or not lo <= value[k] <= hi for k, (lo, hi) in bounds.items()):
        raise ValueError('Invalid scheduler.json; workers 1..8, pending 1..64, queue timeout 30..86400')
    return value


def directory(root, task_id):
    return Path(root) / 'async' / 'jobs' / state.task_directory(root, task_id).name


def job_path(root, task_id):
    return directory(root, task_id) / 'job.json'


def batch_path(root, batch_id):
    return Path(root) / 'async' / 'batches' / (state.task_directory(root, batch_id).name + '.json')


def jobs(root):
    return [state.read(p) for p in (Path(root) / 'async' / 'jobs').glob('*/job.json')]


@contextmanager
def acquire_lock(path):
    deadline = time.monotonic() + 5
    while True:
        guard = state.lock(path)
        try:
            guard.__enter__()
            break
        except ValueError:
            if time.monotonic() >= deadline:
                raise ValueError('Queue is busy; inspect status before submitting again')
            time.sleep(.02)
    try:
        yield
    finally:
        guard.__exit__(None, None, None)


def queue_lock(root):
    return acquire_lock(Path(root) / 'async' / 'queue.lock')


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        return {'pid': pid, 'start_ticks': fields[19]}
    except (OSError, IndexError):
        return None


def alive(job):
    ident = job.get('process')
    return bool(ident and process_identity(ident['pid']) == ident)


def signal_worker(job):
    if not alive(job): return
    if hasattr(os, 'pidfd_open'):
        fd = os.pidfd_open(job['process']['pid'])
    else:
        # Some local Python builds omit os.pidfd_open although Linux/glibc
        # provide it. Use the same kernel handle, never a bare recycled PID.
        libc = ctypes.CDLL(None, use_errno=True)
        open_fd = libc.pidfd_open
        open_fd.argtypes = [ctypes.c_int, ctypes.c_uint]
        open_fd.restype = ctypes.c_int
        fd = open_fd(job['process']['pid'], 0)
        if fd < 0: raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    try:
        if alive(job): signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)


def observed(job):
    job = dict(job)
    dispatching = job['status'] == 'queued' and not job.get('process') and time.time() - job.get('queued_at', 0) < 5
    if job['status'] in ACTIVE and not alive(job) and not dispatching:
        job['status'] = 'outcome_unknown'
        job['detail'] = 'Worker exited without a committed result; inspect receipts, never resubmit blindly.'
    return job


def result_status(result):
    status = result.get('structuredContent', {}).get('status', 'error')
    return 'error' if status == 'completed' and result.get('isError') else status


def prepare(args, root):
    """Validate and freeze identity, budgets and permission scopes before enqueue."""
    import server
    server.validate_arguments(args)
    if set(args) - RUN_KEYS:
        raise ValueError('Async task contains non-run arguments')
    if not args.get('task', '').strip(): raise ValueError('Async task needs nonempty task')
    state.task_directory(root, args.get('task_id'))
    if not Path(args.get('cwd', '')).is_absolute() or not Path(args['cwd']).is_dir():
        raise ValueError('Async task needs an existing absolute cwd')
    cwd = str(Path(args['cwd']).resolve())
    settings, settings_pin = project.project_config(cwd)
    defaults = settings.get('defaults', {})
    budget_keys = {'max_steps', 'timeout_sec', 'max_output_tokens', 'output_budget', 'input_budget'}
    if not isinstance(defaults, dict) or set(defaults) - budget_keys:
        raise ValueError('Project defaults may only configure execution budgets')
    server.validate_arguments(defaults)
    normalized = {**defaults, **args, 'cwd': cwd, 'action': 'run',
                  'model': args.get('model', server.DEFAULT_MODEL),
                  'effort': args.get('effort', server.DEFAULT_EFFORT)}
    for key in ('contract_path', 'host_context_path'):
        if normalized.get(key): normalized[key] = str(project.resolve(normalized[key], cwd))
    server.validate_arguments(normalized)
    contract = project.load_contract(normalized.get('contract_path'), cwd)
    mutates = args.get('allow_write', False) or args.get('allow_shell', False)
    if mutates and contract is None:
        raise ValueError('Async write/shell requires a contract for conflict scheduling')
    # Validate explicit host material now; no API or credentials are accessed.
    import host_bridge
    packet = host_bridge.load_packet(normalized.get('host_context_path'), cwd, contract)
    if packet['images'] and normalized['model'] != 'deepseek-flash':
        raise ValueError('Pinned image input requires deepseek-flash')
    pins = [x for x in (settings_pin, contract['pin'] if contract else None, packet['pin']) if x]
    if contract:
        scopes = [{'path': path, 'write': key == 'write_paths'}
                  for key in ('read_paths', 'write_paths') if key == 'read_paths' or mutates
                  for path, _ in contract['scopes'][key]]
    else:
        # Uncontracted readers are unrestricted, so exclude concurrent writers.
        scopes = [{'path': '/', 'write': False}]
    if args.get('allow_shell'):
        scopes += [{'path': cwd, 'write': True}]
        scopes += [{'path': str(Path(c['cwd']).resolve()), 'write': True}
                   for c in contract['value']['commands']]
    return {'arguments': normalized, 'pins': pins, 'scopes': scopes}


def conflicts(first, second):
    for a in first['scopes']:
        for b in second['scopes']:
            if not (a['write'] or b['write']): continue
            pa, pb = Path(a['path']), Path(b['path'])
            if pa == pb or pa.is_relative_to(pb) or pb.is_relative_to(pa): return True
    return False


def worker_command(root, task_id):
    return [sys.executable, str(Path(__file__).resolve()), '--worker', str(Path(root).resolve()), task_id]


def launch(root, job):
    """Called under queue.lock. Persist intent before creating the process."""
    target = job_path(root, job['task_id'])
    job.update(status='queued', queued_at=time.time(), process=None, result=None, cancel=False)
    state.atomic_json(target, job)
    try:
        log_path = directory(root, job['task_id']) / 'worker.log'
        with log_path.open('ab') as stream:
            os.chmod(log_path, 0o600)
            proc = subprocess.Popen(worker_command(root, job['task_id']), stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=stream, start_new_session=True,
                                    env={**os.environ, 'DEEPSEEK_SUBAGENT_RUNS_DIR': str(Path(root).resolve())})
        job['process'] = process_identity(proc.pid)
        # The worker waits on queue.lock before acting. Losing this receipt is
        # therefore visible and cannot cause a silently duplicated run.
        state.atomic_json(target, job)
    except OSError as exc:
        job.update(status='dispatch_failed', detail=str(exc))
        state.atomic_json(target, job)


def submit(root, tasks, batch_id=None):
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 24:
        raise ValueError('Batch requires 1..24 tasks')
    candidates = []
    for raw in tasks:
        if not isinstance(raw, dict): raise ValueError('Each task must be an object')
        args = dict(raw)
        deps = args.pop('depends_on', [])
        if not isinstance(deps, list) or len(deps) > 24 or not all(isinstance(x, str) and x for x in deps) or len(set(deps)) != len(deps):
            raise ValueError('depends_on must contain unique task IDs')
        prepared = prepare(args, root)
        candidates.append({**prepared, 'task_id': args['task_id'], 'depends_on': deps,
                           'fingerprint': state.digest({'arguments': prepared['arguments'], 'depends_on': deps}),
                           'operation': prepared['arguments']})
    ids = [x['task_id'] for x in candidates]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate task IDs in batch')
    if batch_id is not None: batch_path(root, batch_id)
    with queue_lock(root):
        existing = {x['task_id']: observed(x) for x in jobs(root)}
        combined = {**existing, **{x['task_id']: x for x in candidates}}
        def visit(task_id, visiting, done):
            if task_id in visiting: raise ValueError('Dependency cycle')
            if task_id in done: return
            if task_id not in combined: raise ValueError('Unknown dependency: ' + task_id)
            for dep in combined[task_id]['depends_on']: visit(dep, visiting | {task_id}, done)
            done.add(task_id)
        done = set()
        for task_id in ids: visit(task_id, set(), done)
        for job in candidates:
            old = existing.get(job['task_id'])
            if old and old['fingerprint'] != job['fingerprint']:
                raise ValueError('task_id already belongs to different arguments: ' + job['task_id'])
            if not old and state.read(state.task_directory(root, job['task_id']) / 'task.json'):
                raise ValueError('task_id already used by a synchronous/native task; inspect status')
        new = [x for x in candidates if x['task_id'] not in existing]
        if sum(x['status'] in RESERVED for x in existing.values()) + len(new) > config()['max_pending']:
            raise ValueError('Async queue full; collect/reconcile existing tasks first')
        if batch_id is not None:
            descriptor = {'batch_id': batch_id, 'task_ids': ids,
                          'fingerprint': state.digest([x['fingerprint'] for x in candidates])}
            old = state.read(batch_path(root, batch_id))
            if old and old != descriptor: raise ValueError('batch_id already belongs to another batch')
            state.atomic_json(batch_path(root, batch_id), descriptor)
        # Save the WHOLE batch first: a crash leaves visible unknown jobs and
        # never an unrecorded paid execution or silently missing dependency.
        for job in new:
            job.update(status='queued', queued_at=time.time(), process=None, result=None, cancel=False)
            state.atomic_json(job_path(root, job['task_id']), job)
        for job in new: launch(root, job)
    return snapshot(root, {'batch_id': batch_id} if batch_id else {'task_id': ids[0]})


def select(root, args):
    if bool(args.get('task_id')) == bool(args.get('batch_id')):
        raise ValueError('Choose exactly one task_id or batch_id')
    if args.get('batch_id'):
        batch = state.read(batch_path(root, args['batch_id']))
        if not batch: raise ValueError('Unknown batch_id')
        return batch['task_ids']
    return [args['task_id']]


def snapshot(root, args):
    summaries = []
    for task_id in select(root, args):
        job = state.read(job_path(root, task_id))
        if not job: raise ValueError('Unknown async task_id: ' + task_id)
        job = observed(job)
        journal = state.read(state.task_directory(root, task_id) / 'task.json', {})
        summary = {k: job.get(k) for k in ('task_id', 'status', 'depends_on', 'detail')}
        summary.update(result=job.get('result'), receipt_path=journal.get('receipt_path'),
                       job_path=str(job_path(root, task_id)), controller_acceptance_required=True)
        result_meta = (job.get('result') or {}).get('structuredContent', {})
        summary.update(report_path=result_meta.get('report_path'), host_request=result_meta.get('host_request'))
        if args.get('batch_id') and summary['result']:
            summary['result'] = {k: v for k, v in summary['result'].items() if k != 'content'}
        summaries.append(summary)
    statuses = [x['status'] for x in summaries]
    status = (statuses[0] if len(summaries) == 1 else 'completed' if all(x == 'completed' for x in statuses)
              else 'running' if any(x in ACTIVE for x in statuses)
              else 'needs_host' if 'needs_host' in statuses else 'incomplete')
    meta = {'status': status, 'tasks': summaries, 'max_workers': config()['max_workers'],
            'all_completed': all(x == 'completed' for x in statuses),
            'controller_acceptance_required': True, 'billing': 'direct-deepseek-api',
            'notification': 'No native push; controller uses bounded wait/status while its turn is active.'}
    meta.update({k: args[k] for k in ('task_id', 'batch_id') if k in args})
    brief = [{k: v for k, v in x.items() if k != 'result'} for x in summaries]
    # isError describes this observation, not whether scientific work passed.
    content = [{'type': 'text', 'text': json.dumps({'status': status, 'tasks': brief}, ensure_ascii=False)}]
    if not args.get('batch_id') and summaries[0]['result']:
        content += summaries[0]['result'].get('content', [])
    return {'content': content,
            'isError': False, 'structuredContent': meta}


def wait(root, args):
    before = snapshot(root, args)
    signature = lambda r: [(x['status'], (x.get('result') or {}).get('structuredContent', {}).get('host_request')) for x in r['structuredContent']['tasks']]
    deadline = time.monotonic() + args.get('wait_sec', 15)
    while time.monotonic() < deadline:
        if not any(x['status'] in ACTIVE for x in before['structuredContent']['tasks']): break
        time.sleep(min(.2, max(0, deadline-time.monotonic())))
        current = snapshot(root, args)
        if signature(current) != signature(before): return current
    return snapshot(root, args)


def cancel(root, args):
    with queue_lock(root):
        selected = select(root, args)
        for task_id in selected:
            target = job_path(root, task_id)
            job = state.read(target)
            if not job: raise ValueError('Unknown async task_id: ' + task_id)
            job = observed(job)
            if job['status'] == 'needs_host' and state.read(state.task_directory(root, task_id) / 'task.json', {}).get('host_claim'):
                job.update(status='outcome_unknown', cancel=True, detail='Claimed host action may still be executing; reconcile its result before releasing scope.')
            elif job['status'] == 'queued' or job['status'] == 'needs_host':
                job.update(status='cancelled', cancel=True, detail='Cancelled; existing files and host actions are not rolled back.')
            elif job['status'] in ('running', 'cancel_requested'):
                job.update(status='cancel_requested', cancel=True,
                           detail='Stop requested; partial writes or API billing may already have occurred.')
                # pidfd binds the signal to this exact process, avoiding PID reuse.
                try:
                    signal_worker(job)
                except ProcessLookupError:
                    pass
                except (OSError, AttributeError) as exc:
                    job['detail'] += ' Immediate signal unavailable: ' + str(exc)
            state.atomic_json(target, job)
    return snapshot(root, args)


def resume(root, args):
    if set(args) - {'action', 'task_id', 'host_result_path'}:
        raise ValueError('Resume cannot change task arguments or budgets')
    with queue_lock(root):
        target = job_path(root, args['task_id'])
        job = state.read(target)
        source = project.pin(args.get('host_result_path'))
        operation = {'action': 'resume', 'task_id': args['task_id'], 'host_result_path': source['path']}
        if job.get('resume_source') == source:
            return snapshot(root, {'task_id': args['task_id']})
        if job['status'] != 'needs_host': raise ValueError('Only needs_host can resume; inspect previous outcomes')
        journal = state.read(state.task_directory(root, args['task_id']) / 'task.json', {})
        if journal.get('status') != 'needs_host' or not journal.get('host_claim'):
            raise ValueError('Claim the pending host request before resume')
        import host_bridge
        request = journal['result']['structuredContent']['host_request']
        contract = project.load_contract(job['arguments'].get('contract_path'), job['arguments']['cwd'])
        additions, _ = host_bridge.result_packet(source['path'], request, job['arguments']['cwd'], contract)
        if json.loads(additions[0]['output'])['status'] == 'unknown':
            raise ValueError('Host result unknown; reconcile before resuming')
        job.update(operation=operation, resume_source=source)
        launch(root, job)
    return snapshot(root, {'task_id': args['task_id']})


def route(root, args):
    action = args.get('action', 'run')
    task_owned = bool(args.get('task_id') and job_path(root, args['task_id']).is_file())
    if action == 'submit':
        return submit(root, [{k: v for k, v in args.items() if k != 'action'}])
    if action == 'batch':
        if set(args) - {'action', 'batch_path'}: raise ValueError('batch only accepts batch_path')
        if not Path(args.get('batch_path', '')).is_absolute(): raise ValueError('batch_path must be absolute')
        data = project.read_json(args['batch_path'], 1024 * 1024)
        if not isinstance(data, dict) or set(data) != {'schema', 'batch_id', 'tasks'} or data['schema'] != 'DEEPSEEK_BATCH_V1':
            raise ValueError('Invalid DEEPSEEK_BATCH_V1')
        return submit(root, data['tasks'], data['batch_id'])
    if action in ('status', 'wait', 'cancel') and (action != 'status' or task_owned or args.get('batch_id')):
        allowed = {'action', 'task_id', 'batch_id'} | ({'wait_sec'} if action == 'wait' else set())
        if set(args) - allowed: raise ValueError('Unexpected observation/cancel arguments')
        return {'status': snapshot, 'wait': wait, 'cancel': cancel}[action](root, args)
    if task_owned and action == 'run':
        expected = state.read(job_path(root, args['task_id']))['arguments']
        if {k: v for k, v in expected.items() if k in args} != args:
            raise ValueError('Task already queued with different arguments; inspect status')
        return snapshot(root, {'task_id': args['task_id']})
    if task_owned and action == 'resume': return resume(root, args)
    if task_owned and action == 'claim':
        job = state.read(job_path(root, args['task_id']))
        if job['status'] != 'needs_host' or job.get('cancel'):
            raise ValueError('Async task has no claimable host request')
    if task_owned and action not in ('status', 'claim'):
        raise ValueError('Async-owned task cannot switch execution mode')
    return None


def worker(root, task_id):
    import server
    server.RUNS_DIR = Path(root)
    target = job_path(root, task_id)
    with acquire_lock(directory(root, task_id) / 'worker.lock'):
        while True:
            with queue_lock(root):
                job = state.read(target)
                if job['status'] != 'queued' or job.get('cancel'): return
                if job.get('process') != process_identity(os.getpid()): return
                inventory = {x['task_id']: observed(x) for x in jobs(root)}
                deps = [inventory[x] for x in job['depends_on']]
                if any(x['status'] not in ACTIVE | {'needs_host', 'completed'} for x in deps):
                    job.update(status='blocked_dependency', detail='A prerequisite did not complete; Astra must inspect its evidence.')
                    state.atomic_json(target, job); return
                if time.time() - job['queued_at'] > config()['queue_timeout_sec']:
                    job.update(status='queue_timeout', detail='Queue waiting budget expired; no model call for this queued segment.')
                    state.atomic_json(target, job); return
                busy = [x for x in inventory.values() if x['task_id'] != task_id]
                active = sum(x['status'] in {'running', 'cancel_requested'} or (x['status'] == 'outcome_unknown' and alive(x)) for x in busy)
                blockers = [x for x in busy if x['status'] in {'running', 'cancel_requested', 'needs_host', 'outcome_unknown'}]
                if all(x['status'] == 'completed' for x in deps) and active < config()['max_workers'] and not any(conflicts(job, other) for other in blockers):
                    # Install before committing running: cancel cannot hit the
                    # gap before cleanup-aware signal handling is available.
                    def stop(signum, frame):
                        raise server.DeadlineExceeded('Async cancellation requested')
                    signal.signal(signal.SIGTERM, stop)
                    job.update(status='running', started_at=time.time())
                    state.atomic_json(target, job)
                    break
            time.sleep(.2)
        execution_started = False
        try:
            for pin in job['pins'] + ([job['resume_source']] if job.get('resume_source') else []):
                if project.pin(pin['path']) != pin: raise ValueError('Queued configuration/context/contract changed: ' + pin['path'])
            # Revalidate symlink targets and permission scopes, as well as pins.
            fresh = prepare({k: v for k, v in job['arguments'].items() if k != 'action'}, root)
            if fresh['pins'] != job['pins'] or fresh['scopes'] != job['scopes']:
                raise ValueError('Queued project or permission scopes changed')
            execution_started = True
            result = server.handle_tools_call({'name': 'deepseek_subagent', 'arguments': job['operation']}, _async_worker=True)
        except BaseException as exc:
            result = server._error(str(exc), 'outcome_unknown' if execution_started or isinstance(exc, server.DeadlineExceeded) else 'preflight_failed')
        # Once model execution stops, commit the outcome even if another cancel
        # arrives; do not lose the report between API completion and persistence.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with queue_lock(root):
            current = state.read(target)
            status = result_status(result)
            journal = state.read(state.task_directory(root, task_id) / 'task.json', {})
            if journal.get('status') == 'started_outcome_unknown': status = 'outcome_unknown'
            if current.get('cancel') and status not in ('completed', 'outcome_unknown'): status = 'cancelled'
            current.update(status=status, result=result, finished_at=time.time())
            state.atomic_json(target, current)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit('Internal worker entry point; use MCP submit/batch to create jobs')
    worker(Path(sys.argv[2]), sys.argv[3])
