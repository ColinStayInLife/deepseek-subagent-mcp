"""Bounded controller messages, evidence-backed context pruning and handoffs.

No model calls. Private checkpoints contain only the worker's own conversation.
Task instructions, contracts, images and tool call/result pairs are never pruned.
"""
import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path

import project_support as project
import state_store as state

SAFE_STOPS = {'completed', 'acceptance_failed', 'step_limit', 'input_budget',
              'output_budget', 'input_budget_prediction', 'context_budget'}
LOCAL_CHECK_STOPS = {'completed', 'step_limit', 'input_budget', 'output_budget',
                     'input_budget_prediction', 'context_budget', 'tool_limit'}
MAX_MESSAGES = 32
MAX_MESSAGE_BYTES = 8192
MAX_MAILBOX_BYTES = 65536


@contextmanager
def control_lock(path):
    # Only lock acquisition is retried, never a message delivery or tool action.
    deadline = time.monotonic() + 1
    while True:
        guard = state.lock(path)
        try:
            guard.__enter__()
            break
        except ValueError:
            if time.monotonic() >= deadline: raise ValueError('Control mailbox busy; inspect message_id status')
            time.sleep(.005)
    try:
        yield
    finally:
        guard.__exit__(None, None, None)


def instruction(args, id_key):
    ident, text = args.get(id_key), args.get('instruction')
    if not isinstance(ident, str) or not ident.strip() or len(ident) > 160:
        raise ValueError(id_key + ' must contain 1..160 characters')
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > MAX_MESSAGE_BYTES:
        raise ValueError('instruction must contain 1..8192 UTF-8 bytes')
    return ident, text


class Mailbox:
    def __init__(self, root, task_id):
        self.directory = state.task_directory(root, task_id)
        self.path = self.directory / 'control.json'

    def _read(self):
        return state.read(self.path, {'accepting': True, 'messages': []})

    def open(self):
        with control_lock(self.directory / 'control.lock'):
            value = self._read()
            value['accepting'] = True
            state.atomic_json(self.path, value)

    def send(self, message_id, text):
        instruction({'message_id': message_id, 'instruction': text}, 'message_id')
        with control_lock(self.directory / 'control.lock'):
            value = self._read()
            old = next((x for x in value['messages'] if x['message_id'] == message_id), None)
            if old:
                if old['text'] != text: raise ValueError('message_id already has different instructions')
                return {'message_id': message_id, 'applied': old['applied'], 'reused': True}
            if not value['accepting']:
                raise ValueError('Task is finalizing or stopped; inspect status and use followup if eligible')
            if len(value['messages']) >= MAX_MESSAGES or sum(len(x['text'].encode()) for x in value['messages']) + len(text.encode()) > MAX_MAILBOX_BYTES:
                raise ValueError('Task steering mailbox budget exhausted')
            value['messages'].append({'message_id': message_id, 'text': text, 'applied': False})
            state.atomic_json(self.path, value)
            return {'message_id': message_id, 'applied': False, 'reused': False}

    def poll(self, finalize=False, stop=False):
        with control_lock(self.directory / 'control.lock'):
            value = self._read()
            pending = [] if stop else [x for x in value['messages'] if not x['applied']]
            for item in pending: item['applied'] = True
            closing = stop or (finalize and not pending)
            changed = pending or (closing and value['accepting'])
            if closing: value['accepting'] = False
            if changed: state.atomic_json(self.path, value)
            return copy.deepcopy(pending)

    def summary(self):
        value = self._read()
        return {'accepting': value['accepting'], 'messages': [
            {'message_id': x['message_id'], 'applied': x['applied']} for x in value['messages']]}


def append_updates(items, updates, stats, skipped=()):
    if not updates: return False
    # Pair every skipped call before adding a new user message. Already executed
    # tools are not undone; in-flight tools run to their normal boundary.
    for call in skipped:
        items.append({'type': 'function_call_output', 'call_id': call['call_id'],
                      'output': 'SKIPPED: newer controller instruction; this tool was not executed. Re-plan within the original contract.'})
    items.append({'role': 'user', 'content': 'Controller steering (original permissions, contract and budgets still apply):\n' +
                  '\n'.join(x['text'] for x in updates)})
    stats['steering_applied'] = stats.get('steering_applied', 0) + len(updates)
    stats['stale_calls_skipped'] = stats.get('stale_calls_skipped', 0) + len(skipped)
    return True


def paired(items):
    pending = set()
    for item in items:
        if item.get('type') == 'function_call':
            if item.get('call_id') in pending: return False
            pending.add(item.get('call_id'))
        elif item.get('type') == 'function_call_output':
            if item.get('call_id') not in pending: return False
            pending.remove(item['call_id'])
    return not pending


def can_checkpoint(items, stats):
    return (stats.get('status') in SAFE_STOPS and stats.get('usage_complete') and paired(items)
            and not any(x.get('status') == 'started_outcome_unknown' for x in stats.get('action_records', [])))


def load_followup(journal):
    if not journal or journal.get('status') not in SAFE_STOPS:
        raise ValueError('Only a verified safe stopping point can follow up; unknown outcomes require reconciliation')
    source = (journal.get('result') or {}).get('structuredContent', {}).get('followup_checkpoint')
    if not source or project.pin(source['path']) != source:
        raise ValueError('Missing or changed followup checkpoint')
    value = state.read(source['path'])
    stats = value['stats']
    if not can_checkpoint(value['items'], stats): raise ValueError('Checkpoint is not a safe continuation')
    budgets = stats['budgets']
    if (stats['steps'] >= budgets['max_steps'] or stats['elapsed_sec'] >= budgets['timeout_sec']
            or stats['input_tokens'] >= budgets['input_budget']
            or budgets['output_budget'] - stats['output_tokens'] < 128):
        raise ValueError('Original task budget exhausted; followup cannot reset it')
    return value


def prune_outputs(items, records, keep_recent=4):
    """Prune only older successful local results with verified full evidence.

    The complete transcript is stored privately by the caller before replacing
    anything. This is syntactic pruning, not semantic or lossless summarization.
    """
    outputs = [i for i, x in enumerate(items) if x.get('type') == 'function_call_output']
    by_index = {x.get('history_index'): x for x in records}
    updated, count, saved = list(items), 0, 0
    for index in outputs[:-keep_recent] if keep_recent else outputs:
        item = items[index]
        text = item.get('output')
        action = by_index.get(index, {})
        pin = action.get('result_file')
        if not isinstance(text, str) or len(text) <= 2000 or action.get('status') != 'returned' or not pin:
            continue
        try:
            if project.pin(pin['path']) != pin: continue
        except (ValueError, OSError):
            continue
        trimmed = (text[:512] + '\n[Older tool output pruned; inspect the full result before relying on omitted details.]\n' +
                   json.dumps({'full_result': pin}, ensure_ascii=False) + '\n' + text[-256:])
        if len(trimmed.encode()) >= len(text.encode()): continue
        updated[index] = {**item, 'output': trimmed}
        count += 1
        saved += len(text.encode()) - len(trimmed.encode())
    return updated, {'results': count, 'bytes_removed': saved}


def handoff(stats, report):
    """A compact index of evidence, never a claim that science was validated."""
    return {'schema': 'DEEPSEEK_HANDOFF_V1', 'status': stats['status'],
            'controller_acceptance_required': True, 'report_excerpt': report[:1600],
            'stop_detail': stats.get('stop_detail'),
            'completion': stats.get('completion'),
            'local_verification': stats.get('local_verification'),
            'diagnostic': stats.get('diagnostic'),
            'tool_budget': stats.get('tool_budget'),
            'not_executed_calls': stats.get('not_executed_calls', []),
            'constraints': {'contract': stats.get('task_contract'), 'project': stats.get('project_config')},
            'usage': {k: stats.get(k) for k in ('steps', 'input_tokens', 'output_tokens', 'usage_complete', 'elapsed_sec')},
            'budgets': stats.get('budgets'), 'acceptance_checks': stats.get('acceptance_checks', []),
            'changed_files': [x['output_file'] for x in stats.get('action_records', []) if x.get('output_file')],
            'evidence': stats.get('evidence_files', []),
            'inspection_index': [{k: x[k] for k in ('sequence', 'tool', 'status', 'arguments', 'result_file') if k in x}
                                 for x in stats.get('action_records', [])
                                 if x.get('tool') in {'read_file', 'json_query', 'file_info', 'grep', 'list_dir'}],
            'unresolved_actions': [{k: x.get(k) for k in ('sequence', 'tool', 'status', 'result_file')}
                                   for x in stats.get('action_records', []) if x.get('status') != 'returned']}


def skip_unexecuted_calls(items, calls, stats, reason):
    """Pair only calls the loop KNOWS it has not dispatched. Never repair an
    interrupted/unknown action or malformed API output with a fake result.
    """
    for call in calls:
        record = {'step': stats['steps'], 'call_id': call['call_id'], 'tool': call['name'],
                  'status': 'not_executed', 'reason': reason}
        stats.setdefault('not_executed_calls', []).append(record)
        items.append({'type': 'function_call_output', 'call_id': call['call_id'],
                      'output': 'NOT_EXECUTED: ' + reason + '; no tool was dispatched for this call.'})


def verification_gate(status, items, stats, contract):
    """Eligibility only; the server bounds actual read-only checks by its timer."""
    value = {'status': 'deferred', 'reason': None, 'checked': 0,
             'scope': 'configured_contract_checks_only', 'model_calls': 0, 'commands_run': 0}
    if status not in LOCAL_CHECK_STOPS:
        value['reason'] = 'execution_not_at_verifiable_stop'
    elif not paired(items) or stats.get('host_request') or any(
            x.get('status') == 'started_outcome_unknown' for x in stats.get('action_records', [])):
        value['reason'] = 'unresolved_tool_or_host_action'
    elif not stats.get('usage_complete'):
        value['reason'] = 'usage_incomplete'
    elif not any((contract or {}).get('value', {}).get(key) for key in ('checks', 'deliverables')):
        value.update(status='not_configured', reason='no_machine_checks_declared')
    else:
        value.update(status='eligible')
    return value


def output_diagnostic(response, round_record, stats):
    """Explain a truncated generation using usage counters, never hidden text."""
    detail = response.get('incomplete_details') or {}
    if not isinstance(detail, dict) or detail.get('reason') != 'max_output_tokens':
        return None
    usage = response.get('usage') or {}
    output = usage.get('output_tokens')
    reasoning = (usage.get('output_tokens_details') or {}).get('reasoning_tokens')
    valid = type(output) is int and type(reasoning) is int and 0 <= reasoning <= output and output > 0
    reasoning_only = valid and reasoning == output
    writes = [x for x in stats.get('action_records', []) if x.get('tool') in
              {'write_file', 'edit_file', 'clone_file', 'run_command', 'run_shell'}]
    return {'code': 'reasoning_output_exhausted' if reasoning_only else 'output_limit_exhausted',
            'step': round_record['step'], 'request_output_limit': round_record['max_output_tokens'],
            'limit_source': ('remaining_task_output_budget' if round_record['max_output_tokens'] < stats['budgets']['max_output_tokens']
                             else 'configured_per_round_limit'),
            'reported_output_tokens': output, 'reported_reasoning_tokens': reasoning,
            'non_reasoning_output_tokens': output - reasoning if valid else None,
            'discarded_tool_calls': sum(x.get('type') == 'function_call' for x in response.get('output') or []),
            'incomplete_round_tools_executed': 0,
            'prior_write_or_command_attempts': len(writes),
            'prior_unknown_write_or_command': any(x.get('status') == 'started_outcome_unknown' for x in writes),
            'host_request_pending': bool(stats.get('host_request')),
            'remaining_output_budget': max(0, stats['budgets']['output_budget'] - stats['output_tokens']),
            'automatic_retry': False, 'controller_action': 'review_evidence_then_narrow_task_or_take_over'}


def local_report_lines(stats):
    verification = stats.get('local_verification', {})
    lines = []
    if verification.get('status') == 'passed':
        lines.append(f"本地契约检查：{verification['checked']} 项全部通过；未新增模型调用或重跑命令。科学与任务完整性仍待主控验收。")
    elif verification.get('status') == 'failed':
        lines.append('本地契约检查未通过；检查 acceptance_checks 中的具体证据和缺失项。')
    elif verification.get('status') == 'deferred':
        lines.append('本地契约检查未完成：' + str(verification.get('reason')) + '；不能据此宣称产物合格。')
    diagnostic = stats.get('diagnostic') or {}
    if diagnostic.get('code') == 'reasoning_output_exhausted':
        lines.append(f"单轮输出 {diagnostic['reported_output_tokens']} token 全部计入推理；该截断轮未执行任何工具。")
    elif diagnostic.get('code') == 'output_limit_exhausted':
        lines.append('单轮输出触及本次请求上限；该截断轮的工具全部未执行。')
    if diagnostic:
        if diagnostic['prior_write_or_command_attempts'] == 0 and not diagnostic['host_request_pending']:
            lines.append('本任务记录中尚无写文件或命令调用；这是工具记录范围的结论，不代表外部程序没有修改文件。')
        else:
            lines.append('此前已有写入、命令或宿主交接记录；接手前核对其实际结果，不重跑未知操作。')
        lines.append('主控按 handoff 的 inspection_index 复用读取证据，明确较小实现范围或接手；不原样自动重试，也不自动提高预算或降低 max。')
    return lines


def reconcile_packet(path, job, journal):
    """Validate an explicit controller review; never replay side effects."""
    data, source = project.snapshot(path, 262144)
    value = json.loads(data)
    required = {'schema', 'task_id', 'job_pin', 'receipt_pin', 'note',
                'effects_reviewed', 'no_processes_running', 'evidence'}
    if not isinstance(value, dict) or set(value) != required or value['schema'] != 'DEEPSEEK_RECONCILE_V1':
        raise ValueError('Invalid DEEPSEEK_RECONCILE_V1')
    if value['task_id'] != job['task_id'] or value['effects_reviewed'] is not True or value['no_processes_running'] is not True:
        raise ValueError('Controller must review effects and confirm all commands/host actions stopped')
    if not isinstance(value['note'], str) or not value['note'].strip() or len(value['note']) > 8192:
        raise ValueError('A bounded reconciliation note is required')
    receipt_path = journal.get('receipt_path')
    expected = project.pin(receipt_path) if receipt_path and Path(receipt_path).is_file() else None
    if value['receipt_pin'] != expected: raise ValueError('Reconciliation receipt changed')
    if not isinstance(value['evidence'], list) or len(value['evidence']) > 40:
        raise ValueError('Reconciliation needs an evidence pin list, at most 40')
    for pin in value['evidence']:
        if project.pin(pin['path']) != pin: raise ValueError('Reconciliation evidence changed')
    return {'source': source, 'review': value, 'recorded_at': time.time()}
