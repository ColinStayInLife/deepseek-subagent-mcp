"""Bounded project data access and explicit task contracts; no API calls.

Path/command checks constrain the exposed tools, not arbitrary code executed
by an authorized command. This is not an operating-system sandbox.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path
import state_store as state

CONFIG_NAME = '.deepseek-subagent.json'
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_SETTINGS_BYTES = 32000


def read_json(path, limit=MAX_JSON_BYTES):
    return json.loads(snapshot(path, limit)[0])


def snapshot(path, limit=MAX_JSON_BYTES):
    """The returned bytes and SHA describe the SAME open file version."""
    path = Path(path).resolve()
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) > limit:
        raise ValueError(f'File exceeds {limit} bytes; prepare a smaller projection')
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError('File changed while reading; use an immutable evidence snapshot')
    return data, {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
        observed = os.fstat(stream.fileno())
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(observed) or signature(observed) != signature(after):
        raise ValueError('File changed while hashing')
    return {'path': str(path), 'bytes': after.st_size, 'sha256': digest.hexdigest()}


def project_config(cwd):
    for directory in (Path(cwd), *Path(cwd).parents):
        path = directory / CONFIG_NAME
        if path.is_file():
            data, identity = snapshot(path, MAX_SETTINGS_BYTES)
            value = json.loads(data)
            if not isinstance(value, dict) or value.get('schema') != 'DEEPSEEK_PROJECT_V1':
                raise ValueError('Unknown project config schema')
            if set(value) - {'schema', 'name', 'instructions', 'defaults', 'tool_output_chars', 'require_contract_for_shell', 'require_contract_for_write'}:
                raise ValueError('Unknown project config key')
            if not isinstance(value.get('instructions', ''), str) or len(value.get('instructions', '')) > 12000:
                raise ValueError('Invalid project instructions')
            if type(value.get('require_contract_for_shell', False)) is not bool:
                raise ValueError('Invalid project shell policy')
            if type(value.get('require_contract_for_write', False)) is not bool:
                raise ValueError('Invalid project write policy')
            cap = value.get('tool_output_chars', 6000)
            if type(cap) is not int or not 1000 <= cap <= 6000:
                raise ValueError('Invalid project tool output cap')
            return value, identity
    return {}, None


def resolve(path, cwd):
    p = Path(path)
    return (p if p.is_absolute() else Path(cwd) / p).resolve()


def within(path, specifications, cwd):
    for spec in specifications:
        target = resolve(spec, cwd)
        if path == target or (spec.endswith('/') and path.is_relative_to(target)):
            return True
    return False


def load_contract(path, cwd):
    if not path:
        return None
    path = resolve(path, cwd)
    data, identity = snapshot(path, MAX_SETTINGS_BYTES)
    value = json.loads(data)
    expected = {'schema', 'objective', 'acceptance', 'read_paths', 'write_paths', 'commands'}
    optional = {'read_pins', 'dependencies', 'checks'}
    if not isinstance(value, dict) or not expected <= set(value) or set(value) - expected - optional or value['schema'] not in ('DEEPSEEK_TASK_V1', 'DEEPSEEK_TASK_V2'):
        raise ValueError('Invalid task contract schema/keys')
    if value['schema'] == 'DEEPSEEK_TASK_V1' and set(value) != expected:
        raise ValueError('Optional evidence and checks require DEEPSEEK_TASK_V2')
    if not isinstance(value['objective'], str) or not value['objective'].strip():
        raise ValueError('Contract needs an objective')
    for key in ('acceptance', 'read_paths', 'write_paths'):
        if not isinstance(value[key], list) or not all(isinstance(x, str) and x for x in value[key]):
            raise ValueError('Invalid contract ' + key)
    if not value['acceptance'] or not value['read_paths']:
        raise ValueError('Contract needs acceptance criteria and readable paths')
    if not isinstance(value['commands'], list) or len(value['commands']) > 20:
        raise ValueError('Invalid contract command inventory')
    ids = set()
    for cmd in value['commands']:
        required = {'id', 'argv', 'cwd', 'timeout_sec', 'inputs'}
        if not isinstance(cmd, dict) or not required <= set(cmd) or set(cmd) - required - {'mode', 'candidate_paths', 'max_attempts'}:
            raise ValueError('Invalid approved command schema')
        if cmd.get('mode', 'once') not in ('once', 'validation'):
            raise ValueError('Command mode must be once or validation')
        if cmd.get('mode') == 'validation':
            candidates = cmd.get('candidate_paths')
            if not isinstance(candidates, list) or not candidates or not all(isinstance(x, str) and x and within(resolve(x, cwd), value['read_paths'], cwd) for x in candidates):
                raise ValueError('Validation needs readable candidate_paths')
            if type(cmd.get('max_attempts', 3)) is not int or not 1 <= cmd.get('max_attempts', 3) <= 5:
                raise ValueError('Validation max_attempts must be 1..5')
        elif 'candidate_paths' in cmd or 'max_attempts' in cmd:
            raise ValueError('candidate_paths/max_attempts only apply to validation')
        if not isinstance(cmd['id'], str) or not cmd['id'] or cmd['id'] in ids:
            raise ValueError('Duplicate or missing command ID')
        ids.add(cmd['id'])
        if not isinstance(cmd['argv'], list) or not cmd['argv'] or not all(isinstance(x, str) and '\x00' not in x for x in cmd['argv']):
            raise ValueError('Approved command needs exact argv')
        if not Path(cmd['argv'][0]).is_absolute():
            raise ValueError('Approved executable must be absolute')
        if not isinstance(cmd['cwd'], str) or not Path(cmd['cwd']).is_absolute() or not Path(cmd['cwd']).is_dir():
            raise ValueError('Approved cwd must be an existing absolute directory')
        if type(cmd['timeout_sec']) is not int or not 1 <= cmd['timeout_sec'] <= 600:
            raise ValueError('Invalid approved command timeout')
        if not isinstance(cmd['inputs'], list) or not cmd['inputs']:
            raise ValueError('Approved command needs pinned source/input files')
        for source in cmd['inputs']:
            if not isinstance(source, dict) or set(source) != {'path', 'sha256', 'bytes'} or pin(source['path']) != source:
                raise ValueError('Approved command input pin mismatch')
    for key in ('read_pins', 'dependencies', 'checks'):
        if not isinstance(value.get(key, []), list):
            raise ValueError('Contract ' + key + ' must be a list')
    for source in value.get('read_pins', []):
        if not isinstance(source, dict) or set(source) != {'path', 'sha256', 'bytes'} or not within(resolve(source['path'], cwd), value['read_paths'], cwd) or pin(source['path']) != source:
            raise ValueError('Read evidence pin mismatch or outside read_paths')
    for dependency in value.get('dependencies', []):
        if not isinstance(dependency, str) or not within(resolve(dependency, cwd), value['read_paths'], cwd) or not resolve(dependency, cwd).is_file():
            raise ValueError('Missing/unreadable contract dependency: ' + str(dependency))
    check_ids = set()
    for check in value.get('checks', []):
        if not isinstance(check, dict) or set(check) not in ({'id', 'path', 'pointer', 'expected'}, {'id', 'path', 'pointer', 'expected_file_sha256'}) or not all(isinstance(check[k], str) and check[k] for k in ('id', 'path')) or not isinstance(check['pointer'], str) or not within(resolve(check['path'], cwd), value['read_paths'], cwd):
            raise ValueError('Invalid acceptance check')
        if 'expected_file_sha256' in check and (not isinstance(check['expected_file_sha256'], str) or not within(resolve(check['expected_file_sha256'], cwd), value['read_paths'], cwd)):
            raise ValueError('Expected hash source must be inside read_paths')
        if check['id'] in check_ids: raise ValueError('Duplicate acceptance check ID')
        check_ids.add(check['id'])
    # Resolve allowed roots once. A later symlink replacement must not move them.
    scopes = {k: [(str(resolve(x, cwd)), x.endswith('/')) for x in value[k]] for k in ('read_paths', 'write_paths')}
    return {'value': value, 'pin': identity, 'cwd': cwd, 'scopes': scopes}


def check_access(name, args, cwd, contract):
    if not contract:
        return
    if pin(contract['pin']['path']) != contract['pin']:
        raise ValueError('Task contract changed during execution')
    value = contract['value']
    if name == 'run_shell':
        raise ValueError('Contract mode exposes only approved command IDs, not arbitrary shell')
    if name == 'run_command':
        return
    key = 'write_paths' if name in ('write_file', 'edit_file') else 'read_paths'
    path = resolve(args.get('path') or '.', cwd)
    def allowed(key):
        return any(path == Path(root) or (directory and path.is_relative_to(root)) for root, directory in contract['scopes'][key])
    if not allowed(key):
        raise ValueError(f'{name} path outside explicit {key}: {path}')
    if name == 'edit_file' and not allowed('read_paths'):
        raise ValueError('edit_file also requires read access')
    for source in value.get('read_pins', []):
        if path == Path(source['path']) and pin(path) != source:
            raise ValueError('Pinned read evidence changed')


def verify_read(identity, contract):
    for source in (contract or {}).get('value', {}).get('read_pins', []):
        if identity['path'] == source['path'] and identity != source:
            raise ValueError('Read snapshot does not match pinned evidence')


def acceptance_results(contract, cwd):
    results = []
    for check in (contract or {}).get('value', {}).get('checks', []):
        try:
            check_access('json_query', {'path': check['path']}, cwd, contract)
            data, source = snapshot(resolve(check['path'], cwd))
            verify_read(source, contract)
            actual = json_pointer(json.loads(data), check['pointer'])
            expected_source = None
            if 'expected_file_sha256' in check:
                check_access('file_info', {'path': check['expected_file_sha256']}, cwd, contract)
                expected_source = pin(resolve(check['expected_file_sha256'], cwd))
                expected = expected_source['sha256']
            else:
                expected = check['expected']
            passed = type(actual) is type(expected) and actual == expected
            results.append({'id': check['id'], 'passed': passed, 'source': source, 'pointer': check['pointer'], 'expected_source': expected_source})
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            results.append({'id': check['id'], 'passed': False, 'error': str(exc)[:300]})
    return results


def json_pointer(value, pointer):
    if pointer == '':
        return value
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('Use JSON Pointer, e.g. /nodes/f0131_002_000/state')
    for part in pointer[1:].split('/'):
        if '~' in part and __import__('re').search(r'~(?![01])', part):
            raise ValueError('Invalid JSON Pointer escape')
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            if not part.isdigit() or (len(part) > 1 and part[0] == '0'):
                raise ValueError('Invalid JSON array index')
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise ValueError('JSON Pointer traverses a scalar')
    return value


def preview(value, depth, limit, offset=0):
    if isinstance(value, dict):
        keys = list(value)
        if depth == 0:
            return {'type': 'object', 'count': len(keys), 'keys': keys[offset:offset + limit], 'next_offset': offset + limit if offset + limit < len(keys) else None}
        selected = keys[offset:offset + limit]
        return {'type': 'object', 'count': len(keys), 'items': {k: preview(value[k], depth - 1, limit) for k in selected}, 'next_offset': offset + limit if offset + limit < len(keys) else None}
    if isinstance(value, list):
        return {'type': 'array', 'count': len(value), 'offset': offset,
                'items': [preview(x, depth - 1, limit) if depth else {'type': type(x).__name__} for x in value[offset:offset + limit]],
                'next_offset': offset + limit if offset + limit < len(value) else None}
    if isinstance(value, str) and len(value) > 1500:
        return {'type': 'string', 'chars': len(value), 'preview': value[:1500], 'truncated': True}
    return value


def tool_json_query(args, cwd, max_chars=6000, contract=None):
    path = resolve(args['path'], cwd)
    pointers = args.get('pointers', [''])
    if not isinstance(pointers, list) or not 1 <= len(pointers) <= 12 or not all(isinstance(x, str) for x in pointers):
        raise ValueError('Supply 1 to 12 JSON Pointers')
    depth, limit, offset = args.get('depth', 1), args.get('limit', 8), args.get('offset', 0)
    if any(type(x) is not int for x in (depth, limit, offset)) or not (0 <= depth <= 3 and 1 <= limit <= 30 and 0 <= offset):
        raise ValueError('Invalid depth/limit/offset')
    data, source = snapshot(path)
    verify_read(source, contract)
    value = json.loads(data)
    result = {'path': str(path), 'source': source, 'selection': {}, 'next_pointer_index': None}
    encode = lambda: json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    for index, pointer in enumerate(pointers):
        try:
            target = json_pointer(value, pointer)
        except (KeyError, IndexError):
            target = {'error': 'pointer not found'}
        chosen_depth, chosen_limit = depth, limit
        result['next_pointer_index'] = index + 1 if index + 1 < len(pointers) else None
        while True:
            if isinstance(target, str) and (offset or len(target) > 1500):
                candidate = {'type': 'string', 'chars': len(target), 'offset': offset,
                             'preview': target[offset:offset+1500], 'next_offset': offset+1500 if offset+1500 < len(target) else None}
            elif target == {'error': 'pointer not found'}:
                candidate = target
            else:
                candidate = preview(target, chosen_depth, chosen_limit, offset)
            if (chosen_depth, chosen_limit) != (depth, limit) and isinstance(candidate, dict):
                candidate = dict(candidate, effective_depth=chosen_depth, effective_limit=chosen_limit)
            result['selection'][pointer] = candidate
            if len(encode()) <= max_chars:
                break
            if chosen_limit > 1:
                chosen_limit = max(1, chosen_limit // 2)
            elif chosen_depth > 0:
                chosen_depth -= 1
            elif isinstance(target, str):
                # Explicit byte-independent character pagination, retaining valid JSON.
                low, high = 0, min(1500, max(0, len(target)-offset))
                while low < high:
                    mid = (low + high + 1) // 2
                    result['selection'][pointer] = {'type': 'string', 'chars': len(target), 'offset': offset,
                        'preview': target[offset:offset+mid], 'next_offset': offset+mid if offset+mid < len(target) else None}
                    if len(encode()) <= max_chars: low = mid
                    else: high = mid - 1
                result['selection'][pointer] = {'type': 'string', 'chars': len(target), 'offset': offset,
                    'preview': target[offset:offset+low], 'next_offset': offset+low if offset+low < len(target) else None}
                if low and len(encode()) <= max_chars: break
                del result['selection'][pointer]
                result['next_pointer_index'] = index
                return encode() if len(encode()) <= max_chars else json.dumps({'error': 'Source metadata exceeds tool budget; use a shorter evidence path'})
            else:
                del result['selection'][pointer]
                result['next_pointer_index'] = index
                if not result['selection']:
                    result['error'] = 'Selection cannot fit; query a child pointer or shorter evidence path'
                return encode() if len(encode()) <= max_chars else json.dumps({'error': 'Source metadata exceeds tool budget; use a shorter evidence path'})
    return encode()


def tool_file_info(args, cwd):
    path = resolve(args['path'], cwd)
    result = pin(path)
    result['mtime_ns'] = path.stat().st_mtime_ns
    return json.dumps(result, ensure_ascii=False)


def data_tool_schemas():
    return [
        {'type': 'function', 'name': 'json_query', 'description': '按JSON Pointer提取同一源SHA的字段。next_pointer_index非null时用剩余pointers续查；next_offset用于页内续查（字符串为字符偏移）。effective_depth/limit表示预算缩减，不能当作已读全部内容。重要字段用精确pointer。',
         'parameters': {'type': 'object', 'additionalProperties': False, 'properties': {'path': {'type': 'string'}, 'pointers': {'type': 'array', 'minItems': 1, 'maxItems': 12, 'items': {'type': 'string'}, 'default': ['']},
             'depth': {'type': 'integer', 'minimum': 0, 'maximum': 3, 'default': 1}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 30, 'default': 8}, 'offset': {'type': 'integer', 'minimum': 0, 'default': 0}}, 'required': ['path']}},
        {'type': 'function', 'name': 'file_info', 'description': '读取文件大小、SHA256和修改时间；不输出文件正文。',
         'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}
    ]


def command_schema(contract):
    return {'type': 'function', 'name': 'run_command', 'description': '运行固定命令。once不可重试；validation仅在候选文件SHA变化时限次重验。同版已有结果复用；未知结果禁止重发。',
            'parameters': {'type': 'object', 'properties': {'id': {'type': 'string', 'enum': [x['id'] for x in contract['value']['commands']]}}, 'required': ['id']}}


def run_command(args, contract, executed, action_dir, ledger_dir=None):
    name = args.get('id')
    commands = {x['id']: x for x in contract['value']['commands']}
    if name not in commands:
        raise ValueError('Unknown or already attempted approved command; no automatic retry')
    cmd = commands[name]
    for source in cmd['inputs']:
        if pin(source['path']) != source:
            raise ValueError('Approved command input changed')
    candidates = [pin(resolve(x, contract['cwd'])) for x in cmd.get('candidate_paths', [])]
    version = state.digest(candidates) if cmd.get('mode') == 'validation' else 'once'
    attempt_key = name + ':' + version
    if attempt_key in executed:
        raise ValueError('Already attempted this command/input version; inspect saved result')
    executed.add(attempt_key)
    key = state.digest({'contract': contract['pin'], 'command_id': name})
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else Path(action_dir).parent / '.command_ledger'
    with state.lock(ledger_dir / (key + '.lock')):
        ledger_path = ledger_dir / (key + '.json')
        ledger = state.read(ledger_path, {'attempts': {}})
        previous = ledger['attempts'].get(version)
        if previous:
            if previous['status'] == 'returned':
                for source in previous['logs']:
                    if pin(source['path']) != source:
                        raise ValueError('Saved command evidence changed')
                return previous['result']
            raise ValueError('Prior command outcome unknown; inspect logs/receipt before reconciliation')
        if any(x['status'] != 'returned' for x in ledger['attempts'].values()):
            raise ValueError('Earlier command outcome unknown; no new candidate run')
        if len(ledger['attempts']) >= cmd.get('max_attempts', 3 if cmd.get('mode') == 'validation' else 1):
            raise ValueError('Approved validation attempt budget exhausted')
        ledger['attempts'][version] = {'status': 'started_outcome_unknown', 'action_dir': str(action_dir), 'candidates': candidates}
        state.atomic_json(ledger_path, ledger)
        result, logs = _execute_command(cmd, name, action_dir, candidates)
        ledger['attempts'][version].update(status='returned', result=result, logs=logs)
        state.atomic_json(ledger_path, ledger)
        return result


def _execute_command(cmd, name, action_dir, candidates):
    action_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    out_path, err_path = action_dir / 'stdout.log', action_dir / 'stderr.log'
    with out_path.open('xb') as stdout, err_path.open('xb') as stderr:
        os.chmod(out_path, 0o600); os.chmod(err_path, 0o600)
        proc = subprocess.Popen(cmd['argv'], cwd=cmd['cwd'], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            proc.wait(timeout=cmd['timeout_sec'])
        except subprocess.TimeoutExpired:
            raise RuntimeError('Approved command timeout; inspect saved logs before any further run')
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    result = {'id': name, 'exit_code': proc.returncode, 'stdout': pin(out_path), 'stderr': pin(err_path), 'candidates': candidates}
    if any(pin(source['path']) != source for source in candidates + cmd['inputs']):
        raise ValueError('Candidate/input changed during validation; logs saved, result not accepted')
    for label, path in [('stdout_tail', out_path), ('stderr_tail', err_path)]:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 1200))
            result[label] = f.read(1200).decode('utf-8', 'replace')
    prefix = '错误：approved command failed\n' if proc.returncode else ''
    return prefix + json.dumps(result, ensure_ascii=False), [result['stdout'], result['stderr']]
