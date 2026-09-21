#!/usr/bin/env python3
"""Launch the installed stdio MCP for one explicit JSON task; never retry.

This also verifies new server code without disrupting an existing client's
stdio connection. Argument files are local and must not contain credentials.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import state_store as state
import platform_support as platform
from server import HARD_TIMEOUT

CLIENT_TIMEOUT = HARD_TIMEOUT + 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arguments', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        parser.error('Result file already exists; inspect the prior attempt instead of overwriting it')
    arguments = json.loads(args.arguments.read_text(encoding='utf-8'))
    if not isinstance(arguments, dict): parser.error('Arguments must be a JSON object')
    if arguments.get('action', 'run') in ('run', 'submit', 'native_request'):
        arguments.setdefault('task_id', 'client-' + state.digest(arguments)[:32])
    # Reserve BEFORE launching. An interrupted client leaves an inspectable
    # claim, so the same result path cannot accidentally start a second job.
    args.result.parent.mkdir(parents=True, exist_ok=True)
    claim = {'status': 'client_started_outcome_unknown', 'task_id': arguments.get('task_id'),
             'batch_id': arguments.get('batch_id'),
             'arguments_sha256': state.digest(arguments),
             'instruction': 'Inspect server task status/receipt; do not blindly rerun.'}
    with args.result.open('x', encoding='utf-8') as stream:
        platform.private_file(args.result)
        json.dump(claim, stream, ensure_ascii=False)
        stream.flush(); os.fsync(stream.fileno())
    messages = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05'}},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'deepseek_subagent', 'arguments': arguments}},
    ]
    proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('server.py'))],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', **platform.process_options())
    try:
        stdout, stderr = proc.communicate('\n'.join(json.dumps(x, ensure_ascii=False) for x in messages) + '\n', timeout=CLIENT_TIMEOUT)
    except BaseException:
        if platform.WINDOWS:
            platform.kill_process_group(proc)
            proc.communicate()
            raise
        # The server owns its command groups. SIGTERM first interrupts Python
        # so its finally blocks clean them; SIGKILL is only the final fallback.
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.communicate()
        raise
    responses = [json.loads(line) for line in stdout.splitlines()]
    if proc.returncode or [x.get('id') for x in responses] != [1, 2, 3]:
        raise RuntimeError('MCP process/protocol failed; inspect runs/ receipts before a new call. ' + stderr[-1500:])
    result = responses[-1]
    state.atomic_json(args.result, {'server': responses[0]['result']['serverInfo'], 'response': result,
                                  'task_id': arguments.get('task_id'), 'batch_id': arguments.get('batch_id')})
    if 'error' in result:
        print(json.dumps(result['error'], ensure_ascii=False)); return 1
    for content in result['result'].get('content', []):
        if content.get('type') == 'text': print(content['text'])
    print('MCP result: ' + str(args.result.resolve()))
    return int(result['result'].get('isError', False))


if __name__ == '__main__':
    platform.configure_stdio()
    sys.exit(main())
