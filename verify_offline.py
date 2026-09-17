#!/usr/bin/env python3
"""Run mock-only regressions and a fresh stdio handshake, saving evidence.

No task is submitted to a real model. Refuses to overwrite an earlier run.
"""
import argparse
import ast
import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import execution_support as execution
import project_support as project
import server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Optional previous server.py for comparing default limits')
    args = parser.parse_args()
    if args.baseline: args.baseline = args.baseline.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    os.chdir(root)
    modules = ['test_server', 'test_project_support', 'test_upgrade', 'test_async_jobs',
               'test_execution_support', 'test_workflow_support', 'test_budget_completion', 'test_budget_settings',
               'test_contract_efficiency']
    stream = io.StringIO()
    environment = {'DEEPSEEK_API_BASE': 'http://127.0.0.1:9', 'DEEPSEEK_API_KEY': 'offline-test-only'}
    # Direct accidental API use fails before credentials/network. Subprocess
    # fixtures mock responses and inherit a dummy key plus a loopback endpoint.
    with patch.dict(os.environ, environment), \
         patch.object(server, 'load_api_key', side_effect=AssertionError('Offline verification cannot load credentials')), \
         patch.object(server._OPENER, 'open', side_effect=AssertionError('Offline verification cannot use the network')):
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (output/'tests.log').write_text(stream.getvalue())

    messages = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05'}},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {
            'name': 'deepseek_subagent', 'arguments': {'task': 'invalid offline probe', 'max_steps': 0}}},
    ]
    proc = subprocess.run([sys.executable, str(root/'server.py')],
        input='\n'.join(json.dumps(m) for m in messages)+'\n', capture_output=True, text=True,
        timeout=10, env={**os.environ, **environment, 'DEEPSEEK_SUBAGENT_RUNS_DIR': str(output/'probe-runs')})
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    protocol_ok = (proc.returncode == 0 and [r.get('id') for r in replies] == [1, 2, 3]
                   and replies[0]['result']['serverInfo']['version'] == server.SERVER_VERSION
                   and {'steer', 'followup', 'reconcile'} <= set(replies[1]['result']['tools'][0]['inputSchema']['properties']['action']['enum'])
                   and replies[2]['result']['isError'] is True
                   and not (output/'probe-runs/usage.jsonl').exists())
    (output/'stdio.json').write_text(json.dumps(replies, ensure_ascii=False, indent=2)+'\n')

    fixture = output/'clone_fixture_source.py'
    fixture.write_text('SCHEMA = "V1"\n' + ''.join(f'# retained fixture line {i:05d}\n' for i in range(4000)))
    target = output/'clone_fixture_result.py'
    source = project.pin(fixture)
    clone_args = dict(source_path=str(fixture), path=str(target), source_sha256=source['sha256'],
                      edits=[dict(old_text='"V1"', new_text='"V2"')])
    cloned = json.loads(execution.clone_file(clone_args, str(output)))
    expected = fixture.read_bytes().replace(b'"V1"', b'"V2"', 1)
    clone_ok = target.read_bytes() == expected and project.pin(fixture) == source
    clone_size = len(json.dumps(clone_args, ensure_ascii=False).encode())
    full_size = len(json.dumps(dict(path=str(target), content=expected.decode()), ensure_ascii=False).encode())

    constants = {'DEFAULT_EFFORT', 'DEFAULT_MAX_STEPS', 'HARD_MAX_STEPS', 'DEFAULT_TIMEOUT',
                 'HARD_TIMEOUT', 'DEFAULT_MAX_OUTPUT_TOKENS', 'DEFAULT_OUTPUT_BUDGET',
                 'DEFAULT_INPUT_BUDGET', 'MAX_REQUEST_BYTES', 'MAX_TOOL_CALLS'}
    def selected_constants(path):
        return {n.targets[0].id: ast.dump(n.value) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in constants}
    limits_unchanged = (selected_constants(root/'server.py') == selected_constants(args.baseline)
                        if args.baseline else None)
    report = {'schema': 'DEEPSEEK_MCP_OFFLINE_VALIDATION_V1', 'server_version': server.SERVER_VERSION,
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'PASS' if result.wasSuccessful() and protocol_ok and clone_ok and limits_unchanged is not False else 'FAIL',
        'tests': {'run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                  'skipped': len(result.skipped), 'log': project.pin(output/'tests.log')},
        'fresh_stdio': {'passed': protocol_ok, 'evidence': project.pin(output/'stdio.json')},
        'clone_fixture': {'passed': clone_ok, 'source': source, 'output': cloned['output'],
                          'clone_argument_bytes': clone_size, 'full_write_argument_bytes': full_size,
                          'interpretation': 'Synthetic request-size comparison; not a paid model or task-success benchmark'},
        'defaults_and_limits_unchanged': limits_unchanged,
        'effective_defaults': server.budget_defaults(),
        'configurable_limits': {k: server.TOOL_DEF['inputSchema']['properties'][k]['maximum']
                                for k in server.budget_defaults()},
        'request_limits': {'text_bytes': server.MAX_REQUEST_BYTES,
                           'context_window_tokens': server.MODEL_CONTEXT_TOKENS,
                           'context_projection_is_estimate': True},
        'paid_model_calls': 0, 'real_model_effectiveness_measured': False,
        'live_connections_restarted': False,
        'sources': [project.pin(root/name) for name in ['server.py', 'execution_support.py', 'project_support.py',
                    'workflow_support.py', 'host_bridge.py', 'state_store.py', 'client.py',
                    'prepare.py', 'async_jobs.py', 'verify_offline.py', 'scheduler.json'] + [m+'.py' for m in modules]]}
    (output/'VALIDATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'server_version', 'tests', 'paid_model_calls')}, ensure_ascii=False))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    sys.exit(main())
