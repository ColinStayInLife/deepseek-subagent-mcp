#!/usr/bin/env python3
"""Prepare pinned task/context/result packets offline; never call a model.

`contract` consumes a concise draft with input_paths/pinned_read_paths and
generates a validated DEEPSEEK_TASK_V2 contract. Existing outputs are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import host_bridge as bridge
import project_support as project


def canonical(path, cwd):
    result = str(project.resolve(path, cwd))
    return result + '/' if path.endswith('/') and result != '/' else result


def directory_rules(draft, key, cwd):
    values = draft.get(key, [])
    if not isinstance(values, list) or not all(isinstance(x, str) and x for x in values):
        raise ValueError(key + ' must be a list of directory paths')
    return [str(project.resolve(x, cwd)).rstrip('/') + '/' for x in values]


def build_contract(draft, cwd):
    allowed = {'objective', 'acceptance', 'read_paths', 'write_paths', 'read_dirs', 'write_dirs', 'commands', 'pinned_read_paths', 'dependencies', 'checks', 'deliverables'}
    if not isinstance(draft, dict) or set(draft) - allowed:
        raise ValueError('Unknown draft key')
    value = {'schema':'DEEPSEEK_TASK_V2', 'objective':draft['objective'], 'acceptance':draft['acceptance'],
             'read_paths':[canonical(x,cwd) for x in draft.get('read_paths',[])] + directory_rules(draft, 'read_dirs', cwd),
             'write_paths':[canonical(x,cwd) for x in draft.get('write_paths',[])] + directory_rules(draft, 'write_dirs', cwd),
             'commands':[], 'read_pins':[project.pin(project.resolve(x,cwd)) for x in draft.get('pinned_read_paths',[])],
             'dependencies':[canonical(x,cwd) for x in draft.get('dependencies',[])],
             'checks':[{**x,'path':canonical(x['path'],cwd)} for x in draft.get('checks',[])]}
    if 'deliverables' in draft:
        value['deliverables'] = [{**x, 'path': canonical(x['path'], cwd)} for x in draft['deliverables']]
    for entry in draft.get('commands',[]):
        command = dict(entry)
        command['cwd'] = str(project.resolve(command.get('cwd',cwd),cwd))
        command['inputs'] = [project.pin(project.resolve(x,cwd)) for x in command.pop('input_paths')]
        if 'candidate_paths' in command:
            command['candidate_paths'] = [canonical(x,cwd) for x in command['candidate_paths']]
        value['commands'].append(command)
    for check in value['checks']:
        if 'expected_file_sha256' in check:
            check['expected_file_sha256'] = canonical(check['expected_file_sha256'],cwd)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'contract.json'
        path.write_text(json.dumps(value))
        project.load_contract(path,cwd)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind',required=True)
    contract = sub.add_parser('contract')
    contract.add_argument('--draft',type=Path,required=True)
    check = sub.add_parser('check-contract', help='Validate an existing contract offline; never run a model or command')
    check.add_argument('--contract', type=Path, required=True)
    check.add_argument('--cwd', type=Path, default=Path.cwd())
    host = sub.add_parser('host')
    host.add_argument('--conversation',type=Path)
    host.add_argument('--capabilities',type=Path)
    host.add_argument('--image',action='append',default=[])
    result = sub.add_parser('host-result')
    result.add_argument('--request-id',required=True)
    result.add_argument('--status',choices=['completed','denied','error','unknown'],required=True)
    result.add_argument('--text-file',type=Path,required=True)
    result.add_argument('--sources',type=Path)
    result.add_argument('--image',action='append',default=[])
    native = sub.add_parser('native-result')
    native.add_argument('--agent-id',required=True)
    native.add_argument('--status',choices=['reported','completed','failed','unknown'],default='reported')
    native.add_argument('--text-file',type=Path,required=True)
    native.add_argument('--evidence',action='append',default=[])
    for child in (contract,host,result,native):
        child.add_argument('--cwd',type=Path,default=Path.cwd())
        child.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    cwd = str(args.cwd.resolve())
    if args.kind == 'check-contract':
        try:
            loaded = project.load_contract(args.contract, cwd)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(json.dumps({'status': 'invalid', 'error': str(exc), 'model_calls': 0,
                              'commands_run': 0}, ensure_ascii=False))
            return 2
        scopes = {key: [{'path': path, 'recursive': recursive, 'exists': Path(path).exists()}
                        for path, recursive in loaded['scopes'][key]] for key in ('read_paths', 'write_paths')}
        print(json.dumps({'status': 'valid', 'contract': loaded['pin'], 'scopes': scopes,
                          'missing_read_paths': [x['path'] for x in scopes['read_paths'] if not x['exists']],
                          'model_calls': 0, 'commands_run': 0,
                          'note': 'Scope/pin validation only; no task acceptance or command execution.'}, ensure_ascii=False))
        return 0
    if args.output.exists(): parser.error('Output already exists; keep the existing evidence version')
    if args.kind == 'contract':
        value = build_contract(project.read_json(args.draft),cwd)
    elif args.kind == 'host':
        value = {'schema':'DEEPSEEK_HOST_V1',
                 'conversation':project.read_json(args.conversation) if args.conversation else [],
                 'capabilities':project.read_json(args.capabilities) if args.capabilities else [],
                 'images':[project.pin(project.resolve(x,cwd)) for x in args.image]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'host.json'; path.write_text(json.dumps(value))
            bridge.load_packet(path,cwd)
    elif args.kind == 'host-result':
        value = {'request_id':args.request_id,'status':args.status,'text':args.text_file.read_text(),
                 'sources':project.read_json(args.sources) if args.sources else [],
                 'images':[project.pin(project.resolve(x,cwd)) for x in args.image]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'; path.write_text(json.dumps(value))
            bridge.result_packet(path,{'request_id':args.request_id,'call_id':'validate-packet'},cwd)
    else:
        value = {'agent_id':args.agent_id,'execution_status':args.status,'text':args.text_file.read_text(),
                 'evidence':[project.pin(project.resolve(x,cwd)) for x in args.evidence]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        os.fchmod(stream.fileno(),0o600)
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps(project.pin(args.output),ensure_ascii=False))


if __name__ == '__main__': raise SystemExit(main())
