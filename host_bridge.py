"""Explicit context/vision input and a resumable HOST-mediated tool handoff.

This is not an implementation of Codex native tool inheritance. Browser and
connector credentials stay in the host. No tools or model requests run here.
"""
import base64
import json
from pathlib import Path
import re
import uuid

import project_support as project
import state_store as state

MAX_PACKET_BYTES = 4 * 1024 * 1024  # Explicit conversation material; separate from image wire limits.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES = 8
MAX_TOTAL_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VISION_REQUEST_BYTES = 24 * 1024 * 1024


def image_parts(images, cwd, contract=None):
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise ValueError('Supply at most 8 explicitly pinned local images')
    parts, identities, total = [], [], 0
    for expected in images:
        if not isinstance(expected, dict) or set(expected) != {'path', 'sha256', 'bytes'}:
            raise ValueError('Images require path, bytes and sha256')
        project.check_access('read_file', {'path': expected['path']}, cwd, contract)
        data, identity = project.snapshot(project.resolve(expected['path'], cwd), MAX_IMAGE_BYTES)
        if identity != expected:
            raise ValueError('Image identity mismatch')
        project.verify_read(identity, contract)
        total += len(data)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError('Image total exceeds 12 MiB')
        if data.startswith(b'\x89PNG\r\n\x1a\n'): mime = 'image/png'
        elif data.startswith(b'\xff\xd8\xff'): mime = 'image/jpeg'
        elif data[:6] in (b'GIF87a', b'GIF89a'): mime = 'image/gif'
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP': mime = 'image/webp'
        else: raise ValueError('Unsupported image bytes; use PNG/JPEG/GIF/WebP')
        identities.append(identity)
        parts.append({'type': 'input_image', 'image_url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')})
    return parts, identities


def load_packet(path, cwd, contract=None):
    if not path:
        return {'messages': [], 'capabilities': [], 'images': [], 'pin': None}
    data, identity = project.snapshot(project.resolve(path, cwd), MAX_PACKET_BYTES)
    value = json.loads(data)
    if not isinstance(value, dict) or value.get('schema') != 'DEEPSEEK_HOST_V1' or set(value) - {'schema', 'conversation', 'images', 'capabilities'}:
        raise ValueError('Invalid DEEPSEEK_HOST_V1 packet')
    conversation = value.get('conversation', [])
    if not isinstance(conversation, list) or len(conversation) > 100:
        raise ValueError('conversation must contain at most 100 explicit messages')
    messages = []
    for message in conversation:
        if not isinstance(message, dict) or set(message) != {'role', 'text'} or message['role'] not in ('user', 'assistant') or not isinstance(message['text'], str):
            raise ValueError('Conversation entries need user/assistant role and text')
        messages.append({'role': message['role'], 'content': message['text']})
    capabilities = value.get('capabilities', [])
    if not isinstance(capabilities, list) or len(capabilities) > 32:
        raise ValueError('At most 32 explicit host capabilities')
    ids = set()
    for cap in capabilities:
        if not isinstance(cap, dict) or set(cap) != {'id', 'description', 'permission'} or not isinstance(cap['id'], str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]{0,63}', cap['id']) or cap['id'] in ids or cap['permission'] not in ('read', 'write') or not isinstance(cap['description'], str) or not 1 <= len(cap['description']) <= 2000:
            raise ValueError('Invalid/duplicate host capability')
        ids.add(cap['id'])
    parts, images = image_parts(value.get('images', []), cwd, contract)
    if parts:
        messages.append({'role': 'user', 'content': [{'type': 'input_text', 'text': 'Pinned images supplied by the controller; inspect pixels, keep scientific acceptance separate.'}] + parts})
    return {'messages': messages, 'capabilities': capabilities, 'images': images, 'pin': identity}


def tool_schema(capabilities):
    return {'type': 'function', 'name': 'request_host',
        'description': '请求主控使用已声明的会话/浏览器/连接器或其他宿主能力；请求后暂停并等待真实结果。不能假定请求已执行；写操作仍受原任务授权限制。',
        'parameters': {'type': 'object', 'additionalProperties': False,
            'properties': {'capability': {'type': 'string', 'enum': [c['id'] for c in capabilities]},
                           'arguments': {'type': 'object'}, 'reason': {'type': 'string'}},
            'required': ['capability', 'arguments', 'reason']}}


def host_request(call, capabilities):
    args = json.loads(call.get('arguments') or '{}')
    if not isinstance(args, dict) or set(args) != {'capability', 'arguments', 'reason'} or not isinstance(args['arguments'], dict) or not isinstance(args['reason'], str):
        raise ValueError('Invalid request_host arguments')
    capability = next((c for c in capabilities if c['id'] == args['capability']), None)
    if capability is None:
        raise ValueError('Undeclared host capability')
    if len(json.dumps(args, ensure_ascii=False)) > 12000:
        raise ValueError('Host request too large')
    return {'request_id': uuid.uuid4().hex, 'call_id': call['call_id'], **args,
            'permission': capability['permission'], 'execution_status': 'not_executed',
            'authorization': 'Host must check scope and its actual tool permissions; never eval arguments as code.'}


def result_packet(path, pending, cwd, contract=None):
    data, source = project.snapshot(project.resolve(path, cwd), MAX_PACKET_BYTES)
    value = json.loads(data)
    if not isinstance(value, dict) or set(value) - {'request_id', 'status', 'text', 'sources', 'images'} or value.get('request_id') != pending['request_id'] or value.get('status') not in ('completed', 'denied', 'error', 'unknown') or not isinstance(value.get('text'), str):
        raise ValueError('Host result must match pending request_id and include status/text')
    if len(value['text']) > 32000 or not isinstance(value.get('sources', []), list):
        raise ValueError('Host result too large or invalid sources')
    for item in value.get('sources', []):
        if not isinstance(item, dict) or not all(isinstance(k, str) and isinstance(v, (str, int)) for k, v in item.items()):
            raise ValueError('Sources must contain simple provenance metadata')
    images, pins = image_parts(value.get('images', []), cwd, contract)
    output = {k: v for k, v in value.items() if k != 'images'}
    output.update(result_source=source, image_sources=pins,
                  provenance='Host-supplied evidence, not independent verification by this MCP')
    items = [{'type': 'function_call_output', 'call_id': pending['call_id'], 'output': json.dumps(output, ensure_ascii=False)}]
    if images:
        items.append({'role': 'user', 'content': [{'type': 'input_text', 'text': 'Images from host result ' + pending['request_id']}] + images})
    return items, source


def request_sizes(payload):
    def without_images(value):
        if isinstance(value, dict):
            return {k: ('[pinned image]' if value.get('type') == 'input_image' and k == 'image_url' else without_images(v)) for k, v in value.items()}
        if isinstance(value, list): return [without_images(x) for x in value]
        return value
    return (len(json.dumps(without_images(payload), ensure_ascii=False).encode()),
            len(json.dumps(payload, ensure_ascii=False).encode()))


def native_handoff(args, contract, settings):
    turns = args.get('native_context_turns', 100)
    if not 1 <= turns <= 100:
        raise ValueError('native_context_turns must be 1..100')
    effort = args.get('effort', 'max')
    if effort not in ('low', 'high', 'max'):
        raise ValueError('This native DeepSeek route supports low/high/max')
    prompt = ('User explicitly authorized this bounded DeepSeek task. Do not spawn further agents.\n'
              'Astra retains scientific/model/architecture decisions and final acceptance.\n'
              'Native tools obey host policy; the MCP path/command enforcement is not a sandbox for native tools.\n'
              'Honor the supplied contract manually; heavy CAD/HPC remain under the existing controller guard.\n'
              f"cwd: {args['cwd']}\nTask: {args['task']}\nContext: {args.get('context', '')}\n"
              f"Project: {settings.get('instructions', '')}\n"
              f"Contract: {json.dumps(contract['value'], ensure_ascii=False) if contract else 'none'}\n"
              f"allow_write={args.get('allow_write', False)} allow_shell={args.get('allow_shell', False)}\n"
              'Use only actions within this task; do not send messages or change external state without existing user authorization. '
              'Return a short report, changed files, test evidence and unresolved items. Do not call deepseek_subagent recursively.')
    return {'status': 'needs_native_agent', 'request_id': uuid.uuid4().hex, 'execution_status': 'not_executed', 'spawn_arguments': {'task_name': 'deepseek_' + state.digest(args)[:10],
        'model': args.get('model', 'deepseek-flash'), 'reasoning_effort': effort, 'fork_turns': str(turns), 'message': prompt},
        'context_inheritance': f'up to last {turns} turns, subject to host support; not guaranteed full-history cloning',
        'tool_inheritance': 'Only tools available and permitted to the native child in this host',
        'billing': 'Host-native billing; not charged through this MCP DeepSeek API key; rates/allowance unverified',
        'instruction': 'Controller may spawn once under explicit task authorization. If unavailable, report limitation; do not silently substitute models or retry.'}
