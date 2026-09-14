"""Local execution helpers. No model calls, retries, or permission expansion."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import project_support as project

MUTATIONS = frozenset(('write_file', 'edit_file', 'clone_file'))
READS = frozenset(('read_file', 'json_query', 'file_info'))


def clone_file(args, cwd, contract=None):
    """Build a new version from pinned UTF-8 bytes, with exact replacements.

    All edits validate before publication. link() publishes a complete file
    without overwriting an existing destination, including a dangling symlink.
    """
    for key in ('source_path', 'path'):
        if not isinstance(args.get(key), str) or not args[key]:
            raise ValueError(key + ' must be a nonempty path')
    digest = args.get('source_sha256')
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('source_sha256 must be the SHA from read_file/file_info')
    edits = args.get('edits')
    if not isinstance(edits, list) or len(edits) > 32:
        raise ValueError('edits must contain 0..32 exact replacements')
    project.check_access('clone_file', args, cwd, contract)
    source = project.resolve(args['source_path'], cwd)
    # Keep the final component unresolved so O_EXCL semantics include symlinks.
    requested = Path(cwd) / args['path']
    target = requested.parent.resolve() / requested.name
    if os.path.lexists(target):
        raise ValueError('Destination already exists; inspect it, then use edit_file if authorized')
    data, identity = project.snapshot(source)
    project.verify_read(identity, contract)
    if identity['sha256'] != digest:
        raise ValueError('Source SHA changed; destination was not created')
    text = data.decode('utf-8')
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'old_text', 'new_text'}:
            raise ValueError('Each edit needs exactly old_text and new_text')
        old, new = edit['old_text'], edit['new_text']
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError('old_text must be nonempty; new_text must be a string')
        count = text.count(old)
        if count != 1:
            raise ValueError(f'Edit matched {count} times; destination was not created')
        text = text.replace(old, new, 1)
    output = text.encode('utf-8')
    if len(output) > project.MAX_JSON_BYTES:
        raise ValueError('Clone exceeds the local file size cap')
    if project.pin(source) != identity:
        raise ValueError('Source changed while preparing clone')
    target.parent.mkdir(parents=True, exist_ok=True)
    project.check_access('clone_file', args, cwd, contract)
    if project.resolve(args['path'], cwd) != target or project.resolve(args['source_path'], cwd) != source:
        raise ValueError('Path changed while preparing clone')
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix='.deepseek-clone-', dir=target.parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(output)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)  # atomic complete publication; never replace
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return json.dumps({'status': 'created', 'source': identity,
                       'output': project.pin(target), 'replacements': len(edits)}, ensure_ascii=False)


def deliverable_baseline(contract, cwd):
    baseline = {}
    for item in (contract or {}).get('value', {}).get('deliverables', []):
        project.check_access('write_file', item, cwd, contract)
        path = project.resolve(item['path'], cwd)
        baseline[item['path']] = project.pin(path) if path.exists() else None
    return baseline


def deliverable_results(contract, cwd, baseline):
    results = []
    for item in (contract or {}).get('value', {}).get('deliverables', []):
        result = {'kind': 'deliverable', 'path': item['path'], 'passed': False}
        try:
            project.check_access('write_file', item, cwd, contract)
            if item['path'] not in baseline:
                raise ValueError('Missing initial baseline; cannot assert delivery after resume')
            source = project.pin(project.resolve(item['path'], cwd))
            before = baseline[item['path']]
            changed = before is None or source['sha256'] != before['sha256']
            result.update(source=source, changed=changed,
                          passed=source['bytes'] > 0 and (changed or not item['must_change']))
        except (OSError, ValueError, TypeError) as exc:
            result['error'] = str(exc)[:300]
        results.append(result)
    return results


def progress(records):
    mutations = [a for a in records if a['tool'] in MUTATIONS]
    return {'write_attempts': len(mutations),
            'writes_returned': sum(a['status'] == 'returned' for a in mutations),
            'writes_unknown': sum(a['status'] == 'started_outcome_unknown' for a in mutations),
            'command_attempts': sum(a['tool'] in ('run_shell', 'run_command') for a in records),
            'read_calls': sum(a['tool'] in READS or a['tool'] in ('grep', 'list_dir') for a in records),
            'tool_errors': sum(a['status'] == 'tool_error' for a in records)}


def next_input_estimate(request_bytes, previous_bytes, previous_input, has_images=False):
    """One-request estimate, not a tokenizer or a hard billing cap.

    Calibrate with the last observed input usage, including cached tokens.
    Images retain at least the previous usage estimate; text-only byte ratios
    cannot reliably predict image tokens.
    """
    if not previous_bytes or not previous_input:
        return None
    ratio = request_bytes / previous_bytes
    if has_images:
        ratio = max(1.0, ratio)
    return math.ceil(previous_input * ratio * 1.15 + 256)


def context_projection(request_bytes, calibrated_input, output_tokens, window, margin, has_images=False):
    """Reserve generation room separately from cumulative usage/wire limits.

    Without provider usage, one UTF-8 byte per input token is a conservative
    text heuristic, not the provider tokenizer. Later requests use the existing
    usage-calibrated estimate. Image tokens cannot be inferred from wire bytes;
    record that uncertainty rather than claiming an exact 1M context check.
    """
    estimated = calibrated_input if calibrated_input is not None else request_bytes
    return {'estimated_input_tokens': estimated, 'reserved_output_tokens': output_tokens,
            'margin_tokens': margin, 'window_tokens': window,
            'projected_tokens': estimated + output_tokens + margin,
            'method': 'usage_calibrated' if calibrated_input is not None else 'utf8_bytes_heuristic',
            'is_estimate': True, 'uncalibrated_images': bool(has_images and calibrated_input is None)}


def duplicate_read_note(name, raw, result, prior_records):
    """Compact only freshly revalidated, byte-identical successful reads.

    The full earlier result remains in the API history and on disk. Never
    cache filesystem contents or reuse a result instead of checking access.
    """
    if name not in READS or result.startswith('错误：'):
        return None
    args_sha = hashlib.sha256(raw.encode()).hexdigest()
    result_sha = hashlib.sha256(result.encode()).hexdigest()
    for action in reversed(prior_records):
        if (action['tool'] == name and action['status'] == 'returned'
                and action['arguments_sha256'] == args_sha and action['result_sha256'] == result_sha):
            note = (f"UNCHANGED_READ: 本次已重新检查权限和文件；内容与 action_{action['sequence']} "
                    "完全相同。使用会话中已有的完整结果；下一步请缩小查询、实现或验证。")
            return note if len(note) < len(result) else None
    return None
