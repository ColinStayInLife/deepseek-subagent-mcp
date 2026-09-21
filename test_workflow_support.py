"""Offline acceptance of steering, continuation, pruning and durable review."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import project_support as p
import server as s
import platform_support as platform
from test_platform_support import assert_private, directory_link, unlink_directory
import state_store as state
import workflow_support as w
from test_server import response


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(self.temp.name).resolve()
        self.runs = self.root / 'runs'
        patch.object(s, 'RUNS_DIR', self.runs).start()
        self.api = patch.object(s, 'call_responses', side_effect=AssertionError('No live model')).start()
        patch.object(s, 'load_api_key', side_effect=AssertionError('No credentials')).start()

    def call(self, **args):
        return s.handle_tools_call({'name': 'deepseek_subagent', 'arguments': args})

    def start(self, **updates):
        return self.call(task_id='task', task='bounded implementation', cwd=str(self.root),
                         **{'max_steps': 8, **updates})

    def feedback(self, ident='f1', **updates):
        return self.call(action='followup', task_id='task', followup_id=ident,
                         **{'instruction': 'Check the acceptance issue', **updates})

    def test_steering_skips_calls_planned_before_new_instruction(self):
        mailbox = w.Mailbox(self.runs, 'task')
        captured = []
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload))
            if len(captured) == 1:
                mailbox.send('m1', 'Do not create the file; report the missing condition')
                return response(calls=[('write_file', {'path': 'wrong.txt', 'content': 'wrong'})])
            return response('Missing condition identified')
        self.api.side_effect = fake
        result = self.start(allow_write=True)
        self.assertFalse(result['isError'], result)
        self.assertFalse((self.root / 'wrong.txt').exists())
        self.assertTrue(w.paired(captured[-1]['input']))
        self.assertIn('SKIPPED', json.dumps(captured[-1]))
        receipt = state.read(result['structuredContent']['receipt_path'])
        self.assertEqual(receipt['stale_calls_skipped'], 1)
        self.assertEqual(receipt['tool_calls'], 0)
        self.assertEqual(len(captured), 2)

    def test_steering_between_tools_keeps_prior_write_skips_remaining(self):
        mailbox = w.Mailbox(self.runs, 'task')
        original = s.dispatch
        def dispatch(*args, **kwargs):
            result = original(*args, **kwargs)
            mailbox.send('m1', 'Stop further writes and report the first output')
            return result
        self.api.side_effect = [response(calls=[('write_file', {'path': 'first.txt', 'content': 'first'}),
                                                ('write_file', {'path': 'second.txt', 'content': 'second'})]), response('first only')]
        with patch.object(s, 'dispatch', side_effect=dispatch):
            result = self.start(allow_write=True)
        self.assertFalse(result['isError'], result)
        self.assertEqual((self.root/'first.txt').read_text(), 'first')
        self.assertFalse((self.root/'second.txt').exists())

    def test_message_at_final_answer_forces_replan_before_completed(self):
        mailbox = w.Mailbox(self.runs, 'task')
        count = 0
        def fake(payload, timeout):
            nonlocal count
            count += 1
            if count == 1: mailbox.send('m1', 'Check the revised acceptance detail')
            return response('revised' if count == 2 else 'stale')
        self.api.side_effect = fake
        result = self.start()
        self.assertEqual(count, 2)
        self.assertIn('revised', result['content'][0]['text'])
        self.assertFalse(mailbox.summary()['accepting'])
        with self.assertRaises(ValueError): mailbox.send('late', 'new work')
        self.assertTrue(mailbox.send('m1', 'Check the revised acceptance detail')['reused'])

    def test_mailbox_duplicate_conflict_size_and_no_permission_escalation(self):
        box = w.Mailbox(self.runs, 'task')
        box.send('m1', 'Enable writing now')
        self.assertTrue(box.send('m1', 'Enable writing now')['reused'])
        with self.assertRaises(ValueError): box.send('m1', 'different')
        with self.assertRaises(ValueError): box.send('m2', '中'*3000)
        self.api.side_effect = [response(calls=[('write_file', {'path': 'denied', 'content': 'x'})]), response('write denied')]
        result = self.start()
        self.assertFalse((self.root/'denied').exists())
        self.assertEqual(state.read(result['structuredContent']['receipt_path'])['tool_errors'], 1)

    def test_followup_reuses_context_remaining_budgets_and_deduplicates(self):
        captured = []
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload))
            return response('first report' if len(captured) == 1 else 'fixed report')
        self.api.side_effect = fake
        first = self.start(output_budget=1000)
        checkpoint = first['structuredContent']['followup_checkpoint']
        assert_private(self, checkpoint['path'])
        follow = self.feedback()
        self.assertFalse(follow['isError'], follow)
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[1]['max_output_tokens'], 980)
        self.assertIn('first report', json.dumps(captured[1]['input']))
        self.assertTrue(all(x['reasoning']['effort'] == 'max' for x in captured))
        receipt = state.read(follow['structuredContent']['receipt_path'])
        self.assertEqual(receipt['input_tokens'], 200)
        self.assertEqual(receipt['output_tokens'], 40)
        self.assertEqual(receipt['steps'], 2)
        self.assertEqual(self.feedback(), follow)
        self.assertTrue(self.feedback(instruction='different')['isError'])
        self.assertEqual(len(captured), 2)
        usage = [json.loads(x) for x in (self.runs/'usage.jsonl').read_text().splitlines()]
        self.assertEqual([x['input_tokens'] for x in usage], [100, 100])

    def test_followup_cannot_change_budget_model_permissions_or_contract(self):
        self.api.side_effect = [response('ready')]
        first = self.start()
        for key, value in [('output_budget', 2000), ('allow_write', True), ('model', 'deepseek-v4-pro')]:
            self.assertTrue(self.feedback(**{key: value})['isError'])
        self.assertEqual(self.api.call_count, 1)
        Path(first['structuredContent']['followup_checkpoint']['path']).write_text('{}')
        self.assertTrue(self.feedback()['isError'])
        self.assertEqual(self.api.call_count, 1)

    def test_followup_freezes_model_even_when_process_defaults_change(self):
        captured = []
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload))
            return response('done')
        self.api.side_effect = fake
        self.start()
        with patch.object(s, 'DEFAULT_MODEL', 'deepseek-v4-pro'), patch.object(s, 'DEFAULT_EFFORT', 'low'):
            follow = self.feedback()
        self.assertFalse(follow['isError'], follow)
        self.assertTrue(all(x['model'] == 'deepseek-flash' and x['reasoning']['effort'] == 'max' for x in captured))

    def test_followup_rejects_exhausted_budget_unknown_and_incomplete_pairs(self):
        self.api.side_effect = [response('ready')]
        self.start(max_steps=1)
        self.assertTrue(self.feedback()['isError'])
        self.assertEqual(self.api.call_count, 1)
        self.assertFalse(w.can_checkpoint([{'type': 'function_call', 'call_id': 'x'}],
                                         {'status': 'step_limit', 'usage_complete': True}))
        self.assertFalse(w.can_checkpoint([], {'status': 'completed', 'usage_complete': False}))
        self.assertFalse(w.can_checkpoint([], {'status': 'error', 'usage_complete': True}))

    def test_followup_rejects_changed_project_and_symlink_scope(self):
        allowed = self.root/'allowed'; allowed.mkdir()
        outside = self.root/'outside'; outside.mkdir()
        link = self.root/'link'; directory_link(link, allowed)
        contract = self.root/'contract.json'
        contract.write_text(json.dumps({'schema': 'DEEPSEEK_TASK_V2', 'objective': 'bounded',
            'acceptance': ['review'], 'read_paths': ['link/'], 'write_paths': [], 'commands': []}))
        self.api.side_effect = [response('ready')]
        self.start(contract_path=str(contract))
        unlink_directory(link); directory_link(link, outside)
        self.assertTrue(self.feedback()['isError'])
        self.assertEqual(self.api.call_count, 1)

    def test_handoff_exposes_evidence_not_worker_transcript(self):
        resp = response('report')
        resp['output'].insert(0, {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'private worker scratch'}]})
        self.api.side_effect = [resp]
        result = self.start()
        handoff = state.read(result['structuredContent']['handoff_path'])
        self.assertEqual(handoff['schema'], 'DEEPSEEK_HANDOFF_V1')
        self.assertTrue(handoff['controller_acceptance_required'])
        self.assertNotIn('private worker scratch', json.dumps(handoff))
        self.assertNotIn('private worker scratch', json.dumps(self.call(action='status', task_id='task')))

    def test_failed_acceptance_can_be_repaired_without_repeating_first_write(self):
        contract = self.root/'contract.json'
        contract.write_text(json.dumps({'schema': 'DEEPSEEK_TASK_V2', 'objective': 'two outputs',
            'acceptance': ['both outputs'], 'read_paths': [str(self.root)+'/'],
            'write_paths': ['first.txt', 'second.txt'], 'commands': [],
            'deliverables': [{'path': 'first.txt', 'must_change': True}, {'path': 'second.txt', 'must_change': True}]}))
        self.api.side_effect = [response(calls=[('write_file', {'path': 'first.txt', 'content': 'first'})]), response('one done'),
                               response(calls=[('write_file', {'path': 'second.txt', 'content': 'second'})]), response('both done')]
        first = self.start(allow_write=True, contract_path=str(contract))
        self.assertEqual(first['structuredContent']['status'], 'acceptance_failed')
        follow = self.feedback(instruction='Create the missing second output and verify')
        self.assertEqual(follow['structuredContent']['status'], 'completed', follow)
        self.assertEqual(state.read(follow['structuredContent']['receipt_path'])['progress']['writes_returned'], 2)

    def test_crashed_followup_cannot_be_replayed_under_same_or_new_id(self):
        self.api.side_effect = [response('ready'), SystemExit('simulated worker crash')]
        self.start()
        with self.assertRaises(SystemExit): self.feedback()
        self.assertTrue(self.feedback()['isError'])
        self.assertTrue(self.feedback('new-id')['isError'])
        self.assertEqual(self.api.call_count, 2)

    def test_followup_host_handoff_keeps_cumulative_usage_and_claim_rules(self):
        packet = self.root/'host.json'
        packet.write_text(json.dumps({'schema': 'DEEPSEEK_HOST_V1', 'capabilities': [
            {'id': 'read', 'description': 'read external evidence', 'permission': 'read'}]}))
        self.api.side_effect = [response('ready'), response(calls=[('request_host', {
            'capability': 'read', 'arguments': {'query': 'evidence'}, 'reason': 'verify correction'})]), response('verified')]
        self.start(host_context_path=str(packet))
        follow = self.feedback()
        self.assertEqual(follow['structuredContent']['status'], 'needs_host', follow)
        request_id = follow['structuredContent']['host_request']['request_id']
        host_result = self.root/'host-result.json'
        host_result.write_text(json.dumps({'request_id': request_id, 'status': 'completed', 'text': 'verified source'}))
        self.assertTrue(self.call(action='resume', task_id='task', host_result_path=str(host_result))['isError'])
        self.assertFalse(self.call(action='claim', task_id='task', request_id=request_id)['isError'])
        done = self.call(action='resume', task_id='task', host_result_path=str(host_result))
        self.assertFalse(done['isError'], done)
        self.assertEqual(state.read(done['structuredContent']['receipt_path'])['input_tokens'], 300)

    def test_pruning_preserves_contract_images_recent_outputs_and_pairing(self):
        items = [{'role': 'user', 'content': 'boundary = 453 K; do not invent parameters'},
                 {'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'unchanged'}]}]
        records = []
        for i in range(8):
            text = f'head {i}\n' + 'x'*5000 + f'\ntail {i}'
            evidence = self.root/f'result{i}.txt'; evidence.write_text(text)
            items.append({'type': 'function_call', 'call_id': str(i), 'name': 'read_file'})
            records.append({'history_index': len(items), 'status': 'returned', 'result_file': p.pin(evidence)})
            items.append({'type': 'function_call_output', 'call_id': str(i), 'output': text})
        before = copy.deepcopy(items)
        pruned, metrics = w.prune_outputs(items, records)
        self.assertEqual(metrics['results'], 4)
        self.assertGreater(metrics['bytes_removed'], 12000)
        self.assertEqual(pruned[:2], before[:2])
        self.assertEqual(pruned[-8:], before[-8:])
        self.assertEqual(items, before)
        self.assertTrue(w.paired(pruned))
        self.assertIn('full_result', pruned[3]['output'])
        self.api.assert_not_called()
        (self.root/'result0.txt').write_text('changed')
        _, metrics = w.prune_outputs(items, records)
        self.assertEqual(metrics['results'], 3)

    def test_context_pressure_prunes_without_extra_model_summary(self):
        captured = []
        for i in range(7): (self.root/f'f{i}.txt').write_text('detail '*700)
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload))
            i = len(captured)-1
            return response(calls=[('read_file', {'path': f'f{i}.txt'})]) if i < 7 else response('done')
        self.api.side_effect = fake
        with patch.object(s, 'MAX_REQUEST_BYTES', 35000):
            result = self.start(max_steps=10)
        self.assertFalse(result['isError'], result)
        receipt = state.read(result['structuredContent']['receipt_path'])
        self.assertTrue(receipt.get('context_pruning'), receipt)
        self.assertEqual(len(captured), 8)
        archive = receipt['context_pruning'][0]['archive']
        self.assertEqual(p.pin(archive['path']), archive)
        assert_private(self, archive['path'])
        self.assertIn('full_result', json.dumps(captured[-1]['input']))


if __name__ == '__main__': unittest.main()
