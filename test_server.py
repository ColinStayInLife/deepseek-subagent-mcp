"""Offline regression checks. Run: python3 -m unittest -v test_server.py"""
import copy
import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import server as s


def response(text=None, calls=(), status="completed", tokens=20):
    output = []
    if text is not None:
        output.append({"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": text}]})
    for i, (name, args) in enumerate(calls):
        output.append({"type": "function_call", "call_id": "call_" + str(i),
                       "name": name, "arguments": json.dumps(args)})
    return {"status": status, "output": output, "usage": {
        "input_tokens": 100, "output_tokens": tokens,
        "input_tokens_details": {"cached_tokens": 50},
        "output_tokens_details": {"reasoning_tokens": 5}}}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = self.tmp.name
        self.kw = dict(task="inspect file", context="", cwd=self.cwd,
                       model="deepseek-flash", effort="low", max_steps=5,
                       allow_write=False, allow_shell=False, timeout_sec=5)

    def tearDown(self):
        self.tmp.cleanup()

    def run_responses(self, responses, **kw):
        captured = []
        def fake(payload, timeout):
            captured.append(copy.deepcopy(payload))
            return responses.pop(0)
        with patch.object(s, "call_responses", side_effect=fake):
            result = s.run_subagent(**(self.kw | kw))
        return result, captured

    def test_multi_round_preserves_assistant_and_pairing(self):
        p = Path(self.cwd, "x.txt"); p.write_text("evidence")
        (report, stats), requests = self.run_responses([
            response("checking", [("read_file", {"path": "x.txt"})]), response("done")])
        self.assertEqual(report, "done")
        self.assertEqual(stats["status"], "completed")
        self.assertEqual(stats["input_tokens"], 200)
        items = requests[1]["input"]
        self.assertTrue(any(x.get("role") == "assistant" for x in items))
        tool_outputs = [x for x in items if x.get('type') == 'function_call_output']
        self.assertEqual(tool_outputs[-1]["call_id"], "call_0")
        self.assertIn("evidence", tool_outputs[-1]["output"])

    def test_incomplete_does_not_execute_write(self):
        (report, stats), _ = self.run_responses([response("partial", [
            ("write_file", {"path": "oops", "content": "bad"})], status="incomplete")], allow_write=True)
        self.assertEqual(stats["status"], "api_incomplete")
        self.assertFalse(Path(self.cwd, "oops").exists())

    def test_empty_response_cannot_reuse_progress(self):
        (_, stats), _ = self.run_responses([
            response("progress", [("list_dir", {})]), response()])
        self.assertEqual(stats["status"], "empty_response")

    def test_cumulative_output_cap(self):
        (_, stats), requests = self.run_responses([
            response(calls=[("list_dir", {})], tokens=450), response("done", tokens=130)],
            output_budget=600, max_output_tokens=512)
        self.assertEqual([r["max_output_tokens"] for r in requests], [512, 150])
        self.assertEqual(stats["output_tokens"], 580)

    def test_budget_exhaustion_stops_new_api_request(self):
        (_, stats), requests = self.run_responses([
            response(calls=[("list_dir", {})], tokens=500)], output_budget=512)
        self.assertEqual(stats["status"], "output_budget")
        self.assertEqual(len(requests), 1)

    def test_context_rejected_before_network(self):
        (_, stats), requests = self.run_responses([], context="字" * s.MAX_REQUEST_BYTES)
        self.assertEqual(stats["status"], "context_budget")
        self.assertEqual(requests, [])

    def test_input_soft_budget(self):
        (_, stats), requests = self.run_responses([
            response(calls=[("list_dir", {})])], input_budget=100)
        self.assertEqual(stats["status"], "input_budget")
        self.assertEqual(len(requests), 1)

    def test_last_round_only_reports_and_is_not_success(self):
        (_, stats), requests = self.run_responses([
            response(calls=[("list_dir", {})]), response("partial results")], max_steps=2)
        self.assertEqual(requests[-1]["tool_choice"], "none")
        self.assertEqual(stats["status"], "step_limit")

    def test_repeated_calls_stop(self):
        (_, stats), _ = self.run_responses([
            response(calls=[("list_dir", {})]) for _ in range(3)])
        self.assertEqual(stats["status"], "repeated_call")
        self.assertEqual(stats["tool_calls"], 2)

    def test_single_round_text_task_can_complete(self):
        (_, stats), requests = self.run_responses([response("done")], max_steps=1)
        self.assertEqual(stats["status"], "completed")
        self.assertEqual(requests[0]["tool_choice"], "none")

    def test_deadline_interrupts_blocked_api(self):
        def slow(*args, **kwargs):
            time.sleep(5)
        with patch.object(s, "call_responses", side_effect=slow), s.wall_deadline(0.05):
            _, stats = s.run_subagent(**self.kw)
        self.assertEqual(stats["status"], "timeout")
        self.assertFalse(stats["usage_complete"])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_readonly_and_exact_edit(self):
        names = [x["name"] for x in s.tool_schemas(False, False)]
        self.assertNotIn("run_shell", names)
        self.assertNotIn("write_file", names)
        p = Path(self.cwd, "code.py"); p.write_text("foo foo")
        args = json.dumps({"path": "code.py", "old_text": "foo", "new_text": "bar"})
        self.assertTrue(s.dispatch("edit_file", args, self.cwd, False, False).startswith("错误："))
        self.assertTrue(s.dispatch("edit_file", args, self.cwd, True, False).startswith("错误："))
        self.assertEqual(p.read_text(), "foo foo")
        args = json.dumps({"path": "code.py", "old_text": "foo foo", "new_text": "bar"})
        self.assertTrue(s.dispatch("edit_file", args, self.cwd, True, False).startswith("已修改"))
        self.assertEqual(p.read_text(), "bar")

    def test_tool_context_clipped_before_replay(self):
        Path(self.cwd, "x.txt").write_text("x" * 40000)
        (_, stats), requests = self.run_responses([
            response(calls=[("read_file", {"path": "x.txt"})]), response("done")])
        outputs = [x['output'] for x in requests[1]['input'] if x.get('type') == 'function_call_output']
        self.assertLessEqual(len(outputs[-1]), s.MAX_TOOL_OUTPUT)

    def test_validation_no_paid_call(self):
        with patch.object(s, "call_responses") as api:
            for args in [{"task": "x", "allow_write": "false"},
                         {"task": "x", "cwd": self.cwd + "/missing"},
                         {"task": "x", "effort": "invalid"},
                         {"task": "x", "max_steps": 0}]:
                result = s.handle_tools_call({"name": "deepseek_subagent", "arguments": args})
                self.assertTrue(result["isError"])
            api.assert_not_called()

    def test_error_flag_artifact_usage(self):
        with patch.object(s, "call_responses", return_value=response("partial", status="incomplete")), \
             patch.object(s, "RUNS_DIR", Path(self.cwd, "runs")):
            result = s.handle_tools_call({"name": "deepseek_subagent", "arguments": {"task": "x"}})
        self.assertTrue(result["isError"])
        records = Path(self.cwd, "runs", "usage.jsonl").read_text().splitlines()
        record = json.loads(records[0])
        self.assertEqual(record["status"], "api_incomplete")
        self.assertNotIn("actions", record)
        self.assertTrue(Path(record["report_path"]).exists())

    def test_shell_timeout_kills_descendants(self):
        marker = Path(self.cwd, "late")
        result = s.tool_run_shell({"command": "(sleep 2; touch late) & wait", "timeout_sec": 1}, self.cwd)
        self.assertIn("超时", result)
        time.sleep(1.2)
        self.assertFalse(marker.exists())

    def test_stdio_handshake_and_invalid_call(self):
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "deepseek_subagent", "arguments": {"task": "x", "max_steps": 0}}}]
        proc = subprocess.run([sys.executable, s.__file__], input="\n".join(map(json.dumps, messages)) + "\n",
                              text=True, capture_output=True, timeout=5)
        result = list(map(json.loads, proc.stdout.splitlines()))
        self.assertEqual([x["id"] for x in result], [1, 2, 3])
        self.assertEqual(result[1]["result"]["tools"][0]["name"], "deepseek_subagent")
        self.assertTrue(result[2]["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
