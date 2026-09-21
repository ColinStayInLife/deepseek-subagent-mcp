"""Native OS regressions; no API credentials or model calls."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import platform_support as platform
import project_support as project
import state_store as state


def directory_link(link, target):
    """Junctions exercise Windows reparse paths without admin/developer mode."""
    if platform.WINDOWS:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def unlink_directory(link):
    if platform.WINDOWS: link.rmdir()
    else: link.unlink()


def assert_private(test, path):
    if not platform.WINDOWS:
        test.assertEqual(Path(path).stat().st_mode & 0o777, 0o600)
        return
    from ctypes import wintypes as w
    get = platform._api(platform.advapi, 'GetFileSecurityW', w.BOOL,
                        [w.LPCWSTR, w.DWORD, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD)])
    convert = platform._api(platform.advapi, 'ConvertSecurityDescriptorToStringSecurityDescriptorW', w.BOOL,
                            [w.LPVOID, w.DWORD, w.DWORD, ctypes.POINTER(w.LPWSTR), w.LPVOID])
    needed = w.DWORD()
    get(str(path), 4, None, 0, ctypes.byref(needed))
    descriptor = ctypes.create_string_buffer(needed.value)
    platform._checked(get(str(path), 4, descriptor, needed.value, ctypes.byref(needed)))
    text = w.LPWSTR()
    platform._checked(convert(descriptor, 1, 4, ctypes.byref(text), None))
    try:
        test.assertTrue(text.value.startswith('D:P'), text.value)
        test.assertEqual(set(text.value[3:].split(')')[:-1]), {'(A;;FA;;;OW', '(A;;FA;;;SY'})
    finally:
        free = platform._api(platform.kernel, 'LocalFree', w.HLOCAL, [w.HLOCAL])
        free(ctypes.cast(text, w.HLOCAL))


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='deepseek 中文 space ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_utf8_atomic_state_and_private_permissions(self):
        path = self.root/'任务.json'
        state.atomic_json(path, {'内容': '中文 😀'})
        self.assertEqual(state.read(path), {'内容': '中文 😀'})
        assert_private(self, path)
        state.atomic_json(path, {'内容': '更新'})
        self.assertEqual(state.read(path), {'内容': '更新'})
        assert_private(self, path)

    def test_process_lock_excludes_other_process_and_releases(self):
        path = self.root/'claim.lock'
        script = "import state_store,sys\nwith state_store.lock(sys.argv[1]): print('claimed')"
        def attempt():
            return subprocess.run([sys.executable, '-X', 'utf8', '-c', script, str(path)],
                                  cwd=Path(__file__).parent, capture_output=True, encoding='utf-8', timeout=5)
        with state.lock(path):
            result = attempt()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('already running', result.stderr)
        self.assertEqual(attempt().stdout.strip(), 'claimed')

    def test_identity_disappears_after_exit_and_stale_cancel_is_ignored(self):
        proc = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], **platform.process_options())
        try:
            identity = platform.process_identity(proc.pid)
            self.assertIsNotNone(identity)
            self.assertEqual(identity, platform.process_identity(proc.pid))
            platform.signal_worker({**identity, 'start_ticks': '0'})
            self.assertIsNone(proc.poll())
        finally:
            proc.kill(); proc.wait()
        self.assertIsNone(platform.process_identity(proc.pid))

    def test_deadline_cleans_descendants_after_parent_already_exited(self):
        marker = self.root/'late'
        child = 'import time,pathlib;time.sleep(1);pathlib.Path(' + repr(str(marker)) + ').touch()'
        parent = 'import subprocess,sys;subprocess.Popen([sys.executable,"-c",' + repr(child) + '])'
        proc = platform.popen_group([sys.executable, '-c', parent], cwd=self.root,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: platform.wait_process(proc, 5)
        finally: platform.kill_process_group(proc)
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_blocked_response_timeout_does_not_publish_late_result(self):
        start = time.monotonic()
        with self.assertRaises(platform.DeadlineExceeded):
            with platform.wall_deadline(.05):
                platform.network_call(lambda: time.sleep(.5))
        self.assertLess(time.monotonic() - start, .3)
        with platform.wall_deadline(.1):
            self.assertEqual(platform.network_call(lambda: 'fresh'), 'fresh')

    def test_blocked_command_output_can_be_interrupted(self):
        proc = platform.popen_group([sys.executable, '-c', 'import time;time.sleep(30)'],
                                    stdout=subprocess.PIPE, text=True, encoding='utf-8')
        try:
            with self.assertRaises(platform.DeadlineExceeded):
                with platform.wall_deadline(.05):
                    list(platform.iter_lines(proc.stdout))
        finally:
            platform.kill_process_group(proc)
            proc.stdout.close()

    def test_directory_link_scope_cannot_escape(self):
        outside = self.root/'outside'; outside.mkdir()
        allowed = self.root/'allowed'; allowed.mkdir()
        link = allowed/'link'
        directory_link(link, outside)
        self.assertFalse(project.within((link/'file').resolve(), [str(allowed) + os.sep], str(self.root)))

    def test_directory_separator_scope(self):
        child = self.root/'folder'/'child.txt'
        self.assertTrue(project.within(child, [str(self.root/'folder') + os.sep], str(self.root)))
        self.assertFalse(project.within(self.root/'sibling'/'child.txt', [str(self.root/'folder') + os.sep], str(self.root)))

    def test_native_directory_separator_survives_contract_validation(self):
        folder = self.root/'folder'; folder.mkdir()
        child = folder/'child.txt'; child.write_text('evidence', encoding='utf-8')
        path = self.root/'contract.json'
        value = {'schema': 'DEEPSEEK_TASK_V2', 'objective': 'bounded read',
                 'acceptance': ['inspect evidence'], 'read_paths': [str(folder) + os.sep],
                 'write_paths': [], 'commands': []}
        state.atomic_json(path, value)
        contract = project.load_contract(path, str(self.root))
        project.check_access('read_file', {'path': str(child)}, str(self.root), contract)
        for invalid in [str(folder), str(child) + os.sep]:
            state.atomic_json(path, {**value, 'read_paths': [invalid]})
            with self.assertRaises(ValueError):
                project.load_contract(path, str(self.root))


if __name__ == '__main__': unittest.main()
