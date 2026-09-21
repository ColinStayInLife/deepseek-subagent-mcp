"""Standard-library Linux/Windows process, deadline and private-file support.

Windows uses byte-range locks (in state_store), creation-time process identity,
named cancellation events, and kill-on-close Job Objects for command trees.
Only the network request runs in an interruptible daemon thread; tools always
run on the controller thread and an abandoned response can never execute tools.
"""
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

WINDOWS = os.name == 'nt'


class DeadlineExceeded(RuntimeError):
    pass


_local = threading.local()


def check_interrupt():
    for scope in getattr(_local, 'deadlines', []):
        if not scope['fired'] and time.monotonic() >= scope['at']:
            # Permit the caller's exception handler to persist the receipt.
            for item in _local.deadlines:
                if item['at'] <= scope['at']: item['fired'] = True
            raise DeadlineExceeded('任务墙钟预算已耗尽')
    cancel = getattr(_local, 'cancel', None)
    if cancel and not getattr(_local, 'cancelled', False) and cancel():
        _local.cancelled = True
        raise DeadlineExceeded('Async cancellation requested')


@contextmanager
def wall_deadline(seconds):
    if not WINDOWS:
        def expired(signum, frame):
            raise DeadlineExceeded('任务墙钟预算已耗尽')
        prior = signal.getitimer(signal.ITIMER_REAL)
        entered = time.monotonic()
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(seconds, prior[0]) if prior[0] else seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if prior[0]:
                signal.setitimer(signal.ITIMER_REAL, max(.000001, prior[0] - (time.monotonic() - entered)), prior[1])
        return
    scopes = getattr(_local, 'deadlines', [])
    _local.deadlines = scopes
    scope = {'at': time.monotonic() + max(0, seconds), 'fired': False}
    scopes.append(scope)
    try:
        check_interrupt()
        yield
        check_interrupt()
    finally:
        scopes.remove(scope)


def sleep(seconds):
    end = time.monotonic() + seconds
    while True:
        check_interrupt()
        remaining = end - time.monotonic()
        if remaining <= 0: return
        time.sleep(min(.01, remaining))


def network_call(function, *args, **kwargs):
    if not WINDOWS: return function(*args, **kwargs)
    check_interrupt()
    result = queue.Queue(maxsize=1)
    def request():
        try:
            result.put((True, function(*args, **kwargs)))
        except BaseException as exc:
            result.put((False, exc))
    threading.Thread(target=request, daemon=True, name='deepseek-http').start()
    while True:
        check_interrupt()
        try:
            ok, value = result.get(timeout=.01)
        except queue.Empty:
            continue
        check_interrupt()
        if ok: return value
        raise value


def wait_process(proc, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        check_interrupt()
        if proc.poll() is not None: return proc.returncode
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        try:
            return proc.wait(timeout=min(.02, remaining) if remaining is not None else .02)
        except subprocess.TimeoutExpired:
            pass


def iter_lines(stream):
    """Drain a pipe without blocking Windows deadline/cancellation checks."""
    if not WINDOWS:
        yield from stream
        return
    lines, stopped = queue.Queue(maxsize=32), threading.Event()
    def publish(item):
        while not stopped.is_set():
            try:
                lines.put(item, timeout=.02)
                return
            except queue.Full:
                pass
    def reader():
        try:
            for line in stream:
                if stopped.is_set(): return
                publish((True, line))
            publish((True, None))
        except Exception as exc:
            publish((False, exc))
    threading.Thread(target=reader, daemon=True, name='deepseek-pipe').start()
    try:
        while True:
            check_interrupt()
            try: ok, line = lines.get(timeout=.01)
            except queue.Empty: continue
            if not ok: raise line
            if line is None: return
            yield line
    finally:
        stopped.set()


def process_options():
    return {'creationflags': subprocess.CREATE_NO_WINDOW} if WINDOWS else {'start_new_session': True}


def is_reparse_link(path):
    try:
        info = Path(path).lstat()
        return bool(getattr(info, 'st_reparse_tag', 0)) or Path(path).is_symlink()
    except OSError:
        return True


def configure_stdio():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace', newline='\n')


def shell_command(command):
    if not WINDOWS: return ['bash', '-lc', command]
    executable = shutil.which('pwsh') or shutil.which('powershell')
    if not executable: raise RuntimeError('PowerShell is required for run_shell on Windows')
    prefix = "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); $OutputEncoding = [Console]::OutputEncoding; "
    return [executable, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', prefix + command]


if WINDOWS:
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    advapi = ctypes.WinDLL('advapi32', use_last_error=True)

    def _api(dll, name, result, args):
        fn = getattr(dll, name)
        fn.restype, fn.argtypes = result, args
        return fn

    _close = _api(kernel, 'CloseHandle', w.BOOL, [w.HANDLE])
    _open = _api(kernel, 'OpenProcess', w.HANDLE, [w.DWORD, w.BOOL, w.DWORD])
    _times = _api(kernel, 'GetProcessTimes', w.BOOL, [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4)
    _wait = _api(kernel, 'WaitForSingleObject', w.DWORD, [w.HANDLE, w.DWORD])
    _create_event = _api(kernel, 'CreateEventW', w.HANDLE, [w.LPVOID, w.BOOL, w.BOOL, w.LPCWSTR])
    _open_event = _api(kernel, 'OpenEventW', w.HANDLE, [w.DWORD, w.BOOL, w.LPCWSTR])
    _set_event = _api(kernel, 'SetEvent', w.BOOL, [w.HANDLE])
    _create_job = _api(kernel, 'CreateJobObjectW', w.HANDLE, [w.LPVOID, w.LPCWSTR])
    _set_job = _api(kernel, 'SetInformationJobObject', w.BOOL, [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD])
    _assign_job = _api(kernel, 'AssignProcessToJobObject', w.BOOL, [w.HANDLE, w.HANDLE])
    _end_job = _api(kernel, 'TerminateJobObject', w.BOOL, [w.HANDLE, w.UINT])

    class _BasicLimits(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', w.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                    ('active', w.DWORD), ('affinity', ctypes.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]

    class _JobLimits(ctypes.Structure):
        _fields_ = [('basic', _BasicLimits), ('io', ctypes.c_uint64 * 6),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]

    def _checked(value):
        if not value: raise ctypes.WinError(ctypes.get_last_error())
        return value


def process_identity(pid):
    if not isinstance(pid, int) or pid <= 0: return None
    if not WINDOWS:
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return None if fields[0] == 'Z' else {'pid': pid, 'start_ticks': fields[19]}
        except (OSError, IndexError):
            return None
    handle = _open(0x1000 | 0x100000, False, pid)  # query limited + synchronize
    if not handle: return None
    try:
        if _wait(handle, 0) != 258: return None
        stamps = [w.FILETIME() for _ in range(4)]
        _checked(_times(handle, *(ctypes.byref(x) for x in stamps)))
        return {'pid': pid, 'start_ticks': str((stamps[0].dwHighDateTime << 32) | stamps[0].dwLowDateTime)}
    finally:
        _close(handle)


def _event_name(identity):
    return 'Local\\DeepSeekSubagent-' + str(identity['pid']) + '-' + str(identity['start_ticks'])


@contextmanager
def cancellation_scope():
    if not WINDOWS:
        previous = signal.getsignal(signal.SIGTERM)
        def stop(signum, frame):
            raise DeadlineExceeded('Async cancellation requested')
        signal.signal(signal.SIGTERM, stop)
        try: yield
        finally: signal.signal(signal.SIGTERM, previous)
        return
    handle = _checked(_create_event(None, True, False, _event_name(process_identity(os.getpid()))))
    previous = getattr(_local, 'cancel', None), getattr(_local, 'cancelled', False)
    _local.cancel, _local.cancelled = lambda: _wait(handle, 0) == 0, False
    try: yield
    finally:
        _local.cancel, _local.cancelled = previous
        _close(handle)


def signal_worker(identity):
    if process_identity(identity['pid']) != identity: return
    if WINDOWS:
        handle = _open_event(0x0002, False, _event_name(identity))  # EVENT_MODIFY_STATE
        if not handle: raise ctypes.WinError(ctypes.get_last_error())
        try: _checked(_set_event(handle))
        finally: _close(handle)
        return
    if hasattr(os, 'pidfd_open'):
        fd = os.pidfd_open(identity['pid'])
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        open_fd = libc.pidfd_open
        open_fd.argtypes, open_fd.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
        fd = open_fd(identity['pid'], 0)
        if fd < 0: raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    try:
        if process_identity(identity['pid']) == identity:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally: os.close(fd)


def popen_group(argv, **kwargs):
    """Create a command tree; Windows gates execution until job assignment.

    Never use this for persistent async workers: their lifetime intentionally
    extends beyond the submitting stdio connection. Commands have no stdin.
    """
    if not WINDOWS:
        return subprocess.Popen(argv, **kwargs, **process_options())
    if kwargs.pop('stdin', subprocess.DEVNULL) != subprocess.DEVNULL:
        raise ValueError('Managed commands require stdin=DEVNULL')
    job = _checked(_create_job(None, None))
    proc = None
    try:
        limits = _JobLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        _checked(_set_job(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        proc = subprocess.Popen([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--command'],
                                stdin=subprocess.PIPE, **kwargs, **process_options())
        _checked(_assign_job(job, int(proc._handle)))
        proc._deepseek_job = job
        payload = json.dumps(argv) + '\n'
        proc.stdin.write(payload if proc.text_mode else payload.encode('utf-8'))
        proc.stdin.close()
        proc.stdin = None
        return proc
    except BaseException:
        if proc is not None:
            proc.kill(); proc.wait()
            if proc.stdin: proc.stdin.close()
        _close(job)
        raise


def kill_process_group(proc):
    if WINDOWS:
        job = getattr(proc, '_deepseek_job', None)
        if job:
            try:
                _checked(_end_job(job, 1))
                # The job becomes signalled when every descendant has exited.
                if _wait(job, 5000) == 258:
                    raise RuntimeError('Command tree did not exit after termination; inspect outcome')
            finally:
                _close(job)
                proc._deepseek_job = None
        elif proc.poll() is None:
            proc.kill()
    else:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    proc.wait()


def private_file(path):
    """Protect private state before writing; Windows DACL is owner + SYSTEM."""
    if not WINDOWS:
        os.chmod(path, 0o600)
        return
    convert = _api(advapi, 'ConvertStringSecurityDescriptorToSecurityDescriptorW', w.BOOL,
                   [w.LPCWSTR, w.DWORD, ctypes.POINTER(w.LPVOID), w.LPVOID])
    set_security = _api(advapi, 'SetFileSecurityW', w.BOOL, [w.LPCWSTR, w.DWORD, w.LPVOID])
    free = _api(kernel, 'LocalFree', w.HLOCAL, [w.HLOCAL])
    descriptor = w.LPVOID()
    _checked(convert('D:P(A;;FA;;;OW)(A;;FA;;;SY)', 1, ctypes.byref(descriptor), None))
    try: _checked(set_security(str(Path(path).resolve()), 0x80000004, descriptor))
    finally: free(descriptor)


def fsync_directory(path):
    if WINDOWS: return  # Windows has no POSIX directory fsync; file data is flushed.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


if __name__ == '__main__':
    if sys.argv[1:] != ['--command']: raise SystemExit('Internal command trampoline')
    configure_stdio()
    # Wait for the parent's successful job assignment before any user command.
    argv = json.loads(sys.stdin.readline())
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
    raise SystemExit(child.wait())
