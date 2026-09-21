"""Private, crash-visible state and process-safe claims; never launches a model."""
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import platform_support as platform

if os.name == 'nt':
    import msvcrt
else:
    import fcntl


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            platform.private_file(name)
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers do not share DELETE by default. Keep publication
        # atomic and retry only the temporary sharing violation, never a task.
        until = time.monotonic() + 1
        while True:
            try:
                os.replace(name, path)
                break
            except OSError as exc:
                if not platform.WINDOWS or getattr(exc, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= until:
                    raise
                time.sleep(.01)
        platform.fsync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            if os.name == 'nt':
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK): raise
            raise ValueError('Task/command is already running; inspect status, do not repeat execution') from exc
        yield
    finally:
        os.close(fd)


def task_directory(root, task_id):
    if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 160:
        raise ValueError('task_id must contain 1..160 characters')
    return Path(root) / 'tasks' / hashlib.sha256(task_id.encode()).hexdigest()
