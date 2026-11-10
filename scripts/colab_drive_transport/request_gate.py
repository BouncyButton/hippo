"""Pace API calls and share throttle cooldowns across workers on this VM only."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import threading
import time

DEFAULT_ROOT = '/content/hippo_drive_request_gate'


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'gate.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def store(root, value):
    temporary = root/f'gate.{os.getpid()}.{threading.get_ident()}.tmp'
    temporary.write_text(json.dumps(value))
    temporary.replace(root/'gate.json')


def pace(root=None, spacing=.1):
    root = Path(root or os.environ.get('HIPPO_DRIVE_REQUEST_GATE', DEFAULT_ROOT))
    while True:
        with locked(root):
            path = root/'gate.json'
            value = json.loads(path.read_text()) if path.exists() else {}
            now = time.time()
            delay = max(value.get('next_request', 0), value.get('cooldown_until', 0))-now
            if delay <= 0:
                value.update(next_request=now+spacing, requests=value.get('requests',0)+1)
                store(root, value)
                return
        time.sleep(min(delay, 1.))


def cooldown(attempt, root=None):
    root = Path(root or os.environ.get('HIPPO_DRIVE_REQUEST_GATE', DEFAULT_ROOT))
    # Never consume the scientific Python RNG to choose retry jitter.
    delay = min(2**(attempt+1), 32)+random.SystemRandom().random()
    with locked(root):
        path = root/'gate.json'
        value = json.loads(path.read_text()) if path.exists() else {}
        value['cooldown_until'] = max(value.get('cooldown_until', 0), time.time()+delay)
        value['throttles'] = value.get('throttles',0)+1
        store(root, value)
    print('DRIVE WAIT: shared retry cooldown', round(delay,1), 'seconds; no progress claimed', flush=True)


def retryable(exc):
    status = getattr(getattr(exc, 'resp', None), 'status', None)
    text = str(exc).lower()
    return status in (429,500,502,503,504) or any(word in text for word in (
        'userratelimitexceeded','ratelimitexceeded','user rate limit exceeded'))


def request_call(operation, attempts=6):
    for attempt in range(attempts):
        pace()
        try: return operation()
        except Exception as exc:
            if not retryable(exc) or attempt+1 == attempts: raise
            cooldown(attempt)
