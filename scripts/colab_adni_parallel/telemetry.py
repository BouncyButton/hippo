"""Local-only observation: the heartbeat must never wait for Google Drive."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time


class LiveStatus:
    def __init__(self, directory, provider=None, interval=30., poll=5.):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.provider, self.interval, self.poll = provider, interval, poll
        self.lock = threading.RLock()
        self.state = dict(phase='starting', checked=0, verified=0, completed=0,
                          current_run=None, last_progress_at=time.time(), progress_events=0)
        self.timings = {}
        self.stopped = threading.Event()
        self.thread = None
        self.provider_thread = None
        self.last_signature = None

    def update(self, progress=False, **values):
        with self.lock:
            self.state.update(values)
            if progress:
                self.state['last_progress_at'] = time.time()
                self.state['progress_events'] += 1

    @contextmanager
    def phase(self, name, run=None, announce=True):
        with self.lock:
            previous = self.state['phase'], self.state['current_run']
        self.update(phase=name, current_run=run, phase_started_at=time.time())
        if announce:
            print('PHASE:', name, run or '', flush=True)
        self.publish()
        start = time.monotonic()
        try:
            yield
        finally:
            with self.lock:
                self.timings[name] = self.timings.get(name, 0.) + time.monotonic()-start
            self.update(phase=previous[0], current_run=previous[1])

    def publish(self, announce=False):
        # Only local SSD operations inside this lock. Never call a Drive function.
        with self.lock:
            value = dict(self.state, updated_at=time.time(), phase_seconds=dict(self.timings))
            value['seconds_since_progress'] = time.time()-value['last_progress_at']
            path = self.directory/'status.json'
            temp = self.directory/'status.tmp'
            temp.write_text(json.dumps(value, allow_nan=False)+'\n')
            temp.replace(path)
            if announce:
                with (self.directory/'timing.jsonl').open('a') as out:
                    out.write(json.dumps(value, allow_nan=False)+'\n')
        if announce:
            gpu = value.get('gpu', {})
            activity = ('progress since last heartbeat' if value['progress_events'] != self.last_signature
                        else 'STILL WAITING — no new archive-read/verification events or completed updates')
            self.last_signature = value['progress_events']
            print(f"HEARTBEAT: {activity}; phase={value['phase']} run={value['current_run'] or '-'}; "
                  f"startup archives checked={value['checked']}/{value.get('archive_total', '?')} "
                  f"verified={value['verified']}; completed={value['completed']}/504; "
                  f"workers={value.get('workers', 0)}/{value.get('target', 1)}; "
                  f"GPU activity={gpu.get('util', '?')}% VRAM={gpu.get('used_gib', '?')}/"
                  f"{gpu.get('total_gib', '?')} GiB; CPU={gpu.get('cpu_percent', '?')}%; "
                  f"RAM available={gpu.get('ram_gib', '?')} GiB; "
                  f"last progress {value['seconds_since_progress']:.0f}s ago", flush=True)
            if value.get('current_file') or value.get('archive_operation'):
                print(f"  ARCHIVE: {value.get('archive_operation', '')}; "
                      f"file={value.get('current_file') or '-'}; "
                      f"files={value.get('archive_files_copied', 0)}/{value.get('archive_files_total') or '?'}; "
                      f"read={value.get('archive_bytes_copied', 0)} bytes", flush=True)
            if value.get('launch_gate'):
                print('  LAUNCH GATE:', value['launch_gate'], flush=True)
            for item in value.get('active', []):
                print(f"  {item['id']}: {item.get('phase', 'starting')} | "
                      f"completed updates={item.get('completed_updates', '?')} "
                      f"charged={item.get('ledger', {}).get('charged_attempts', '?')}/"
                      f"{item.get('max_updates', '?')} | "
                      f"recovery={item.get('recovery_phase', 'idle')} | "
                      f"status age={time.time()-item.get('updated_at', time.time()):.0f}s", flush=True)

    def start(self):
        def sample():
            while not self.stopped.wait(self.poll):
                try:
                    self.provider()
                except Exception as exc:
                    self.update(observation_error=repr(exc))
        def observe():
            last_print = time.monotonic()
            while not self.stopped.wait(self.poll):
                try:
                    announce = time.monotonic()-last_print >= self.interval
                    self.publish(announce=announce)
                    if announce:
                        last_print = time.monotonic()
                except Exception as exc:
                    self.update(observation_error=repr(exc))
                    print('Local observation error:', repr(exc), flush=True)
        self.last_signature = self.state['progress_events']
        self.publish()
        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()
        if self.provider:
            self.provider_thread = threading.Thread(target=sample, daemon=True)
            self.provider_thread.start()

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=12)
        if self.provider_thread:
            self.provider_thread.join(timeout=12)
        self.publish(announce=True)


def directories(path):
    """One listing, including empty recovery dirs; no per-planned-task probes."""
    import os
    try:
        with os.scandir(path) as entries:
            # Include logical NVMe links, including broken ones that must be
            # audited before launch, without probing every planned task.
            return {e.name for e in entries if e.is_dir(follow_symlinks=False) or e.is_symlink()}
    except FileNotFoundError:
        return set()
