"""Yield the matrix at an archived individual-run boundary; always resume it."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main(root):
    work = root/'nnunet_long_constraints_v1'; work.mkdir(exist_ok=True)
    audit = {'status': 'checking_matrix', 'policy': 'Never signal a training child; finish and archive the current individual run'}
    def save(): (work/'handover.json').write_text(json.dumps(audit, indent=2)+'\n')
    dispatchers = [p for p in psutil.process_iter(['cmdline','status'])
        if p.info['status'] != 'zombie' and str(root/'resume_three_models.py') in (p.info['cmdline'] or [])]
    if len(dispatchers) > 1: raise RuntimeError('Ambiguous matrix controller')
    dispatcher = dispatchers[0] if dispatchers else None
    stopped = retired = False; stop = None
    env = dict(os.environ, PYTHONPATH=':'.join(str(root/'bundle'/p) for p in
        ('compatibility','source','source/scripts','source/baselines/nnUNet')), PYTHONUNBUFFERED='1')
    try:
        if dispatcher:
            command, cwd = dispatcher.cmdline(), dispatcher.cwd()
            dispatcher.suspend(); stopped = True
            state = json.loads((root/'constraint_search/search_status.json').read_text())
            current = state.get('current')
            if not current or Path(current).name != current: raise RuntimeError('No unambiguous current recipe')
            candidate = root/'constraint_search'/current/'STOP'
            if candidate.exists(): raise RuntimeError('Existing STOP belongs to another operation')
            candidate.write_text('Temporary between-run handover for nnunet_long_constraints_v1\n'); stop = candidate
            audit.update(status='waiting_for_current_run_archive', dispatcher=dispatcher.pid, recipe=current); save()
            print('WAITING_FOR_ARCHIVED_RUN', current, flush=True)
        with (root/'bands_campaign.lock').open('a') as lock:
            deadline = time.time()+1800
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.time() > deadline: raise TimeoutError('Current run still active; matrix retained')
                    time.sleep(5)
            if dispatcher:
                while any(p.status() != 'zombie' for p in dispatcher.children(recursive=True)):
                    if time.time() > deadline: raise TimeoutError('Recipe still archiving; matrix retained')
                    time.sleep(2)
                if dispatcher.status() != 'stopped': raise RuntimeError('Dispatcher state changed')
                dispatcher.kill(); retired = True
                audit.update(retired_idle_dispatcher=True, status='pilot_starting'); save()
            fcntl.flock(lock, fcntl.LOCK_UN)
        result = subprocess.run([sys.executable, '-m', 'scripts.colab_hfh.nnunet_long_constraints'],
            cwd=root/'bundle/source', env=env)
        audit.update(pilot_returncode=result.returncode, status='pilot_finished'); save()
    finally:
        if stop is not None and stop.exists(): stop.unlink()
        if retired:
            with (root/'three_model_recovery_after_constraints.log').open('a') as log:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            audit.update(new_dispatcher=process.pid, status='matrix_resumed'); save()
            print('MATRIX_RESUMED', process.pid, flush=True)
        elif stopped and dispatcher.is_running():
            dispatcher.resume(); audit.update(status='original_matrix_resumed'); save()


if __name__ == '__main__': main(Path('/content/hippo_hfh_baselines_v2'))
