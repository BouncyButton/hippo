"""Stage frequent checkpoint writes in Colab shared memory; keep recovery on SSD."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, environment, output):
    root = Path('/dev/shm')
    marker = output / 'active_output.json'
    active = root / 'hippo_full_data' / output.parent.parent.name / output.name
    needs_resume = marker.is_file()
    if needs_resume:
        if json.loads(marker.read_text())['path'] != str(active) or not active.is_dir():
            raise RuntimeError('Staged output missing; inspect the existing SSD recovery before resuming')
    else:
        # Existing jobs and VMs without ample memory retain their original path.
        memory = dict((line.split(':')[0], line.split(':')[1].split()[0])
                      for line in Path('/proc/meminfo').read_text().splitlines())
        if (not root.is_mount() or shutil.disk_usage(root).free < 8 * 1024**3
                or int(memory['MemAvailable']) < 14 * 1024**2
                or any(p.name != 'training.log' for p in output.iterdir())):
            subprocess.run(command, env=environment, check=True)
            return
        active.mkdir(parents=True, exist_ok=False)
        tmp = marker.with_suffix('.tmp')
        tmp.write_text(json.dumps({'path': str(active), 'recovery': environment['HIPPO_COLAB_RECOVERY']}))
        tmp.replace(marker)
    args = list(command)
    args[args.index('--output') + 1] = str(active)
    if not (active / 'completion.json').is_file():
        subprocess.run(args, env=environment, check=True)
    # A partial copy never advertises completion. Retrying reuses finished training.
    files = sorted(active.iterdir(), key=lambda p: (p.name == 'completion.json', p.name))
    assert files and files[-1].name == 'completion.json'
    for source in files:
        if not source.is_file():
            raise RuntimeError(f'Unexpected staged output: {source.name}')
        destination = output / source.name
        tmp = destination.with_name(destination.name + '.copying')
        shutil.copyfile(source, tmp)
        if digest(source) != digest(tmp):
            raise OSError(f'Checkpoint transfer failed verification: {source.name}')
        os.replace(tmp, destination)
    marker.unlink()
    for source in files:
        source.unlink()
    active.rmdir()
