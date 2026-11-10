"""Remove only the snake work explicitly retired by the user."""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    base = Path('/mnt/beegfsstudents/home/3160552')
    queue = subprocess.check_output(
        ['squeue', '-u', '3160552', '-h', '-o', '%i|%j|%T'], text=True
    )
    active_snakes = [line for line in queue.splitlines()
                     if 'snake' in line.lower() or line.split('|')[0] in {'681383', '682190'}]
    if active_snakes:
        raise RuntimeError(f'Refusing cleanup while a snake job is active: {active_snakes}')
    names = ['snakes_swinunetr_fold0_seed0_20261003_01',
             'snakes_swinunetr_fold0_seed0_20261003_01.tar.gz',
             'snakes_swinunetr_fold0_seed0_20261004_100e_01',
             'hippo/datasets/Dataset201_PARTIMAGENET']
    removed = []
    for name in names:
        path = base / name
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(base):
            raise RuntimeError(f'Unexpected path: {path}')
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
    record = {'deleted_utc': datetime.now(timezone.utc).isoformat(),
              'authorization': 'User requested deletion of all snake experiment work.',
              'removed_paths': removed, 'queue_at_cleanup': queue,
              'former_jobs': ['681383', '682190']}
    (base / 'snake_retirement_20261004.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
