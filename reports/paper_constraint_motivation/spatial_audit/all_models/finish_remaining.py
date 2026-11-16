"""Use the available GPU for unstarted partitions when a Slurm lane finishes."""
import json
import os
from pathlib import Path
import subprocess


def main():
    stage = Path(os.environ['HIPPO_SPATIAL_STAGE'])
    results = Path(os.environ['HIPPO_SPATIAL_RESULTS'])
    base = stage.parent
    scratch = base / 'full_data_baselines_20261115/matched_MNI/source'
    medical = base / 'full_data_baselines_20261115/cluster_medical/source'
    jobs = json.loads((stage / 'jobs.json').read_text())
    order = sorted(jobs, key=lambda k: (jobs[k]['fold'], jobs[k]['dataset'], jobs[k]['model']), reverse=True)
    for key in order:
        out = results / key
        if (out / 'complete.json').exists() or (out / 'log.txt').exists():
            continue
        out.mkdir(exist_ok=True)
        is_medical = jobs[key]['model'] == 'medsam3'
        python = (base / 'medsam3_runtime_20260928/venv/bin/python' if is_medical
                  else base / '.conda/envs/hippocampus/bin/python')
        sources = (medical, scratch) if is_medical else (scratch, medical)
        paths = [str(p) for root in sources for p in (root, root / 'scripts', root / 'baselines/nnUNet')]
        with (out / 'log.txt').open('a') as log:
            print('Available GPU processing', key, flush=True)
            subprocess.run([str(python), '-u', str(stage / 'worker.py'), key, '--single'],
                           cwd=stage, env=dict(os.environ, PYTHONPATH=os.pathsep.join(paths)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = results.parent / 'paper_spatial_all_models_20261116_summary/all_models/summarize_models.py'
    if summary.is_file() and all((results / k / 'complete.json').is_file() for k in jobs):
        import fcntl
        with (summary.parent / 'summary.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (summary.parent / 'spatial_results.md').exists():
                subprocess.run([str(base / '.conda/envs/hippocampus/bin/python'), str(summary)], check=True)


if __name__ == '__main__':
    main()
