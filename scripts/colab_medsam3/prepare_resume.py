"""Publish a complete resume notebook using the already uploaded scientific ZIP."""
import ast
import hashlib
import json
from pathlib import Path

from scripts.colab_medsam3.prepare_parallel import ROOT, SOURCE, digest


def main():
    previous = ROOT/'reports/medsam3_msd_colab_108_fast_eval_20261015'
    export = ROOT/'reports/medsam3_msd_colab_resume_20261018'
    export.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((previous/'MedSAM3_MSD_108_parallel_fast_evaluation.ipynb').read_text())
    payload = {name: (SOURCE/name).read_text() for name in ('parallel.py', 'fast_eval.py')}
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in payload.items()}
    # The evaluation path is exactly the already published, audited version.
    import zipfile
    with zipfile.ZipFile(previous/'medsam3_msd_108_fast_eval_code.zip') as archive:
        assert archive.read('controller/fast_eval.py') == payload['fast_eval.py'].encode()
    install = f'''

# Resume-only scheduling fix; the frozen bundle and evaluation code stay identical.
resume_sources = {payload!r}
resume_hashes = {hashes!r}
resume_controller = Path('/content/hippo_medsam3_parallel_resume_v3')
import fcntl
with (BUNDLE/'work/sweep.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, value in resume_sources.items():
        target = resume_controller/name
        expected = resume_hashes[name]
        if hashlib.sha256(value.encode()).hexdigest() != expected:
            raise RuntimeError('Resume source hash mismatch')
        if target.is_symlink() or (target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected):
            raise RuntimeError('Existing resume controller differs; preserving it: '+str(target))
    resume_controller.mkdir(parents=True, exist_ok=True)
    for name, value in resume_sources.items():
        target = resume_controller/name
        if not target.exists():
            temporary = target.with_suffix('.staging')
            temporary.write_text(value)
            temporary.replace(target)
    (RESULTS/'resume_controller.json').write_text(json.dumps(dict(files=resume_hashes,
        scientific_bundle_changed=False, augmentation='none', total_experiments=108), indent=2))
CONTROLLER = resume_controller
print('Drive quota retry controller ready. Completed work and scientific files preserved.')
'''
    for cell in notebook['cells']:
        if cell['id'] == 'parallel_02':
            cell['source'] = cell['source'].replace('NOTEBOOK_STARTED = time.monotonic()',
                "NOTEBOOK_STARTED = globals().get('NOTEBOOK_STARTED', time.monotonic())")
        if cell['id'] == 'parallel_04':
            cell['source'] += install
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['cells'][0]['source'] = '''# Resume the existing MSD MedSAM3 campaign

Run all cells in order on the same A100-SXM4-80GB type and software environment,
signed into the same Drive account. Keep the existing input files and
`results_matched200_noaug_108_v1` folder. No new ZIP or data upload is required:
this complete notebook includes the updated scheduler and uses the already
uploaded `medsam3_msd_108_fast_eval_code.zip`, `medsam3_msd_data.zip` and `sam3.pt`.

The previous session stopped on a temporary Drive API rate limit. Quota checks now
retry with capped exponential backoff (up to 10 retries) while existing workers
continue. New workers still require a successful fresh storage check. Persistent
errors trigger the original clean checkpoint shutdown. Manual STOP and the
22-hour session stop are still respected during retries.

Legacy recovery archives omitted the training-history JSON sidecar. This version
restores a missing sidecar from the exact history already saved inside the
checkpoint, after checking its protocol, update ledger and completion hash.
It does not replay updates or redo completed evaluation. Future clean-stop
supplements also include this sidecar.

Completed experiments are verified and skipped. Interrupted evaluation resumes
from its saved checkpoint and completed volumes. Matching calibration and
preflight are reused. The original 108 task identities, 200-update recipe,
no augmentation, medical adapter and LoRA weights remain unchanged. Bands and
bands+edge are not repeated; the 27 historical results remain reference only.
Use only one runtime writing this results folder. This is a continuation, not
another set of 108 runs.
'''
    notebook['cells'][1]['source'] = notebook['cells'][1]['source'].split('**Switch only')[0]
    path = export/'MedSAM3_MSD_108_resume.ipynb'
    notebook['metadata']['colab']['name'] = path.name
    notebook['metadata']['accelerator'] = 'GPU'
    notebook['metadata']['colab']['machine_shape'] = 'hm'
    notebook['metadata']['colab']['gpuClass'] = 'premium'
    notebook['metadata']['colab']['gpuType'] = 'A100'
    path.write_text(json.dumps(notebook, indent=1)+'\n')
    record = dict(notebook=str(path), notebook_sha256=digest(path), controller_hashes=hashes,
                  frozen_bundle_sha256='716f2e774fb9de3140baf1077eabff75fc0dc144f9a9e6752ec620dc5ddd6ac2',
                  training_changed=False, evaluation_changed=False, total_runs=108, augmentation='none',
                  existing_uploads_reused=True, gpu_test=False)
    (export/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
