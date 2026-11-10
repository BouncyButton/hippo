"""Publish the complete 108-run notebook with guarded evaluation acceleration."""
import ast
import hashlib
import json
import zipfile

from scripts.colab_medsam3.prepare_parallel import ORIGINAL, ROOT, SOURCE, digest, notebook
from scripts.colab_medsam3.parallel import FROZEN_MANIFEST

EXPORT = ROOT/'reports/medsam3_msd_colab_108_fast_eval_20261015'
ZIP_NAME = 'medsam3_msd_108_fast_eval_code.zip'
NOTEBOOK_NAME = 'MedSAM3_MSD_108_parallel_fast_evaluation.ipynb'


def fast_notebook(uploads, package_sha):
    value = notebook(uploads, package_sha)
    replacements = {
        'medsam3_msd_108_parallel_code.zip': ZIP_NAME,
        '/content/hippo_medsam3_parallel_v1': '/content/hippo_medsam3_parallel_fast_eval_v1',
        '# MSD MedSAM3 — resume the same 108 runs with adaptive concurrency':
            '# MSD MedSAM3 — the same 108 runs, with faster evaluation',
        'Only scheduling changes. Production CUDA numerical variability remains; concurrency is not a\nguarantee of bitwise agreement with serial execution.':
            'Training and calibration use the unchanged engine. Evaluation can reuse image/text features\n'
            'only after the checks below. Production CUDA numerical variability remains; concurrency\n'
            'is not a guarantee of bitwise agreement with serial execution.',
        "'--hours', str(remaining_hours)]": "'--hours', str(remaining_hours), '--fast-evaluation']",
        '**GPU usage, worker count, update/200 and evaluated volumes**':
            '**GPU usage, worker count, update/200, evaluated volumes, current slice and evaluation backend**',
    }
    found = {k: 0 for k in replacements}
    for item in value['cells']:
        for old, new in replacements.items():
            found[old] += item['source'].count(old)
            item['source'] = item['source'].replace(old, new)
        if item['cell_type'] == 'code': ast.parse(item['source'])
    if any(n == 0 for n in found.values()):
        raise ValueError(f'Notebook template changed: {found}')
    details = '''## Evaluation acceleration and verification

The original evaluator encodes each 1008×1008 slice twice, once per class prompt.
This version can encode it once, reuse only its prompt-independent image features,
and reuse the two text encodings within this trained checkpoint. Both prompt-conditioned
decoders, all slices, both classes, image resolution, BF16 precision and volume metrics
remain the same. No augmentation, batching change, extra experiments or optimizer updates.

When a run reaches evaluation, the **first unfinished volume** is compared slice by slice
against the original path. The original predictions produce that volume's metrics.
The faster path is enabled only if all predicted labels are identical, log probabilities
pass the existing strict tolerances (rtol 1e-5, atol 1e-7), RNG remains unchanged, and
measured slice time improves by at least 2%. The first slice of each subsequent volume
is checked again. A failed comparison or absent speed gain automatically keeps the
original evaluator for that checkpoint. No tolerance is relaxed to obtain a pass.

This adds one volume's candidate inference per checkpoint plus one checked slice per
remaining volume. There are no throwaway training runs or repeated calibration. GPU
speed is measured on your runtime; the package's local tests cannot establish A100 speed.
Look for `EVAL FAST` and the observed slice speedup in the worker's `train.log`; the
notebook shows `audit`, `fast`, or `original` and current slice progress every 30 seconds.

Completed archives and already evaluated volumes are skipped. A clean stop preserves
partial evaluation. The audit is reused only for the same checkpoint, code and environment.
Receipts and extension source are saved under `RESULTS/evaluation_extension/`; the frozen
training recipe and archive layout stay unchanged. The feature caches are scoped to one
checkpoint and are never used during training or calibration.

**Switch only after the previous Run cell has finished its clean stop and verified Drive
recovery. Upload the new code ZIP, then run this complete notebook from the top using the
same results folder. Keep your existing data ZIP and sam3.pt. Do not run both notebooks
against the results folder at once.** If keeping an older VM, reduce SESSION_HOURS to its
remaining session time; opening this notebook does not renew the Colab runtime.
'''
    value['cells'].insert(1, dict(cell_type='markdown', metadata={}, source=details, id='fast_evaluation_details'))
    value['metadata']['colab']['name'] = NOTEBOOK_NAME
    return value


def main():
    EXPORT.mkdir(parents=True, exist_ok=True)
    transfer = json.loads((ORIGINAL/'transfer_manifest.json').read_text())
    original_zip = ORIGINAL/'medsam3_msd_108_code.zip'
    if digest(original_zip) != transfer['uploads'][original_zip.name]:
        raise ValueError('Original code ZIP changed')
    with zipfile.ZipFile(original_zip) as archive:
        payload = {'bundle/'+name: archive.read(name) for name in archive.namelist()}
    for name in ('parallel.py', 'fast_eval.py'):
        payload['controller/'+name] = (SOURCE/name).read_bytes()
    members = {name: hashlib.sha256(content).hexdigest() for name, content in payload.items()}
    payload['package_manifest.json'] = (json.dumps(members, indent=2)+'\n').encode()
    target = EXPORT/ZIP_NAME
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(payload.items()): archive.writestr(name, content)
    package_sha = digest(target)
    nb = EXPORT/NOTEBOOK_NAME
    nb.write_text(json.dumps(fast_notebook(transfer['uploads'], package_sha), indent=1)+'\n')
    record = dict(code_zip=target.name, code_sha256=package_sha, notebook=nb.name,
        notebook_sha256=digest(nb), original_bundle_sha256=FROZEN_MANIFEST,
        frozen_files_unchanged=True, training_calibration_unchanged=True,
        evaluation_extension_sha256=members['controller/fast_eval.py'],
        existing_results_preserved=True, same_results_folder='results_matched200_noaug_108_v1',
        total_experiments=108, added_experiments=0, updates=200, augmentation='none',
        local_gpu_test=False, gpu_guard='full volume, exact labels, strict logits, unchanged RNG, measured speed gain',
        data_uploads_unchanged={k: v for k, v in transfer['uploads'].items() if k in ('medsam3_msd_data.zip', 'sam3.pt')})
    (EXPORT/'transfer_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    (EXPORT/'UPLOAD_AND_RESUME.txt').write_text(
        '1. Interrupt the CURRENT Run cell once. Wait until every worker verifies its Drive recovery and the cell returns.\n'
        '2. Upload '+ZIP_NAME+' to MyDrive/hippo_medsam3_msd_extension/input/. Keep the existing data ZIP and sam3.pt.\n'
        '3. Open '+NOTEBOOK_NAME+' in Colab. Use the same A100 type, High RAM and Drive account.\n'
        '4. Run the notebook cells from the top. Keep RESULTS as results_matched200_noaug_108_v1.\n'
        '5. Completed runs/volumes are skipped, interrupted work resumes, and matching preflight/calibration is reused.\n'
        '6. Evaluation enables feature reuse only after its checkpoint-specific equivalence and speed audit.\n'
        '7. An older VM keeps its original lifetime. Reduce SESSION_HOURS accordingly; do not run two notebooks at once.\n'
        'No augmentation. 108 existing tasks, zero added tasks. Same original 200-update training recipe.\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
