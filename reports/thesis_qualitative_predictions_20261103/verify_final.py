"""Verify archived qualitative reproductions before using them in the thesis."""
from pathlib import Path
import hashlib
import json
import tarfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
read = lambda p: json.loads(p.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def rawsha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()

tasks = read(ROOT / 'payload/plan.json')['tasks']
report = dict(checked_at=time.time(), status='passed', archives=[], runs=[], pairs=[],
              previous_checks=read(ROOT / 'partial_verification.json'),
              original_aggregates_replaced=False, further_training_submitted=False)
for receipt_path in sorted((ROOT / 'receipts').glob('*.tar.gz.json')):
    receipt = read(receipt_path)
    archive = Path(receipt['local_archive'])
    assert receipt['status'] == 'verified_off_cluster' and sha(archive) == receipt['sha256']
    prefix = archive.name.removesuffix('.tar.gz')
    extracted = ROOT / 'recovered' / prefix
    manifest = read(extracted / 'archive_manifest.json')['files']
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        assert all(m.isfile() and m.name.startswith(prefix + '/') and '..' not in Path(m.name).parts for m in members)
        assert {m.name.removeprefix(prefix + '/') for m in members} == set(manifest) | {'archive_manifest.json'}
        for m in members:
            rel = m.name.removeprefix(prefix + '/')
            expected = manifest.get(rel, sha(extracted / rel))
            assert hashlib.sha256(tf.extractfile(m).read()).hexdigest() == expected
            assert sha(extracted / rel) == expected
    report['archives'].append(dict(archive=archive.name, members=len(members), sha256=receipt['sha256']))
assert len(report['archives']) == 7

all_records = {}
for task in tasks:
    folder = ROOT / 'recovered' / task['id']
    protocol = read(folder / 'protocol.json')
    summary = read(folder / 'training_summary.json')
    done = read(folder / 'completion.json')
    identity = read(folder / 'prediction_identity.json')
    records = read(folder / 'evaluation.json')['records']
    history = read(folder / 'training_history.json')
    assert done['status'] == 'complete' and done['updates'] == task['updates'] == summary['updates']
    assert read(folder / 'qualitative_complete.json')['status'] == 'complete'
    assert [r['update'] for r in history] == list(range(1, task['updates'] + 1))
    assert summary['frozen_before'] == summary['frozen_after']
    assert identity['adapter_sha256'] == summary['adapter_after']
    assert identity['protocol_sha256'] == sha(folder / 'protocol.json')
    assert summary['schedule_sha256'] == sha(folder / 'slab_schedule.json')
    expected_cases = {('training_support', c) for c in protocol['train_cases']} | {('validation', c) for c in protocol['evaluation_cases']}
    assert {(r['split'], r['case']) for r in records} == expected_cases
    assert len(records) == len(expected_cases)
    k = 3 if task['dataset'] == 'MSD' else 2
    for record in records:
        p = folder / record['artifact']
        assert sha(p) == record['artifact_sha256']
        with np.load(p, allow_pickle=False) as data:
            y, pred = data['truth'], data['prediction']
            assert list(y.shape) == record['shape'] and y.shape == pred.shape == data['image'].shape
            assert rawsha(y) == record['truth_sha256'] and rawsha(pred) == record['prediction_sha256']
            assert y.min() >= 0 and pred.min() >= 0 and y.max() < k and pred.max() < k
            cm = np.bincount((y.astype(np.int64) * k + pred).ravel(), minlength=k*k).reshape(k, k)
            den = cm.sum(0) + cm.sum(1)
            dice = np.divide(2 * np.diag(cm), den, out=np.ones(k), where=den > 0)
            metrics = record['qualitative_metrics']
            assert cm.tolist() == metrics['confusion']
            assert np.allclose(dice, metrics['per_class_dice'], rtol=0, atol=1e-12)
            assert abs(dice[1:].mean() - metrics['foreground_dice']) < 1e-12
            assert abs(dice.mean() - metrics['all_class_dice']) < 1e-12
            assert abs(dice[0] - metrics['background_dice']) < 1e-12
            assert int(cm[1:, 0].sum()) == metrics['foreground_fn']
            assert int(cm[0, 1:].sum()) == metrics['foreground_fp']
            assert (int(cm[1, 2] + cm[2, 1]) if k == 3 else None) == metrics['ap_swaps']
    for split, key in [('validation', 'validation_dice'), ('training_support', 'training_dice')]:
        assert abs(np.mean([r['qualitative_metrics']['foreground_dice'] for r in records if r['split'] == split]) - done[key]) < 1e-12
    original_checks = {}
    if task['dataset'] == 'ADNI':
        old = REPO / 'reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/runs' / task['id']
        for name in ['protocol.json', 'slab_schedule.json', 'training_inventory.json', 'calibration.json']:
            assert read(folder / name) == read(old / name)
        original_summary = read(old / 'training_summary.json')
        for key in ['adapter_before', 'frozen_before', 'weights']:
            assert summary[key] == original_summary[key]
        original_records = {(r['split'], r['case_name']): r for r in read(old / 'evaluation.json')['records']}
        assert set(original_records) == expected_cases
        assert all(r['truth_sha256'] == original_records[(r['split'], r['case'])]['truth_sha256'] for r in records)
        first_old = read(old / 'training_history.json')[0]
        assert all(history[0][key] == first_old[key] for key in ['native', 'bands', 'edge', 'sdf', 'depth', 'total'])
        original_checks = dict(protocol_sampling_inventory_calibration_initial_weights_reference_hashes_match=True,
                               first_forward_losses_identical=True,
                               first_gradient_original=first_old['gradient_norm'],
                               first_gradient_reproduction=history[0]['gradient_norm'])
    report['runs'].append(dict(id=task['id'], volumes=len(records), held_out=len(protocol['evaluation_cases']),
                              final_checkpoint_sha256=done['checkpoint_sha256'], prediction_identity=identity,
                              metrics_recomputed=True, original_checks=original_checks,
                              comparison=read(folder / 'original_comparison.json')))
    all_records[task['id']] = {(r['split'], r['case']): r for r in records}

for dataset in ['GeoFWI3D', 'MSD', 'ADNI']:
    base, aux = [next(t for t in tasks if t['dataset'] == dataset and t['role'] == role) for role in ['baseline', 'constrained']]
    paths = [ROOT / 'recovered' / t['id'] for t in [base, aux]]
    a, b = [all_records[t['id']] for t in [base, aux]]
    assert a.keys() == b.keys()
    for filename in ['training_inventory.json', 'slab_schedule.json']:
        assert read(paths[0] / filename) == read(paths[1] / filename)
    for key in a:
        with np.load(paths[0] / a[key]['artifact']) as x, np.load(paths[1] / b[key]['artifact']) as y:
            for name in ['image', 'truth', 'affine', 'classes']:
                assert np.array_equal(x[name], y[name]), (dataset, key, name)
    val = [dict(case=c, delta_pp=100*(b[(s,c)]['qualitative_metrics']['foreground_dice'] - a[(s,c)]['qualitative_metrics']['foreground_dice'])) for s,c in a if s == 'validation']
    ordered = sorted(val, key=lambda x: (x['delta_pp'], x['case']))
    selection = read(ROOT / 'recovered/figures' / (dataset + '_selection.json'))
    for actual, expected in zip(selection['examples'], [ordered[len(ordered)//2], ordered[0], ordered[-1]]):
        assert actual['case'] == expected['case'] and abs(actual['delta_pp'] - expected['delta_pp']) < 1e-10
        with np.load(paths[0] / a[('validation', actual['case'])]['artifact']) as x:
            assert actual['slice_axis'] == 2 and actual['slice_zero_based'] == int(np.argmax((x['truth'] > 0).sum(axis=(0,1))))
    report['pairs'].append(dict(dataset=dataset, matched_images_masks_affines_classes=True,
                               matched_training_inventory_sampling=True, predetermined_selection_verified=True,
                               original_gain_pp=100*(aux['expected_dice']-base['expected_dice']),
                               reproduced_gain_pp=float(np.mean([x['delta_pp'] for x in val])),
                               positive_cases=sum(x['delta_pp'] > 0 for x in val), held_out_cases=len(val)))

report['discrepancy_assessment'] = (
    'Original protocols, sampling, initial weights and reference identities were checked against archived records. '
    'Optimization diverges before inference. ADNI and GeoFWI3D first forward losses agree while first gradient norms differ; '
    'MSD baseline forward losses already differ. Exact kernel/stochastic cause is unestablished. '
    'MSD recorded GPU differs (original A100-SXM4-80GB versus A100 PCIe MIG 4g40gb). '
    'Medical original calibration values were reused; geological five-update calibration was repeated and differs for edge/depth. '
    'All three paired mean gains remain positive. New outputs are qualitative reproductions, not recovered original checkpoints '
    'and not replacement evidence for the original nine-cell aggregates. See previous_checks for full investigation.')
(ROOT / 'final_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(status='passed', archives=len(report['archives']), runs=len(report['runs']),
                     volumes=sum(x['volumes'] for x in report['runs']), pairs=report['pairs']), indent=2))
