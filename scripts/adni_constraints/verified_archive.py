"""Archive ADNI scalar results off BeeGFS before retiring completed checkpoints."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shlex
import struct
import subprocess
import tarfile
import time


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def diagnostics(output):
    """Recover unique integer intersections from the exact recorded Dice formula.

    The frozen evaluator computes float32 2*TP/(predicted+target), without
    smoothing. Verify the recovered TP and its neighbours using that same
    float32 rounding; fail closed if it is not unique.
    """
    done = read(output / 'completion.json')
    assert done['status'] == 'complete'
    results = {}
    f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
    for role in ('best', 'latest'):
        path = output / f'validation_{role}.json'
        report = read(path)
        cases = []
        for row in report['cases']:
            p, t = row['predicted_voxels'], row['target_voxels']
            d = row['hippocampus_hard_dice']
            assert 0 < p + t <= 2 * 64**3
            tp = round(d * (p + t) / 2)
            candidates = [v for v in range(max(0, tp-1), min(p, t, tp+1)+1)
                          if f32(2*v/(p+t)) == d]
            assert candidates == [tp], (row['case_name'], candidates)
            cases.append(dict(case_name=row['case_name'], subject_id=row['subject_id'],
                              dice=d, tp_voxels=tp, fn_voxels=t-tp, fp_voxels=p-tp,
                              target_voxels=t, predicted_voxels=p,
                              fn_rate=(t-tp)/t, fp_fraction=(p-tp)/p if p else 0.))
        assert len(cases) == 54 and len({r['subject_id'] for r in cases}) == 27
        totals = {k: sum(r[k] for r in cases) for k in
                  ('tp_voxels', 'fn_voxels', 'fp_voxels', 'target_voxels', 'predicted_voxels')}
        results[role] = dict(validation_sha256=digest(path), cases=cases,
                             mean_dice=sum(r['dice'] for r in cases)/len(cases),
                             totals=totals, fn_rate=totals['fn_voxels']/totals['target_voxels'],
                             fp_fraction=totals['fp_voxels']/totals['predicted_voxels'])
    write(output / 'evaluation_diagnostics.json', dict(
        status='complete', method='Exact integer TP recovered and float32 checked from frozen Dice and voxel counts',
        ap_swaps=dict(status='not_applicable', reason='Binary whole-hippocampus ADNI annotations; no A/P classes'),
        selected_update=done['selected_step'], stopping_update=done['updates'],
        selected_crop_passes=done['selected_step']/10, stopping_crop_passes=done['updates']/10,
        evaluations=results))


def prepare(output):
    if (output / 'archive_manifest.json').exists():
        return read(output / 'archive_manifest.json')
    diagnostics(output)
    required = {'completion.json', 'config.json', 'metrics.csv', 'budget_ledger.json',
                'calibration.json', 'validation_best.json', 'validation_latest.json',
                'evaluation_diagnostics.json', 'runtime_recipe.json'}
    assert required <= {p.name for p in output.iterdir()}
    done = read(output / 'completion.json')
    metadata, bulky = {}, []
    for p in sorted(output.iterdir()):
        assert p.is_file() and not p.is_symlink(), p
        if p.name in ('checkpoint_best.pt', 'checkpoint_latest.pt'):
            sha = digest(p)
            assert sha == done[p.stem + '_sha256']
            bulky.append(dict(name=p.name, bytes=p.stat().st_size, sha256=sha))
        else:
            assert p.suffix in ('.json', '.csv', '.log', '.lock'), p
            metadata[p.name] = dict(bytes=p.stat().st_size, sha256=digest(p))
    manifest = dict(status='ready_for_archive', completion_sha256=digest(output/'completion.json'),
                    metadata=metadata, deletion_candidates=bulky,
                    policy='Verify durable local scalar archive before deleting checkpoints; binary ADNI A/P swaps not applicable')
    write(output/'archive_manifest.json', manifest)
    return manifest


def retire(output):
    manifest = read(output/'archive_manifest.json')
    ack = read(output/'archive_acknowledgment.json')
    assert ack['archive_manifest_sha256'] == digest(output/'archive_manifest.json')
    assert ack['status'] == 'verified' and not ack['durable_archive'].startswith('/mnt/beegfs')
    assert ack['source_archive_verified'] is True
    for name, item in manifest['metadata'].items():
        assert digest(output/name) == item['sha256']
    receipt = output/'cleanup.json'
    previous = read(receipt) if receipt.exists() else None
    if previous:
        assert previous['archive_manifest_sha256'] == ack['archive_manifest_sha256']
    for item in manifest['deletion_candidates']:
        path = output/item['name']
        if path.exists():
            assert not path.is_symlink() and digest(path) == item['sha256']
            assert path.stat().st_size == item['bytes']
        else:
            assert previous is not None, 'Checkpoint disappeared without a cleanup journal'
    record = dict(status='in_progress', archive_manifest_sha256=ack['archive_manifest_sha256'],
                  durable_archive=ack['durable_archive'], deleted=manifest['deletion_candidates'],
                  released_file_bytes=sum(r['bytes'] for r in manifest['deletion_candidates']),
                  timestamp_unix=time.time())
    write(receipt, record)
    for item in manifest['deletion_candidates']:
        (output/item['name']).unlink(missing_ok=True)
    assert not list(output.glob('*.pt'))
    record['status'] = 'complete'
    write(receipt, record)
    return record


def remote(command):
    return subprocess.check_output(['ssh', 'bocconi-cluster', shlex.join(command)])


def archive_one(output, destination, source_archive):
    """Local side: download only manifest-listed metadata, verify, then acknowledge."""
    source_receipt = read(source_archive/'source_archive.json')
    for name, sha in source_receipt['files'].items():
        assert digest(source_archive/name) == sha
    manifest_bytes = remote(['cat', str(output/'archive_manifest.json')])
    manifest = json.loads(manifest_bytes)
    names = list(manifest['metadata']) + ['archive_manifest.json']
    assert all(Path(n).name == n and n not in ('.', '..') for n in names)
    blob = remote(['tar', '-czf', '-', '-C', str(output), *names])
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        assert set(archive.getnames()) == set(names)
        assert all(m.isfile() for m in archive.getmembers())
        archive.extractall(destination, filter='data')
    assert (destination/'archive_manifest.json').read_bytes() == manifest_bytes
    for name, item in manifest['metadata'].items():
        path = destination/name
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        if path.suffix == '.json': read(path)
        elif path.suffix == '.csv':
            with path.open(newline='') as stream: assert len(list(csv.DictReader(stream))) == 20
        else: path.read_text()
    assert read(destination/'evaluation_diagnostics.json')['status'] == 'complete'
    ack = dict(status='verified', durable_archive=str(destination.resolve()),
               archive_manifest_sha256=digest(destination/'archive_manifest.json'),
               source_archive_verified=True, source_archive=str(source_archive.resolve()),
               source_archive_sha256=digest(source_archive/'source_archive.json'), verified_at_unix=time.time())
    write(destination/'archive_acknowledgment.json', ack)
    subprocess.run(['scp', str(destination/'archive_acknowledgment.json'),
                    'bocconi-cluster:'+str(output/'archive_acknowledgment.json.tmp')], check=True)
    remote(['mv', str(output/'archive_acknowledgment.json.tmp'), str(output/'archive_acknowledgment.json')])
    return ack


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'retire', 'archive'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--source-archive', type=Path)
    args = parser.parse_args()
    if args.mode == 'archive':
        result = archive_one(args.output, args.destination, args.source_archive)
    else:
        result = {'prepare': prepare, 'retire': retire}[args.mode](args.output)
    print(json.dumps(result), flush=True)
