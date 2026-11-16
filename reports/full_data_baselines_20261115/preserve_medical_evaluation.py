"""Save a completed training checkpoint during a long, resumable volume evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from continue_colab import FOLDER, existing, publish, session, sha


def restore(key):
    assert key == 'ADNI_medsam3_f0_s0_coverage_control'
    root = Path('/content/covered_medical_ADNI')
    root.mkdir(exist_ok=True)
    (root / 'work').mkdir(exist_ok=True)
    dest = root / 'work' / key
    if dest.exists():
        raise RuntimeError('Restore only into an empty replacement runtime before starting the worker')
    drive = session()
    name = key + '_final_training_recovery.zip'
    file = existing(drive, name)
    if not file:
        raise FileNotFoundError(name)
    blob = root / name
    if not blob.is_file() or sha(blob) != file['sha256Checksum']:
        part = blob.with_suffix('.download')
        with part.open('wb') as stream:
            download = MediaIoBaseDownload(stream, drive.files().get_media(fileId=file['id']), chunksize=8 * 1024**2)
            done = False
            while not done:
                _, done = download.next_chunk(num_retries=5)
        assert sha(part) == file['sha256Checksum']
        part.replace(blob)
    import tempfile
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        ready = Path(tmp) / key
        ready.mkdir()
        with zipfile.ZipFile(blob) as z:
            assert z.testzip() is None
            assert all(Path(n).name == n and (n.endswith('.json') or n == 'checkpoint_latest.pt') for n in z.namelist())
            z.extractall(ready)
        ident = json.loads((ready / 'recovery_identity.json').read_text())
        assert ident['key'] == key and ident['updates'] == 200 and ident['eligible_crops'] == 216
        assert sha(ready / 'checkpoint_latest.pt') == ident['checkpoint_sha256']
        assert sha(ready / 'protocol.json') == ident['protocol_sha256']
        later = existing(drive, key + '_evaluation_progress.json')
        if later:
            data = drive.files().get_media(fileId=later['id']).execute(num_retries=5)
            assert hashlib.sha256(data).hexdigest() == later['sha256Checksum']
            delta = json.loads(data)
            assert delta['recovery_sha256'] == file['sha256Checksum'] and delta['checkpoint_sha256'] == ident['checkpoint_sha256']
            path = ready / 'evaluation.json'
            old = json.loads(path.read_text())['records'] if path.is_file() else []
            assert delta['records'][:len(old)] == old
            path.write_text(json.dumps({'records': delta['records']}, allow_nan=False))
        ready.rename(dest)
    print('Restored verified final training weights and case evaluations', key, flush=True)


def run(key):
    assert key == 'ADNI_medsam3_f0_s0_coverage_control'
    root = Path('/content/covered_medical_ADNI')
    out = root / 'work' / key
    drive = session()
    name = key + '_final_training_recovery.zip'
    blob = root / name
    protocol = json.loads((out / 'protocol.json').read_text())
    assert protocol['name'] == key and protocol['arm'] == 'dice'
    order = json.loads((out / 'slab_schedule.json').read_text())
    assert len(order) == 400 and len({c for c, _ in order}) == len(protocol['train_cases']) == 216
    assert json.loads((out / 'checkpoint_ledger.json').read_text())['step'] == protocol['updates'] == 200
    assert json.loads((out / 'training_summary.json').read_text())['updates'] == 200
    weights = out / 'checkpoint_latest.pt'
    digest = sha(weights)
    prior = existing(drive, name)
    if prior is None:
        with zipfile.ZipFile(blob.with_suffix('.partial'), 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            z.write(weights, weights.name)
            for path in sorted(out.glob('*.json')):
                z.writestr(path.name, path.read_bytes())
            z.writestr('recovery_identity.json', json.dumps(dict(key=key, checkpoint_sha256=digest,
                protocol_sha256=sha(out / 'protocol.json'), updates=200, eligible_crops=216)))
        blob.with_suffix('.partial').replace(blob)
        with zipfile.ZipFile(blob) as z:
            assert z.testzip() is None and hashlib.sha256(z.read(weights.name)).hexdigest() == digest
        quota = drive.about().get(fields='storageQuota(limit,usage)').execute(num_retries=4)['storageQuota']
        free = int(quota['limit']) - int(quota['usage'])
        print('Recovery bytes', blob.stat().st_size, 'Drive bytes free', free, flush=True)
        if free < blob.stat().st_size + 24 * 1024**2:
            raise RuntimeError('The verified recovery is ready on the VM for direct download; Drive space is insufficient')
        prior = publish(drive, key, blob)
    assert blob.is_file() and sha(blob) == prior['sha256Checksum']
    print('Final training recovery verified', prior['id'], prior['sha256Checksum'], flush=True)
    evaluation = out / 'evaluation.json'
    sidecar = root / (key + '_evaluation_progress.json')
    remote = existing(drive, sidecar.name)
    last_count = 0
    while not (out / 'completion.json').exists():
        if evaluation.is_file():
            data = json.loads(evaluation.read_bytes())
            records = data['records']
            assert len({(r['split'], r['case_name']) for r in records}) == len(records)
            if len(records) > last_count:
                payload = dict(recovery_id=prior['id'], recovery_sha256=prior['sha256Checksum'],
                               checkpoint_sha256=digest, records=records)
                sidecar.write_text(json.dumps(payload, allow_nan=False))
                upload = MediaFileUpload(str(sidecar), mimetype='application/json', resumable=True)
                if remote:
                    req = drive.files().update(fileId=remote['id'], media_body=upload, fields='id,sha256Checksum')
                else:
                    req = drive.files().create(body=dict(name=sidecar.name, parents=[FOLDER]),
                                               media_body=upload, fields='id,sha256Checksum')
                remote = req.execute(num_retries=4)
                assert remote['sha256Checksum'] == sha(sidecar)
                print('Verified case evaluations', len(records), remote['id'], flush=True)
                last_count = len(records)
        time.sleep(90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('key')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    (restore if args.restore else run)(args.key)
