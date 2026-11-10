"""Transfer one verified run at a time while the single cluster job executes."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import os


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def remote(code):
    command = "python3 -u - <<'PY'\n" + code + "\nPY"
    result = subprocess.run(['rtk', 'proxy', 'ssh', '-o', 'ConnectTimeout=20',
        '-o', 'ServerAliveInterval=20', 'bocconi-cluster', command],
        capture_output=True, text=True, check=True, timeout=180)
    return json.loads(result.stdout)


def snapshot(root):
    return remote('import pathlib,json,subprocess\n'
        + 'root=pathlib.Path(' + repr(str(root)) + ')\n'
        + "out={}\nfor name in ['progress.json','ready.json']:\n p=root/name\n if p.exists():out[name]=json.loads(p.read_text())\n"
        + "if (root/'summary.json').exists():\n summary=json.loads((root/'summary.json').read_text())\n if summary['status'] in ('complete','failed'):out['summary.json']=summary\n"
        + "out['queue']=subprocess.run(['squeue','-u','3160552','-h','-o','%i|%T|%M|%R'],capture_output=True,text=True).stdout.strip()\n"
        + "print(json.dumps(out))")


def receive(root, report, ready):
    run_id = ready['run_id']
    if not re.fullmatch(r'\d{2}_(nnunet|swinunetr|unetr_vit_b16|basic_unet16)_dice(?:_(?:bands|edge|sdf|depth))*', run_id):
        raise ValueError('Unexpected export identity')
    if ready['archive_name'] != run_id + '.tar' or not re.fullmatch(r'[0-9a-f]{64}', ready['archive_sha256']):
        raise ValueError('Unexpected export filename/hash')
    if shutil.disk_usage(report).free < ready['archive_bytes'] * 2 + 1024**3:
        raise RuntimeError('Local export needs additional free storage')
    download = report / 'downloads' / ready['archive_name']
    download.parent.mkdir(exist_ok=True)
    if not download.exists() or digest(download) != ready['archive_sha256']:
        partial = download.with_suffix('.part')
        subprocess.run(['rtk', 'proxy', 'scp', '-o', 'ConnectTimeout=20',
            'bocconi-cluster:' + str(root / 'outbox' / ready['archive_name']), str(partial)],
            check=True, timeout=1800)
        if digest(partial) != ready['archive_sha256']:
            raise ValueError('Downloaded archive checksum differs')
        partial.replace(download)
    destination = report / ready['model'] / ready['objective'] / 'fold0/seed0/run'
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(download) as tar:
        members = tar.getmembers()
        if any(not member.isfile() or Path(member.name).name != member.name for member in members):
            raise ValueError('Unsafe export members')
        manifest_raw = tar.extractfile('export_manifest.json').read()
        if hashlib.sha256(manifest_raw).hexdigest() != ready['export_manifest_sha256']:
            raise ValueError('Export manifest checksum differs')
        manifest = json.loads(manifest_raw)
        if {m.name for m in members} != set(manifest['files']) | {'export_manifest.json'}:
            raise ValueError('Export file membership differs')
        for member in members:
            source = tar.extractfile(member)
            target = destination / member.name
            temporary = target.with_suffix(target.suffix + '.tmp')
            with temporary.open('wb') as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
            expected = ready['export_manifest_sha256'] if member.name == 'export_manifest.json' else manifest['files'][member.name]
            if digest(temporary) != expected:
                raise ValueError('Export member checksum differs')
            temporary.replace(target)
    receipt = dict(run_id=run_id, archive_sha256=ready['archive_sha256'],
        all_files_sha256_verified=True, local_model_directory=str(destination),
        model_tensor_sha256=manifest['model_tensor_sha256'], received_at_unix=time.time())
    write_json(report / 'transfer_receipts' / (run_id + '.json'), receipt)
    acknowledgment = json.dumps(receipt)
    remote('import pathlib,json\nroot=pathlib.Path(' + repr(str(root)) + ')\n'
        + 'receipt=json.loads(' + repr(acknowledgment) + ')\n'
        + "target=root/'acks'/(receipt['run_id']+'.json')\n"
        + "target.parent.mkdir(exist_ok=True)\ntemporary=target.with_suffix('.tmp')\n"
        + "temporary.write_text(json.dumps(receipt,indent=2)+'\\n')\ntemporary.replace(target)\n"
        + "print(json.dumps({'acknowledged':receipt['run_id']}))")
    download.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'/mnt/beegfsstudents/home/3160552/capped_depth_delay_31976_\d+', str(args.root)):
        raise ValueError('Unexpected cluster run root')
    args.report.mkdir(parents=True, exist_ok=True)
    with (args.report / 'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(args.report / 'collector_process.json', dict(pid=os.getpid(), root=str(args.root), status='running'))
        while True:
            try:
                state = snapshot(args.root)
                write_json(args.report / 'live_status.json', state)
                if state.get('ready.json'):
                    receipt = receive(args.root, args.report, state['ready.json'])
                    print(json.dumps({'verified_transfer': receipt['run_id']}), flush=True)
                summary = state.get('summary.json')
                if summary and summary['status'] in ('complete', 'failed'):
                    write_json(args.report / 'summary.json', summary)
                    count = len(list((args.report / 'transfer_receipts').glob('*.json')))
                    if summary['status'] == 'complete' and count != 44:
                        raise ValueError('Completed allocation has missing verified transfers')
                    write_json(args.report / 'collector_process.json', dict(pid=os.getpid(), root=str(args.root),
                        status=summary['status'], verified_transfers=count))
                    print(json.dumps({'collection_complete': summary['status'], 'verified_transfers': count}), flush=True)
                    return
            except (subprocess.SubprocessError, OSError) as error:
                write_json(args.report / 'collector_last_connection_error.json', dict(error=repr(error), at_unix=time.time()))
                print(json.dumps({'transfer_retry': repr(error)}), flush=True)
            time.sleep(20)


if __name__ == '__main__':
    main()
