"""Copy and hash-verify scheduler logs from completed MNI batches."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE, write
from scripts.unetr_subfields.contracts import ROOT


def archive(name):
    code = '''from pathlib import Path
import hashlib,json,sys
p=Path(sys.argv[1]);state=json.loads((p/'status.json').read_text())
assert state['status'] in ('complete','paused'),state['status']
assert not list((p/'runs').rglob('*.pt')),'Checkpoints remain'
files={}
for f in [p/'status.json',*p.glob('slurm-*')]:
    if f.is_file():
        b=f.read_bytes();files[f.name]=dict(text=b.decode('utf-8'),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
print(json.dumps(files))'''
    files = json.loads(remote(['python3', '-c', code, REMOTE / name]))
    destination = ROOT / 'reports' / name / 'job_logs'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename, item in files.items():
        assert Path(filename).name == filename
        target = destination / filename
        target.write_bytes(item['text'].encode('utf-8'))
        data = target.read_bytes()
        assert len(data) == item['bytes']
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        manifest[filename] = {k: item[k] for k in ('sha256', 'bytes')}
    write(destination / 'manifest.json', manifest)
    print(json.dumps(dict(name=name, verified_files=len(files))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    for name in parser.parse_args().names:
        archive(name)
