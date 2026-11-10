"""Read-only final audit of this MNI campaign's archives, cleanup and quota."""
import json
from pathlib import Path

from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.launch_unetr_subfields import remote
from scripts.unetr_subfields.contracts import ROOT

STAGES = [
    'mni_matched_lr_dice_ce', 'mni_one_subject_diagnostic',
    'mni_nnunet_cap4_pilot', 'mni_nnunet_cap4_full',
    'mni_swin_cap4_lr3e4_pilot', 'mni_swin_foreground_squared_cap4_pilot',
    'mni_swin_balanced_dice_cap4_pilot', 'mni_swin_allclass_dice_cap4_pilot',
    'mni_swin_uniform_dice_hold_cap4_pilot', 'mni_nnunet_cap8_full',
    'mni_swin_squared_hold_cap8_pilot', 'mni_swin_squared_hold_cap8_full',
    'mni_swin_log_dice_cap8_pilot', 'mni_swin_log_dice_cap8_full',
    'mni_unetr_geo_cap8', 'mni_unet_cap8',
]


def main():
    names = [name + '_20261106' for name in STAGES]
    code = '''from pathlib import Path
import hashlib,json,subprocess,sys,time,re
root=Path(sys.argv[1]);names=json.loads(sys.argv[2]);rows=[]
for name in names:
 p=root/name;state=json.loads((p/'status.json').read_text())
 assert state['status'] in ('complete','paused'),(name,state['status'])
 files=[f for f in (p/'runs').rglob('*') if f.is_file()]
 bulky=[str(f.relative_to(p)) for f in files if f.suffix in ('.pt','.pth','.ckpt','.tmp') or f.name.endswith('.nii.gz')]
 assert not bulky,(name,bulky)
 receipts={str(f.parent.relative_to(p/'runs')):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.name=='cleanup.json'}
 rows.append(dict(name=name,status=state['status'],cleanup_receipts=receipts,bulky_files=bulky))
queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).strip()
assert not queue,queue
raw=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
used,limit=map(float,re.search(r'(\\d+(?:\\.\\d+)?)GiB/(\\d+(?:\\.\\d+)?)GiB',raw).groups())
print(json.dumps(dict(stages=rows,queue=queue,quota=dict(used_gib=used,limit_gib=limit,free_gib=limit-used,raw=raw),checked_at=time.time())))'''
    result = json.loads(remote(['python3', '-c', code, REMOTE, json.dumps(names)]))
    released = 0
    runs = 0
    for row in result['stages']:
        report = ROOT / 'reports' / row['name']
        for run, receipt_sha in row['cleanup_receipts'].items():
            destination = report / 'runs' / run
            assert digest(destination / 'cleanup.json') == receipt_sha
            receipt = read(destination / 'cleanup.json')
            assert receipt['status'] == 'complete'
            manifest = read(destination / 'archive_manifest.json')
            verify_archive(destination, manifest)
            acknowledgment = read(destination / 'archive_acknowledgment.json')
            assert acknowledgment['status'] == 'verified'
            assert acknowledgment['manifest_sha256'] == digest(destination / 'archive_manifest.json')
            assert acknowledgment['source_verified']
            assert Path(acknowledgment['durable_archive']).resolve() == destination.resolve()
            assert all(not (destination / item['name']).exists() for item in manifest['deletion_candidates'])
            released += receipt['released_file_bytes']
            runs += 1
    result.update(status='passed',verified_run_archives=runs,
                  cumulative_released_file_bytes=released,
                  audit_source_sha256=digest(Path(__file__)))
    write(ROOT / 'reports/mni_baseline_rescue_20261106/final_storage_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'stages'}))


if __name__ == '__main__':
    main()
