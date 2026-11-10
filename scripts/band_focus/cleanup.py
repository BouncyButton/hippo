"""Retire only verified lower-ranked weights from completed job 690825."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/beegfsstudents/home/3160552/band_chamfer_fold0_seed0_5pct_20261008_01')
AUDIT=Path('/mnt/beegfsstudents/home/3160552/band_focus_cleanup_20261011')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def inactive():
    if os.getuid()!=1395 or ROOT.is_symlink():raise RuntimeError('Unexpected account/root')
    state=json.loads((ROOT/'status.json').read_text())
    if state['state']!='complete' or state['completed_runs']!=64:raise RuntimeError('Campaign is not complete')
    dirs=subprocess.check_output(['squeue','-u','3160552','-h','-o','%Z'],text=True).splitlines()
    if any(str(ROOT) in directory for directory in dirs):raise RuntimeError('Campaign has an active allocation')


def plan():
    inactive();AUDIT.mkdir(exist_ok=True)
    if (AUDIT/'plan.json').exists():
        print(json.dumps(dict(plan_sha256=sha(AUDIT/'plan.json'),existing=True)));return
    result=json.loads((ROOT/'results.json').read_text());p=json.loads((ROOT/'protocol.json').read_text())
    rows=result['rows'];assert len(rows)==64
    keep={'bands','pair13','pair_independent13'}
    keep.update(x['objective'] for x in sorted(rows,key=lambda x:-x['macro_dice'])[:5])
    for metric in ('surface_distance_mean_mm','ap_best_plane_error_mm','ap_interface_chamfer_mm','ap_interface_hd95_mm'):
        keep.add(min(rows,key=lambda x:x[metric])['objective'])
    files=[];verified=0
    for entry in p['runs']:
        folder=ROOT/'runs'/entry['id'];record=json.loads((folder/'completion.json').read_text())
        expected=dict(protocol_sha256=p['identity_sha'],run=entry,**p['splits'][entry['fraction']])
        assert record['identity']==expected
        for name,digest in record['artifacts'].items():
            assert sha(folder/name)==digest,(entry['id'],name)
            verified+=1
        if entry['id'] not in keep:
            file=folder/'best.pt'
            files.append(dict(path=str(file),run=entry['id'],sha256=record['artifacts']['best.pt'],bytes=file.stat().st_size))
    aggregate=dict(protocol_identity=p['identity_sha'],rows=rows,completed=64,
                   note='Aggregate results only; original metrics/predictions/configuration remain on cluster.')
    write(AUDIT/'aggregate_results.json',aggregate)
    value=dict(root=str(ROOT),created_unix=time.time(),reason='User authorized freeing space for the single band-focus follow-up job.',
               keep=sorted(keep),files=files,bytes=sum(x['bytes'] for x in files),artifact_hashes_verified=verified,
               aggregate_sha256=sha(AUDIT/'aggregate_results.json'),preserve='Every prediction, metric, history, calibration, source, protocol, completion provenance and shared warmup; retained control/top/surface/A-P models.')
    write(AUDIT/'plan.json',value)
    print(json.dumps(dict(plan_sha256=sha(AUDIT/'plan.json'),checkpoints=len(files),logical_gib=value['bytes']/2**30,keep=sorted(keep))))


def apply(expected):
    inactive();plan_path=AUDIT/'plan.json'
    assert sha(plan_path)==expected
    plan=json.loads(plan_path.read_text());local=json.loads((AUDIT/'local_verified.json').read_text())
    assert local['plan_sha256']==expected and local['aggregate_sha256']==plan['aggregate_sha256']
    assert sha(AUDIT/'aggregate_results.json')==plan['aggregate_sha256']
    for item in plan['files']:
        path=Path(item['path']);folder=path.parent
        assert folder.parent==ROOT/'runs' and folder.name==item['run'] and path.name=='best.pt'
        assert not path.is_symlink() and not folder.is_symlink()
        if path.exists():assert sha(path)==item['sha256']
        else:assert (folder/'checkpoint_retired.json').exists()
    retired=ROOT/'retired_checkpoint_records';retired.mkdir(exist_ok=True)
    for item in plan['files']:
        path=Path(item['path']);folder=path.parent;completion=folder/'completion.json'
        record=json.loads(completion.read_text());before=retired/(item['run']+'.completion.json')
        if not before.exists():write(before,record)
        artifact={'best.pt':item['sha256']}
        write(folder/'checkpoint_retired.json',dict(identity=record['identity'],artifacts=artifact,
            reason='Completed screen; verified results preserved and model outside retained controls/top/surface/A-P references',
            cleanup_plan_sha256=expected,unix=time.time()))
        if 'best.pt' in record['artifacts']:
            assert record['artifacts'].pop('best.pt')==item['sha256']
            record['retired_artifacts']=artifact;write(completion,record)
        path.unlink(missing_ok=True)
    assert all(not Path(x['path']).exists() for x in plan['files'])
    receipt=dict(completed=True,unix=time.time(),plan_sha256=expected,deleted_checkpoints=len(plan['files']),
                 deleted_logical_bytes=plan['bytes'],retained=plan['keep'],preservation=plan['preserve'])
    write(AUDIT/'receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('plan','apply'));parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.mode=='plan':plan()
    else:apply(args.plan_sha256)
