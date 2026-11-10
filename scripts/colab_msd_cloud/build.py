"""Build the audited recovery package without changing frozen scientific files."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).parent))
from store import atomic, digest
from recover import checked_bytes, recover_one, science_check

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/msd_colab_seeds0220_recovery_20261018'
OLD=ROOT/'reports/msd_colab_seeds0220_noaug_20261018'
PARENT=ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015/bundle'
SOURCE=Path(__file__).parent
ZIP='msd_seeds0220_server_verified_resume.zip'
BOOK='MSD_SEEDS_0_2_20_SERVER_VERIFIED_RESUME.ipynb'


def read(path):return json.loads(path.read_text())


def pack(root,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
                z.write(p,str(p.relative_to(root)))


def audit():
    backup=OUT/'cloud_backup';ready=OUT/'preserved'
    # Generated staging may be regenerated; original downloaded bytes are immutable inputs.
    if ready.exists():shutil.rmtree(ready)
    ready.mkdir()
    for phase in ('02','20'):
        (ready/phase/'runs').mkdir(parents=True)
        for a,b in [('binding','environment_binding'),('preflight','preflight')]:
            shutil.copyfile(backup/(a+phase+'.json'),ready/phase/(b+'.json'))
    inv=read(OLD/'cloud_inventory_20261018.json')
    rows={r['id']:r for r in read(backup/'retained_seeds02.json')['runs']}
    tasks={t['id']:t for t in read(PARENT/'plan.json')['tasks']}
    reference=read(PARENT/'initialization_reference.json');binding=read(backup/'binding02.json')
    blobs={str(p.relative_to(backup/'runs')):p.read_bytes() for p in (backup/'runs').rglob('*') if p.is_file()}
    records=[]
    for r in inv['records']:
        source=backup/'runs'/r['id'];task=tasks[r['id']];dest=ready/'02/runs'/r['id']
        if {p.name for p in source.iterdir()}!={f['name'] for f in r['files']}:
            raise RuntimeError('Download inventory incomplete; do not classify as lost: '+r['id'])
        local={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
        record=dict(id=r['id'],source_folder_id=r['folder_id'],reported_result=rows[r['id']])
        try:
            m=json.loads(local['archive_manifest.json'])
            if m['task']!=task:raise ValueError('Task differs')
            for name,info in m['metadata'].items():checked_bytes(name,local[name],info)
            ack=json.loads(local['archive_acknowledgment.json'])
            if ack['manifest_sha256']!=digest(source/'archive_manifest.json') or ack['source_manifest_sha256']!=binding['bundle_sha256'] or not ack['source_verified'] or ack['status']!='verified':
                raise ValueError('Acknowledgment differs')
            if json.loads(local['cleanup.json'])['status']!='complete':raise ValueError('Cleanup incomplete')
            science_check(local,task,rows[r['id']],reference)
            shutil.copytree(source,dest)
            record['action']='preserve_verified'
        except (KeyError,ValueError,FileNotFoundError) as error:
            try:
                proof=recover_one(source,dest,task,rows[r['id']],reference,binding,blobs,
                                  digest(PARENT/'protocols'/task['protocol']/'protocol.json'))
                record.update(action='preserve_recovered',recovery=proof)
            except (KeyError,ValueError,FileNotFoundError) as exc:
                record.update(action='rerun_unrecoverable',reason=str(exc),
                    prior_attempt_optimization_flops=rows[r['id']]['training_flops'],
                    prior_attempt_calibration_flops=rows[r['id']]['calibration_flops'],
                    prior_attempt_updates=rows[r['id']]['updates'],
                    original_artifacts_preserved=True,
                    accounting_note='Prior completed attempt cost remains additional overhead; fresh trajectory has the original cap.')
        records.append(record)
    old_audit=read(OLD/'disconnect_audit_20261018.json');interrupted=[]
    observed={x['id']:x.get('ledger',{}) for x in read(backup/'status.json')['active']}
    for r in old_audit['recovery']:
        item=dict(id=r['id'],original_evidence=r,last_charge=observed.get(r['id'],{}),
                  last_charge_is_lower_bound=True)
        for k,v in r.items():
            if isinstance(v,dict) and 'training_flops' in v and 'charged_attempts' in v:
                item['last_charge']=v
        files=[f for f in r['files'] if f['name'].endswith('.tar')]
        if files:
            f=files[0];item['available_checkpoint']=dict(id=f['drive_id'],name=f['name'],sha256=f['name'][9:-4])
        item['known_minimum_charged_updates']=r.get('latest',{}).get('charged_attempts')
        interrupted.append(item)
    counts=dict(Counter(r['action'] for r in records))
    report=dict(status='cloud_bytes_audited',counts=counts,records=records,
        retained=sum(counts.get(k,0) for k in ('preserve_verified','preserve_recovered')),
        rerun_completed=[r for r in records if r['action']=='rerun_unrecoverable'],
        interrupted=interrupted,legacy_runtime_terminated_confirmed=True,
        augmentation='none',seeds=[0,2,20],folds=[0,1,2],total=504,
        unknown_original_charges='Some old Drive ledgers are empty; exact lost-attempt cost cannot be reconstructed. Never treat it as zero.')
    atomic(OUT/'recovery_plan.json',report)
    return report


def build():
    report=audit();package=OUT/'package'
    if package.exists():shutil.rmtree(package)
    shutil.copytree(OLD/'package',package,ignore=shutil.ignore_patterns('__pycache__'),symlinks=True)
    cloud=package/'cloud';cloud.mkdir()
    for name in ('store.py','hooks.py','driver.py','recover.py','local_recovery.py'):
        shutil.copyfile(SOURCE/name,cloud/name)
    for phase in ('02','20'):
        control=package/('control'+phase)
        p=control/'worker.py';s=p.read_text()
        old='    sys.argv = [str(args.bundle/\'entry.py\'), *trainer_args]'
        assert s.count(old)==1
        s=s.replace(old,"    sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'cloud'))\n    from hooks import install_entry\n    install_entry(entry, output, args.bundle, stats)\n"+old)
        p.write_text(s)
        m=read(control/'control_manifest.json');m['files']['worker.py']=digest(p)
        m['server_transport']='server_verified_v2';atomic(control/'control_manifest.json',m)
    p=package/'phase.py';s=p.read_text();needle='    verify_original = sweep.r.verify_bundle'
    assert s.count(needle)==1
    s=s.replace(needle,"    from hooks import install_sweep\n    install_sweep(sweep, '02' if args.seeds == [0, 2] else '20')\n"+needle);p.write_text(s)
    shutil.copyfile(OUT/'recovery_plan.json',package/'recovery_plan.json')
    pack(OUT/'preserved',package/'preserved_results.zip')
    pack(OUT/'cloud_backup',package/'recovery_evidence.zip')
    files={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*'))
           if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name!='package_manifest.json'}
    atomic(package/'package_manifest.json',files)
    pack(package,OUT/ZIP)
    from notebook import make_notebook
    book=make_notebook(digest(OUT/ZIP),digest(package/'package_manifest.json'),report)
    for c in book['cells']:
        if c['cell_type']=='code':ast.parse(c['source'])
    atomic(OUT/BOOK,book)
    atomic(OUT/'export_summary.json',dict(notebook=BOOK,zip=ZIP,sha256=digest(OUT/ZIP),
        preserved=report['retained'],rerun_completed=len(report['rerun_completed']),
        interrupted=len(report['interrupted']),gpu_tested=False,scientific_source_changed=False,
        no_augmentation=True,server_commit_before_cleanup=True))
    print(json.dumps(read(OUT/'export_summary.json'),indent=2))


if __name__=='__main__':build()
