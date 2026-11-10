"""Resume from verified server objects; the mounted Drive is never a write target."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

from store import Cloud, IntegrityError, atomic, digest, extract_zip, member


def pack_tree(root, destination):
    import zipfile
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                z.write(p,str(p.relative_to(root)))


def unpack_checkpoint(path, output):
    with tarfile.open(path) as tar:
        names=[]
        for m in tar.getmembers():
            member(m.name)
            if not m.isfile():
                raise ValueError('Non-file recovery archive member')
            names.append(m.name)
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate recovery members')
        tar.extractall(output,filter='data')
    manifest=json.loads((output/'snapshot_manifest.json').read_text())
    for name,sha in manifest.items():
        member(name)
        if digest(output/name)!=sha:
            raise ValueError('Recovery member checksum differs: '+name)
    import torch
    state=torch.load(str(output/'checkpoint_latest.pt'),map_location='cpu',weights_only=False,mmap=True)
    ledger=json.loads((output/'budget_ledger.json').read_text())
    if state['step']!=state['charged_attempts'] or state['charged_attempts']!=ledger['charged_attempts']:
        raise ValueError('Checkpoint and saved compute ledger differ')
    if ledger.get('calibration_attempts',0) and digest(output/'calibration.json')!=ledger['calibration_sha256']:
        raise ValueError('Checkpoint calibration differs')
    (output/'snapshot_manifest.json').unlink()
    return ledger


def recovery_event(task, charged, saved, evidence, source):
    """Lost work is an explicit charged overhead, never erased from accounting."""
    if source == 'server_store' and saved:
        for key in ('charged_attempts','training_flops','calibration_flops'):
            if key not in charged or saved.get(key,0) > charged[key]:
                raise ValueError('Checkpoint exceeds the server charge ledger: '+task+' / '+key)
    result=dict(task=task,source=source,action='resume_checkpoint' if saved else 'restart_unrecoverable_attempt',
        durable_charge=charged,restored_checkpoint_charge=saved,evidence=evidence,
        optimization_budget_per_trajectory=31976450667090,
        accounting_note='Recovery overhead is additional to the scientific trajectory cap. Unknown pre-disconnect charges remain explicitly unknown; no zero is imputed.')
    for k in ('charged_attempts','training_flops','calibration_flops'):
        a=charged.get(k) if charged else None
        b=saved.get(k,0) if saved else 0
        result['lost_'+k]=max(0,a-b) if a is not None else None
    return result


def restore_server_results(cloud, objects, local, cache, bindings):
    grouped={}
    for obj in objects:
        p=obj.get('appProperties',{})
        if p.get('kind') in ('archive','result'):
            grouped.setdefault((p['phase'],p['task']),[]).append(obj)
    done=set()
    for (phase,task), group in sorted(grouped.items()):
        complete=[x for x in group if x['appProperties']['kind']=='archive']
        candidates=complete or group
        # Multiple result commits must describe the same scientific completion.
        previous=None
        for obj in candidates:
            path=cloud.download(obj['id'],cache/(obj['id']+'.zip'),obj['appProperties']['sha256'])
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                temp=Path(tmp);extract_zip(path,temp)
                signature=digest(temp/'completion.json')
                if previous is not None and signature!=previous:
                    raise RuntimeError('Conflicting completed attempts: '+task)
                previous=signature
                from scripts.cross_dataset_campaign import storage
                manifest=json.loads((temp/'archive_manifest.json').read_text())
                if manifest['task']['id']!=task:
                    raise ValueError('Server archive identity differs')
                storage.verify_archive(temp,manifest)
                if not (temp/'cleanup.json').exists():
                    atomic(temp/'archive_acknowledgment.json',dict(status='verified',source_verified=True,
                        source_manifest_sha256=bindings[phase]['bundle_sha256'],
                        manifest_sha256=digest(temp/'archive_manifest.json'),server_commit=obj,
                        durable_archive='https://drive.google.com/file/d/'+obj['id']+'/view'))
                    atomic(temp/'cleanup.json',dict(status='complete',deleted=[],released_file_bytes=0,
                        reason='Server result was committed before prior VM ended; no local checkpoint survives in this new runtime.'))
                target=local/phase/'runs'/task
                if target.exists() and (target/'completion.json').exists() and digest(target/'completion.json')!=signature:
                    raise RuntimeError('A verified result already exists with different bytes: '+task)
                shutil.copytree(temp,target,dirs_exist_ok=True)
        done.add(task)
    return done


def restore_work(cloud, objects, local, cache, package, roots, done):
    groups={}
    for o in objects:
        p=o.get('appProperties',{})
        if p.get('kind') in ('checkpoint','ledger'):
            groups.setdefault((p['phase'],p['task']),[]).append(o)
    legacy=json.loads((package/'recovery_plan.json').read_text())
    for item in legacy['interrupted']:
        groups.setdefault(('02',item['id']),[])
    old={x['id']:x for x in legacy['interrupted']}
    for (phase,task), group in sorted(groups.items()):
        if task in done:
            continue
        root=roots[phase];output=root/'work'/task
        if (output/'budget_ledger.json').exists():
            # A transport stop may leave the charge ahead of the latest local
            # checkpoint. Keep that exact newer checkpoint, account for the
            # lost tail on the server, then satisfy the unchanged resume guard.
            from local_recovery import reconcile_local
            task_spec=next(t for t in json.loads((root/'plan.json').read_text())['tasks'] if t['id']==task)
            reconcile_local(cloud,group,output,root,local,cache,phase,task_spec,recovery_event)
            continue
        ledger_obj=[x for x in group if x['appProperties']['kind']=='ledger']
        if len(ledger_obj)>1:
            raise RuntimeError('Multiple active charge ledgers; refusing to guess: '+task)
        charged={};evidence={};saved=None
        if ledger_obj:
            f=cloud.download(ledger_obj[0]['id'],cache/(ledger_obj[0]['id']+'.json'))
            charged=json.loads(f.read_text())
            evidence={'server_ledger_metadata':ledger_obj[0]}
            atomic(output.parent/(task+'_ledger_id.json'),ledger_obj[0])
        elif task in old:
            evidence=old[task]
            charged=old[task].get('last_charge',{})
        checkpoints=sorted([x for x in group if x['appProperties']['kind']=='checkpoint'],
                           key=lambda x:x['modifiedTime'],reverse=True)
        if not checkpoints and task in old and old[task].get('available_checkpoint'):
            candidate=old[task]['available_checkpoint']
            checkpoints=[dict(id=candidate['id'],name=candidate['name'],
                             appProperties={'sha256':candidate['sha256']})]
        # Try valid snapshots newest first, without discarding any failed evidence.
        failures=[]
        for obj in checkpoints:
            try:
                archive=cloud.download(obj['id'],cache/(obj['id']+'.checkpoint'),obj['appProperties']['sha256'])
            except IntegrityError as exc:
                failures.append(str(exc));continue
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                temp=Path(tmp)
                try:
                    trial=unpack_checkpoint(archive,temp)
                    cfg=json.loads((temp/'config.json').read_text())
                    if cfg['augmentation']!='none' or cfg['budget_flops']!=31976450667090:
                        raise ValueError('Checkpoint science differs')
                    expected=json.loads((root/'plan.json').read_text())['tasks']
                    t=next(t for t in expected if t['id']==task)
                    if (cfg['model_identity'],cfg['objective'],cfg['fold'],cfg['seed'])!=(t['model'],t['arm'],t['fold'],t['seed']):
                        raise ValueError('Wrong checkpoint task')
                except Exception as exc:
                    failures.append(str(exc));continue
                if group:
                    recovery_event(task,charged,trial,evidence,'server_store')
                output.mkdir(parents=True,exist_ok=True)
                shutil.copytree(temp,output,dirs_exist_ok=True)
                saved=trial;break
        if failures:
            evidence=dict(evidence,invalid_snapshots=failures)
        event=recovery_event(task,charged,saved,evidence,'server_store' if group else 'terminated_original_VM')
        event_path=cache/(task+'_recovery_event.json');atomic(event_path,event)
        cloud.put_immutable(event_path,'recovery_event',task,phase)
        atomic(local/'recovery_events'/(task+'.json'),event)
        if saved:
            # A replay starts from a coherent checkpoint ledger; lost charged work
            # remains in the independently committed recovery event above.
            atomic(local/phase/'recovery'/task/'charged_ledger.json',saved)
            if ledger_obj:
                p=cache/(task+'_restored_charge.json');atomic(p,saved)
                cloud.upload(p,file_id=ledger_obj[0]['id'])
            print('RECOVERED CHECKPOINT',task,'at update',saved['charged_attempts'],
                  '| lost charged tail retained in accounting:',event['lost_charged_attempts'],flush=True)
        else:
            # Ledger objects belong to the failed attempt, not the fresh attempt.
            # Preserve their bytes in the committed event; a fresh ledger is created.
            (output.parent/(task+'_ledger_id.json')).unlink(missing_ok=True)
            for previous in ledger_obj:
                props=dict(previous['appProperties'],kind='retired_ledger')
                cloud.execute(cloud.service().files().update(fileId=previous['id'],body={'appProperties':props},fields='id'))
            print('RESTART REQUIRED',task,'| evidence and spent compute preserved',flush=True)


def accounting_report(cloud,local):
    """Account for failed attempts separately from the final scientific trajectory."""
    objects=cloud.children();cache=local/'accounting_cache';cache.mkdir(exist_ok=True)
    events=[];unfinished=[];complete=[];completed_ids=set()
    for phase in ('02','20'):
        for p in (local/phase/'runs').glob('*/completion.json'):
            c=json.loads(p.read_text());complete.append(c);completed_ids.add(p.parent.name)
    for obj in objects:
        props=obj.get('appProperties',{});kind=props.get('kind')
        if kind=='recovery_event' or (kind=='ledger' and props.get('task') not in completed_ids):
            path=cloud.download(obj['id'],cache/(obj['id']+'.json'))
            value=json.loads(path.read_text())
            (events if kind=='recovery_event' else unfinished).append(value)
    wasted_training=sum(e.get('prior_attempt_optimization_flops',e.get('lost_training_flops')) or 0 for e in events)
    wasted_calibration=sum(e.get('prior_attempt_calibration_flops',e.get('lost_calibration_flops')) or 0 for e in events)
    unknown=[e.get('task',e.get('id')) for e in events if
        ('lost_training_flops' in e and e['lost_training_flops'] is None)
        or e.get('evidence',{}).get('last_charge_is_lower_bound')]
    value=dict(completed_results=len(complete),
        completed_trajectory_optimization_flops=sum(c['consumed_training_flops'] for c in complete),
        completed_trajectory_calibration_flops=sum(c['calibration_flops'] for c in complete),
        unfinished_charged_optimization_flops=sum(c['training_flops'] for c in unfinished),
        unfinished_charged_calibration_flops=sum(c['calibration_flops'] for c in unfinished),
        recovery_overhead_optimization_flops_lower_bound=wasted_training,
        recovery_overhead_calibration_flops_lower_bound=wasted_calibration,
        overhead_not_fully_known_for=sorted(set(unknown)),recovery_events=events,
        trajectory_cap=31976450667090,
        note='The trajectory cap excludes separately reported calibration and failure/replay overhead. Old truncated ledgers make total spent compute a lower bound, not an exact total. Seed1 is excluded from this selected study.')
    path=local/'recovery_accounting.json';atomic(path,value);cloud.put_immutable(path,'accounting_report')
    return value


def prepare(package, local, roots, terminated_session=None):
    cloud=Cloud();folder=cloud.ensure_folder();local.mkdir(parents=True,exist_ok=True)
    cache=local/'cache';cache.mkdir(exist_ok=True)
    # Fail before starting diagnostics/training if Google will not accept saves.
    probe=cache/'write_probe.json';atomic(probe,dict(purpose='server durability check',nonce=uuid.uuid4().hex))
    check=cloud.upload(probe,properties={'kind':'probe'})
    cloud.download(check['id'],cache/'probe_readback.json',digest(probe))
    cloud.execute(cloud.service().files().delete(fileId=check['id']))
    print('PASSED: Drive server write, checksum and independent download.',flush=True)
    objects=cloud.children()
    leases=[o for o in objects if o.get('appProperties',{}).get('kind')=='lease']
    if leases:
        if len(leases)!=1 or terminated_session!=leases[0]['appProperties'].get('token'):
            raise RuntimeError('A server controller lease is present. Confirm in the notebook that its runtime terminated; never reclaim a running controller.')
        proof=cache/'confirmed_terminated_session.json'
        atomic(proof,dict(previous_lease=leases[0],human_confirmed_runtime_terminated=True))
        cloud.put_immutable(proof,'abandoned_session')
        release_lease(cloud,leases[0]['id'],leases[0]['appProperties']['token'])
        for phase,name in [('02','.msd_controller_lock'),('20','.msd_seeds0220_controller_lock')]:
            stale=local/phase/name
            if stale.exists():
                marker=stale/'owner.json'
                if marker.exists():
                    cloud.put_immutable(marker,'abandoned_local_lease',phase=phase)
                    marker.unlink()
                stale.rmdir()  # Unexpected contents block cleanup; never recursive.
    configs=[o for o in objects if o.get('appProperties',{}).get('kind')=='store_config']
    if len(configs)>1:
        raise RuntimeError('Duplicate store configurations')
    if configs:
        path=cloud.download(configs[0]['id'],cache/'store_config.json')
        state=json.loads(path.read_text())
    else:
        lease_id=cloud.execute(cloud.service().files().generateIds(count=1,space='drive'))['ids'][0]
        state=dict(lease_id=lease_id,folder_id=folder,schema=2)
        path=cache/'store_config.json';atomic(path,state);cloud.put_immutable(path,'store_config')
    config=dict(state,local_root=str(local),package=str(package),roots={k:str(v) for k,v in roots.items()})
    atomic(local/'cloud_config.json',config)
    os.environ['HIPPO_CLOUD_CONFIG']=str(local/'cloud_config.json')
    # This compact import was prepared and hash-verified by the recovery audit.
    # It replaces thousands of mounted-Drive probes with one immutable object.
    imported=package/'preserved_results.zip'
    cloud.put_immutable(imported,'imported_results')
    cloud.put_immutable(package/'recovery_evidence.zip','recovery_evidence')
    extract_zip(imported,local/'imported')
    for phase in ('02','20'):
        target=local/phase;target.mkdir(exist_ok=True)
        for name in ('environment_binding.json','preflight.json'):
            source=local/'imported'/phase/name
            if (target/name).exists() and (target/name).read_bytes()!=source.read_bytes():
                raise RuntimeError('Existing environment binding differs; preserved')
            shutil.copyfile(source,target/name)
        for src in (local/'imported'/phase/'runs').glob('*'):
            dest=target/'runs'/src.name
            if not dest.exists():shutil.copytree(src,dest)
    bindings={p:json.loads((local/p/'environment_binding.json').read_text()) for p in ('02','20')}
    sys.path.insert(0,str(roots['02']))
    import runtime
    runtime.activate(roots['02'])
    actual=runtime.environment()
    for phase in ('02','20'):
        if bindings[phase] != dict(bundle_sha256=digest(roots[phase]/'bundle_manifest.json'),environment=actual):
            raise RuntimeError('Restore the original A100-SXM4-80GB and software environment. No recovery ledger was reset.')
    # Load frozen storage validator without constructing any GPU model.
    sys.path.insert(0,str(roots['02']/'source'))
    done={p.name for p in (local/'02/runs').glob('*')}
    done |= restore_server_results(cloud,objects,local,cache,bindings)
    restore_work(cloud,objects,local,cache,package,roots,done)
    audit=json.loads((package/'recovery_plan.json').read_text())
    for item in audit['rerun_completed']:
        if item['id'] in done:continue
        p=cache/(item['id']+'_lost_archive.json');atomic(p,item)
        cloud.put_immutable(p,'recovery_event',item['id'],'02')
    accounting_report(cloud,local)
    if (local/'SAVE_BLOCKED.json').exists():
        old=local/'SAVE_BLOCKED.json'
        cloud.put_immutable(old,'save_failure')
        old.rename(local/('resolved_save_failure_'+str(time.time_ns())+'.json'))
    print('READY:',len(done),'/504 completed results retained; only missing/unrecoverable tasks will launch.',flush=True)
    print('SERVER RESULTS: https://drive.google.com/drive/folders/'+folder,flush=True)
    return config


def next_lease(cloud, first):
    """Follow released leases; never attempt to reuse a deleted Drive file ID."""
    from googleapiclient.errors import HttpError
    seen=set();current=first
    while current not in seen:
        seen.add(current)
        try:meta=cloud.metadata(current)
        except HttpError as exc:
            if exc.resp.status==404:return current
            raise
        props=meta.get('appProperties',{})
        if meta.get('parents') != [cloud.folder] or props.get('kind')!='released_lease':
            raise RuntimeError('An existing controller lease is present. Confirm its runtime terminated before reclaiming it.')
        current=props['next_lease_id']
    raise RuntimeError('Invalid controller lease chain')


def release_lease(cloud, lease_id, token):
    meta=cloud.metadata(lease_id)
    props=meta.get('appProperties',{})
    if meta.get('parents')!=[cloud.folder] or props.get('token')!=token:
        raise RuntimeError('Controller lease ownership changed; no lease was released')
    if props.get('kind')=='released_lease':return
    successor=cloud.execute(cloud.service().files().generateIds(count=1,space='drive'))['ids'][0]
    updated=dict(props,kind='released_lease',next_lease_id=successor)
    cloud.execute(cloud.service().files().update(fileId=lease_id,
        body={'name':'RELEASED_CONTROLLER','appProperties':updated},fields='id,appProperties'))


@contextmanager
def ownership(config):
    cloud=Cloud(config['folder_id']);token=uuid.uuid4().hex
    from googleapiclient.errors import HttpError
    lease_id=next_lease(cloud,config['lease_id'])
    try:
        cloud.execute(cloud.service().files().create(body={'id':lease_id,
            'name':'ACTIVE_CONTROLLER','parents':[config['folder_id']],
            'appProperties':{'kind':'lease','token':token,'host':socket.gethostname()}},fields='id'))
    except Exception:
        # A timed-out response may still have created this exact lease.
        meta=cloud.metadata(lease_id)
        if meta.get('appProperties',{}).get('token')!=token:
            raise RuntimeError('Another controller acquired the lease. Do not launch twice.')
    try:
        yield
    finally:
        release_lease(cloud,lease_id,token)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--local',type=Path,required=True)
    parser.add_argument('--parent-bundle',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=22)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--max-runs',type=int,default=504)
    parser.add_argument('--terminated-session',default=None)
    args=parser.parse_args()
    roots={'02':args.parent_bundle,'20':args.bundle}
    if args.command=='prepare':
        prepare(args.package,args.local,roots,args.terminated_session);return
    config=json.loads((args.local/'cloud_config.json').read_text())
    os.environ['HIPPO_CLOUD_CONFIG']=str(args.local/'cloud_config.json')
    os.environ['PYTHONPATH']=str(args.package/'cloud')+os.pathsep+os.environ.get('PYTHONPATH','')
    common=['--parent-bundle',str(args.parent_bundle),'--parent-results',str(args.local/'02'),
            '--bundle',str(args.bundle),'--durable',str(args.local/'20')]
    command=[sys.executable,'-u',str(args.package/'campaign.py'),args.command,*common,
             '--hours',str(args.hours),'--workers',str(args.workers),'--max-runs',str(args.max_runs)]
    with ownership(config):
        process=subprocess.Popen(command)
        import signal
        def stop(signum,frame):
            if process.poll() is None:process.send_signal(signal.SIGINT)
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        code=process.wait()
        # Preserve the two phase snapshots even when a clean stop occurs mid-sweep.
        cloud=Cloud(config['folder_id'])
        for name in ('retained_seeds02.json','seed20_snapshot.json','comparison.json'):
            p=args.local/'20'/name
            if p.exists():cloud.put_immutable(p,'report')
        accounting_report(cloud,args.local)
        if code:raise RuntimeError('Campaign stopped with an error; server archives and local recovery retained')


if __name__=='__main__':main()
