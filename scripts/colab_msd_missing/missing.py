"""Prepare only the fixed missing tasks and this new account's future results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from store import Cloud, atomic, digest
from recovery_driver import (accounting_report, release_lease, restore_server_results,
                             restore_work)


def prepare(package, local, roots, terminated_session=None):
    cloud=Cloud();folder=cloud.ensure_folder()
    local.mkdir(parents=True,exist_ok=True)
    cache=local/'cache';cache.mkdir(exist_ok=True)
    scope=json.loads((package/'missing_scope.json').read_text())
    allowed={t['id'] for t in scope['missing_tasks']}
    print('FIXED SCOPE:',len(allowed),'missing tasks;',len(scope['excluded_completed_tasks']),
          'prior completions excluded. No historical run folders are scanned.',flush=True)
    probe=cache/'write_probe.json';atomic(probe,dict(purpose='server durability check',nonce=uuid.uuid4().hex))
    check=cloud.upload(probe,properties={'kind':'probe'})
    cloud.download(check['id'],cache/'probe_readback.json',digest(probe))
    cloud.execute(cloud.service().files().delete(fileId=check['id']))
    print('PASSED: server write, checksum and independent readback.',flush=True)
    objects=cloud.children()
    for obj in objects:
        props=obj.get('appProperties',{})
        if props.get('task') and props['task'] not in allowed:
            raise RuntimeError('Out-of-scope task in this destination; no training started')
    leases=[o for o in objects if o.get('appProperties',{}).get('kind')=='lease']
    if leases:
        if len(leases)!=1 or terminated_session!=leases[0]['appProperties'].get('token'):
            raise RuntimeError('A controller lease is active; confirm its VM terminated before recovery')
        proof=cache/'terminated_controller.json'
        atomic(proof,dict(previous_lease=leases[0],human_confirmed_runtime_terminated=True))
        cloud.put_immutable(proof,'abandoned_session')
        release_lease(cloud,leases[0]['id'],terminated_session)
        for phase,name in [('02','.msd_controller_lock'),('20','.msd_seeds0220_controller_lock')]:
            stale=local/phase/name
            if stale.exists():
                marker=stale/'owner.json'
                if marker.exists():
                    cloud.put_immutable(marker,'abandoned_local_lease',phase=phase);marker.unlink()
                stale.rmdir()
    configs=[o for o in objects if o.get('appProperties',{}).get('kind')=='store_config']
    if len(configs)>1:raise RuntimeError('Duplicate store configuration')
    if configs:
        p=cloud.download(configs[0]['id'],cache/'store_config.json')
        state=json.loads(p.read_text())
        if state['scope_sha256']!=digest(package/'missing_scope.json'):
            raise RuntimeError('Fixed missing-task scope changed; no training started')
    else:
        lease_id=cloud.execute(cloud.service().files().generateIds(count=1,space='drive'))['ids'][0]
        state=dict(lease_id=lease_id,folder_id=folder,schema=3,
                   scope_sha256=digest(package/'missing_scope.json'))
        p=cache/'store_config.json';atomic(p,state);cloud.put_immutable(p,'store_config')
    config=dict(state,local_root=str(local),package=str(package),roots={k:str(v) for k,v in roots.items()},
                checkpoint_interval_seconds=900,storage_policy='checkpoint_15min_reserve_v3')
    atomic(local/'cloud_config.json',config)
    os.environ['HIPPO_CLOUD_CONFIG']=str(local/'cloud_config.json')
    # Exercise MORE THAN ONE resumable chunk before any optimizer work. The old
    # tiny JSON probe could not expose an HTTP-308 transport configuration bug.
    bulk_probe=cache/'resumable_save_probe.bin'
    with bulk_probe.open('wb') as stream:
        block=os.urandom(1024**2)
        for _ in range(17):stream.write(block)
    found=[o for o in objects if o.get('appProperties',{}).get('kind')=='resumable_probe']
    if len(found)>1:raise RuntimeError('Duplicate checkpoint-save probes')
    receipt=cloud.upload(bulk_probe,'RESUMABLE_SAVE_PROBE.bin',{'kind':'resumable_probe'},
                         file_id=found[0]['id'] if found else None)
    cloud.download(receipt['id'],cache/'resumable_probe_readback.bin',digest(bulk_probe))
    print('PASSED: multi-chunk checkpoint upload and independent server readback (17 MiB).',flush=True)
    for name in ('missing_scope.json','historical_reference.json','recovery_plan.json'):
        cloud.put_immutable(package/name,'scope_evidence')
    bindings={}
    for phase in ('02','20'):
        target=local/phase;target.mkdir(exist_ok=True)
        for name in ('environment_binding.json','preflight.json'):
            source=package/'bindings'/phase/name
            if (target/name).exists() and (target/name).read_bytes()!=source.read_bytes():
                raise RuntimeError('Existing environment binding differs; no state reset')
            shutil.copyfile(source,target/name)
        bindings[phase]=json.loads((target/'environment_binding.json').read_text())
    sys.path.insert(0,str(roots['02']))
    import runtime
    runtime.activate(roots['02'])
    actual=runtime.environment()
    for phase in ('02','20'):
        if bindings[phase]!=dict(bundle_sha256=digest(roots[phase]/'bundle_manifest.json'),environment=actual):
            raise RuntimeError('Restore the original A100-SXM4-80GB/software environment. No task or ledger was reset.')
    sys.path.insert(0,str(roots['02']/'source'))
    # These objects belong only to this new missing-task campaign. None of the
    # excluded 180 archives is listed, downloaded, reconstructed or reverified.
    # Finalize completed local work first: its last log and shutdown diagnostics
    # may be newer than the worker's already-safe server snapshot. No training.
    from hooks import install_sweep
    from types import SimpleNamespace
    for phase, root in roots.items():
        for task in scope['missing_tasks']:
            if (phase=='20') != (task['seed']==20):continue
            output=root/'work'/task['id']
            if not (output/'completion.json').exists():continue
            if (output/'cleanup.json').exists() and json.loads((output/'cleanup.json').read_text()).get('status')=='complete':continue
            adapter=SimpleNamespace(r=runtime,launch=lambda task:None,check_stop=lambda:False,request_stop=lambda reason:None)
            install_sweep(adapter,phase)
            adapter.r.archive_completed(root,output,local/phase,local/phase/'recovery'/task['id'],task)
            print('RECOVERED COMPLETED LOCAL RESULT:',task['id'],'without rerunning training.',flush=True)
    objects=cloud.children()
    done=restore_server_results(cloud,objects,local,cache,bindings)
    restore_work(cloud,objects,local,cache,package,roots,done)
    # Protect the NEWEST local checkpoint before restarting optimization. The
    # previous budget stop may have prevented its final upload. Never replace
    # it with an older server checkpoint when the current VM still exists.
    protect_unfinished(runtime,scope,roots,local,done)
    audit=json.loads((package/'recovery_plan.json').read_text())
    for item in audit['rerun_completed']:
        if item['id'] in done:continue
        p=cache/(item['id']+'_previous_attempt.json');atomic(p,item)
        cloud.put_immutable(p,'recovery_event',item['id'],'02')
    accounting_report(cloud,local)
    fault=local/'SAVE_BLOCKED.json'
    if fault.exists():
        cloud.put_immutable(fault,'save_failure')
        fault.rename(local/('resolved_save_failure_'+uuid.uuid4().hex+'.json'))
    budget_stop=local/'CHECKPOINT_BUDGET_STOP.json'
    if budget_stop.exists():
        cloud.put_immutable(budget_stop,'budget_stop')
        budget_stop.rename(local/('resolved_budget_stop_'+uuid.uuid4().hex+'.json'))
    print('STORAGE POLICY v3: full server checkpoints every 15 minutes; local checkpoints unchanged; clean stops force a verified save.',flush=True)
    print('READY:',len(done),'/',len(allowed),'new tasks completed;',len(allowed)-len(done),
          'remaining. Historical exclusions are never launched.',flush=True)
    print('RESULTS: https://drive.google.com/drive/folders/'+folder,flush=True)


def protect_unfinished(runtime,scope,roots,local,done):
    from hooks import install_entry
    from types import SimpleNamespace
    for task in scope['missing_tasks']:
        if task['id'] in done:
            continue
        phase='20' if task['seed']==20 else '02'
        root=roots[phase];output=root/'work'/task['id']
        if not (output/'checkpoint_latest.pt').exists() or not (output/'budget_ledger.json').exists():
            continue
        if (output/'completion.json').exists():
            raise RuntimeError('Completed local work must archive before recovery: '+task['id'])
        entry=SimpleNamespace(write=runtime.write,read=runtime.read,RecoveryWriter=runtime.RecoveryWriter)
        old={k:os.environ.get(k) for k in ('HIPPO_CLOUD_PHASE','HIPPO_CLOUD_TASK')}
        try:
            os.environ['HIPPO_CLOUD_PHASE']=phase;os.environ['HIPPO_CLOUD_TASK']=json.dumps(task)
            install_entry(entry,output,root,{})
            writer=entry.RecoveryWriter(output,local/phase/'recovery'/task['id'])
            writer.close(force=True)
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
        print('PROTECTED UNFINISHED CHECKPOINT:',task['id'],'| verified on Drive before training.',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','run','report','preflight'])
    for name in ('package','local','parent-bundle','bundle'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--terminated-session')
    a,extra=p.parse_known_args()
    roots={'02':a.parent_bundle,'20':a.bundle}
    if a.command=='prepare':
        if extra:raise ValueError('Unexpected prepare arguments')
        prepare(a.package,a.local,roots,a.terminated_session)
        return
    if a.command=='preflight':
        command=[sys.executable,'-u',str(a.package/'preflight.py'),
            '--parent-bundle',str(a.parent_bundle),'--parent-results',str(a.local/'02'),
            '--bundle',str(a.bundle),'--durable',str(a.local/'20')]
    else:
        command=[sys.executable,'-u',str(a.package/'cloud/recovery_driver.py'),a.command,
            '--package',str(a.package),'--local',str(a.local),
            '--parent-bundle',str(a.parent_bundle),'--bundle',str(a.bundle),*extra]
    # exec preserves signal handling and avoids an unresponsive wrapper process.
    os.execv(sys.executable,command)


if __name__=='__main__':main()
