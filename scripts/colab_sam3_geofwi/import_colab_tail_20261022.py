"""Reconcile the six Colab-owned tasks after cutoff, preserving trained updates.

Run on the cluster with an audited Drive-state snapshot and any recovery/progress
ZIPs named by their SHA256 under incoming/snapshots/. Keep originals in local
reports before transferring them. The helper never contacts or writes Drive.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile


def recovery_gap(state, name, incoming, digest):
    """Quarantine acknowledged work missing from the available checkpoint."""
    record = state['objects'].get('recovery/'+name)
    charge = state['charges'].get(name)
    if record is None or charge is None:
        return None
    archive = incoming/'snapshots'/(record['local_sha256']+'.zip')
    if digest(archive) != record['local_sha256']:
        raise ValueError('Recovery archive checksum differs')
    with zipfile.ZipFile(archive) as snapshot:
        ledger = json.loads(snapshot.read('work/'+name+'/checkpoint_ledger.json'))
    if ledger == charge:
        return None
    if (ledger['protocol_sha256'] != charge['protocol_sha256']
            or ledger['step'] >= charge['step']):
        raise ValueError('Unexpected recovery ledger mismatch')
    return dict(reason='Charged updates newer than saved checkpoint; replay prohibited',
                charged_step=charge['step'], checkpoint_step=ledger['step'],
                recovery_sha256=record['local_sha256'], recovery_drive_id=record['id'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--incoming',type=Path,required=True)
    args=parser.parse_args(); stage=args.stage.resolve(); incoming=args.incoming.resolve()
    sys.path.insert(0,str(stage))
    import cluster_runner as runner
    root,rt=runner.activate(stage)
    import replication
    import replication_store as rs
    assignment=runner.read(stage/'assignment.json')
    if time.time()<assignment['deadline_epoch']:
        raise RuntimeError('Colab still owns its tasks until 15:00 Rome')
    source_path=incoming/'source_campaign_state.json'
    state=runner.read(source_path)
    if state['bundle_sha256']!=runner.BUNDLE_SHA:
        raise ValueError('Colab scientific source changed')
    done=[n for n in assignment['colab_tasks'] if n in state['completed']]
    extra=[n for n in assignment['colab_tasks'] if n not in state['completed']]
    result=dict(additional_task_ids=extra,colab_completed=done,
        assignment_sha256=runner.digest(stage/'assignment.json'),
        source_state_sha256=runner.digest(source_path),reconciled_at=time.time(),imports={},blocked_tasks={})
    handoff=stage/'colab_handoff.json'
    if handoff.exists():
        previous=runner.read(handoff)
        if previous['additional_task_ids']!=extra or previous['colab_completed']!=done:
            raise ValueError('Colab ownership changed after handoff')
        print('Handoff already installed.'); return
    if (any(state['objects'].get('recovery/'+n) for n in extra)
            and not (root/'durable/environment_binding.json').exists()):
        raise RuntimeError('Wait for the first cluster worker to establish its GPU binding before importing checkpoints. The initial 47 tasks may proceed.')
    for name in extra:
        recovery=state['objects'].get('recovery/'+name)
        progress=state['objects'].get('progress/'+name)
        charge=state['charges'].get(name)
        gap=recovery_gap(state,name,incoming,runner.digest)
        if gap:
            result['blocked_tasks'][name]=gap
            result['imports'][name]={'kind':'blocked_recovery_gap',**gap}
            continue
        if recovery is None:
            if charge is not None or progress is not None:
                raise ValueError('Unfinished task lacks a coherent recovery: '+name)
            result['imports'][name]={'kind':'not_started'}
            continue
        task=next(t for t in replication.tasks() if t['id']==name)
        scratch=incoming/name
        if scratch.exists():
            raise ValueError('Import scratch already exists; inspect before retry: '+str(scratch))
        scratch.mkdir()
        for kind,record in (('recovery',recovery),('progress',progress)):
            if record is None: continue
            archive=incoming/'snapshots'/(record['local_sha256']+'.zip')
            if runner.digest(archive)!=record['local_sha256']:
                raise ValueError('Import ZIP differs: '+kind+'/'+name)
            rs.unpack(archive,scratch)
            ledger=runner.read(scratch/'work'/name/'checkpoint_ledger.json')
            if charge is not None and ledger!=charge:
                raise ValueError('Charged updates are newer than recovery; never replay: '+name)
            if charge is None and ledger['step']!=0:
                raise ValueError('Uncharged nonzero checkpoint')
        output=scratch/'work'/name
        old_protocol=rt.read(output/'protocol.json')
        new_protocol=replication.protocol(root,task)
        strip=lambda p:{k:v for k,v in p.items() if k not in ('data','upstream')}
        if strip(old_protocol)!=strip(new_protocol):
            raise ValueError('Scientific protocol differs; only path migration is allowed')
        import torch
        checkpoint=output/'checkpoint_latest.pt'
        original_checkpoint_sha=rt.digest(checkpoint)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        original_protocol_sha=rt.digest(output/'protocol.json')
        if (payload['protocol_sha256']!=original_protocol_sha or payload['step']!=ledger['step']
                or len(payload['history'])!=payload['step']):
            raise ValueError('Checkpoint and history disagree with acknowledged charge')
        rt.write(output/'protocol.json',new_protocol)
        new_protocol_sha=rt.digest(output/'protocol.json')
        payload['protocol_sha256']=new_protocol_sha
        torch.save(payload,checkpoint)
        ledger=dict(step=payload['step'],protocol_sha256=new_protocol_sha)
        rt.write(output/'checkpoint_ledger.json',ledger)
        probe=scratch/'durable/probes'/name
        if probe.exists():
            for path in probe.glob('*.json'):
                value=rt.read(path)
                if value.get('identity',{}).get('protocol_sha256')==original_protocol_sha:
                    value['identity']['protocol_sha256']=new_protocol_sha
                    rt.write(path,value)
        # Old partial predictions and metrics remain in the archived source ZIPs.
        # Evaluation restarts on the destination GPU; trained weights/updates do not.
        for path in list(output.glob('evaluation*')):
            if path.is_file(): path.unlink()
        for filename in ('completion.json','archive_manifest.json','archive_acknowledgment.json','cleanup.json','curves.png'):
            (output/filename).unlink(missing_ok=True)
        audit=dict(kind='checkpoint_resume',step=payload['step'],
            original_checkpoint_sha256=original_checkpoint_sha,new_checkpoint_sha256=rt.digest(checkpoint),
            original_protocol_sha256=original_protocol_sha,new_protocol_sha256=new_protocol_sha,
            source_state_sha256=result['source_state_sha256'],
            policy='Only protocol paths/hash rebound. Adapter, optimizer, history, calibration and RNG preserved. Reevaluate all volumes on the cluster GPU.')
        rt.write(output/'colab_import_audit.json',audit)
        live=root/'work'/name
        if live.exists() or (stage/'task_state'/(name+'.json')).exists():
            raise ValueError('Destination task already exists; refusing overwrite')
        shutil.copytree(output,live)
        if probe.exists(): shutil.copytree(probe,root/'durable/probes'/name)
        binding=rt.read(root/'durable/environment_binding.json')
        runner.write(stage/'task_state'/(name+'.json'),dict(completed={},charges={name:ledger},
            objects={},binding=binding,bundle_sha256=runner.BUNDLE_SHA,import_audit=audit))
        result['imports'][name]=audit
    runner.write(handoff,result)
    print(json.dumps(dict(colab_completed=len(done),added_to_cluster=len(extra),
        blocked_tasks=result['blocked_tasks']),indent=2))


if __name__=='__main__': main()
