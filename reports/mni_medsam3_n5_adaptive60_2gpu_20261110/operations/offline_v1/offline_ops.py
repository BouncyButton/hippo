"""Cluster-owned operation of the frozen MedSAM3 study; no laptop gates.

Scientific training/selection code stays frozen. Completed milestones can unlock
continuation after local validation. Only a real off-BeeGFS archive acknowledgment
permits checkpoint retirement; sleeping laptops never produce synthetic acks.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts.cross_dataset_campaign import controller, adni_foundation_full_pool as pool
from scripts.cross_dataset_campaign.common import read, write, digest, stopping
from scripts.cross_dataset_campaign.storage import prepare, quota, verify_archive, retire
from scripts.mni_medsam3_cluster import adaptive_campaign as study
from scripts.mni_medsam3_cluster.adaptive_policy import choose_seeds, next_trials

OPS = Path(__file__).resolve().parent
VALIDATED = {}


def verify(stage):
    plan = study.verify(stage)
    for name, sha in read(OPS/'manifest.json').items():
        if digest(OPS/name) != sha:
            raise ValueError('Operational source changed: '+name)
    config = read(OPS/'config.json')
    if str(stage) != config['stage'] or digest(stage/'manifest.json') != config['scientific_manifest_sha256']:
        raise ValueError('Operations belong to a different frozen study')
    return plan


def validate_milestone(stage, task):
    out = stage/'runs'/task['id']
    manifest_path = out/'archive_manifest.json'
    if not manifest_path.exists():
        return None
    manifest_sha = digest(manifest_path)
    key = (str(out), manifest_sha)
    if key in VALIDATED:
        return VALIDATED[key]
    manifest = read(manifest_path)
    verify_archive(out, manifest)
    done = read(out/'completion.json')
    if (done['status'] != 'complete' or done['updates'] != task['updates']
            or digest(out/'completion.json') != manifest['completion_sha256']
            or digest(out/'evaluation.json') != done['evaluation_sha256']
            or digest(out/'protocol.json') != done['protocol_sha256']):
        raise ValueError('Completed milestone integrity failed')
    row = dict(**task, validation_dice=done['validation_dice'], training_dice=done['training_dice'],
        observed_at=read(out/'milestone_observed.json')['time'])
    write(stage/'offline_state/verified'/f"{task['id']}.json", dict(status='locally_verified',
        archive_manifest_sha256=manifest_sha, completion_sha256=digest(out/'completion.json'),
        off_beegfs_archive=False, verified_at=time.time()))
    VALIDATED[key] = row
    return row


def records(stage, plan):
    return [r for t in plan['tasks'] if (r := validate_milestone(stage, t)) is not None]


def selected_seeds(stage, rows, plan):
    path = stage/'selected_seeds.json'
    if path.exists():
        return read(path)['seeds']
    selected = choose_seeds(rows, plan['threshold'])
    if selected:
        write(path, dict(seeds=selected, threshold=plan['threshold'], selected_at=time.time(),
            method='First two seeds whose integrity-verified fold0 milestone reaches60%; archival is independent.',
            evidence=[r for r in rows if r['fold']==0 and r['seed'] in selected and r['validation_dice']>=plan['threshold']]))
    return selected


def summarize(stage, plan, status, active=None):
    rows = records(stage, plan)
    selected = selected_seeds(stage, rows, plan)
    pending_archive = [r['id'] for r in rows if not (stage/'runs'/r['id']/'archive_acknowledgment.json').exists()]
    pending_cleanup = [r['id'] for r in rows if not (stage/'runs'/r['id']/'cleanup.json').exists()
        or read(stage/'runs'/r['id']/'cleanup.json')['status'] != 'complete']
    outcome = None
    if status in ('complete', 'complete_pending_archive'):
        passed = {(r['fold'],r['seed']) for r in rows if r['validation_dice']>=plan['threshold']}
        outcome = ('fewer_than_two_seeds_reached60_at_or_below1000' if not selected else
            'two_seeds_qualified_on_all_three_folds' if all((f,s) in passed for f in (1,2) for s in selected)
            else 'some_validation_folds_below60_at1000')
        status = 'complete_pending_archive' if pending_archive or pending_cleanup else 'complete'
    write(stage/'status.json', dict(status=status, active=active or [], completed_milestones=len(rows),
        selected_seeds=selected, outcome=outcome, threshold=plan['threshold'], maximum_updates=1000,
        runs=rows, job_id=os.environ.get('SLURM_JOB_ID'), updated_at=time.time(), quota=quota(),
        pending_durable_archive=pending_archive, pending_cleanup=pending_cleanup,
        operational_manifest_sha256=digest(OPS/'manifest.json'),
        offline_policy='Training, selection, model delivery and continuation are cluster-owned. Retain artifacts until a real verified off-BeeGFS archive returns.'))


def prepare_completed(stage, task):
    out = stage/'runs'/task['id']
    if not (out/'milestone_observed.json').exists():
        write(out/'milestone_observed.json', dict(time=time.time()))
    if not (out/'archive_manifest.json').exists():
        write(out/'operations.json', dict(operational_manifest_sha256=digest(OPS/'manifest.json'),
            scientific_manifest_sha256=digest(stage/'manifest.json'), laptop_required=False))
    prepare(out, task)
    validate_milestone(stage, task)


def retire_archived(stage, plan, rows, selected, active_ids):
    # Missing archival acknowledgments are never treated as permission to delete.
    archived = [r for r in rows if (stage/'runs'/r['id']/'archive_acknowledgment.json').exists()]
    study.retire_ready(stage, plan, archived, selected, active_ids)


def dispatch(stage, plan, base_path, publish):
    workers = {}
    reason = failure = None
    try:
        while True:
            changed = False
            if stopping() and reason is None:
                reason = 'paused' if read(stage/'control.json').get('pause') else 'continuing_at_time_limit'
            if reason:
                pool.signal_workers(workers)
            for lane, w in list(workers.items()):
                code = w['process'].poll()
                if code is None:
                    continue
                w['stream'].close(); out=w['output']
                if code:
                    marker=out/'requested_stop.json'
                    if not (reason and marker.exists() and read(marker)['checkpoint_sha256']==digest(out/'checkpoint_latest.pt')):
                        failure=f"{w['task']['id']}: worker exited{code}";reason='failed'
                else:
                    if not (out/'completion.json').exists():
                        raise RuntimeError('Worker returned without complete evaluation')
                    prepare_completed(stage,w['task'])
                del workers[lane];changed=True
            rows=records(stage,plan);selected=selected_seeds(stage,rows,plan)
            retire_archived(stage,plan,rows,selected,{w['task']['id'] for w in workers.values()})
            if reason:
                pool.signal_workers(workers)
                if not workers:
                    publish(reason,{})
                    if failure:raise RuntimeError(failure)
                    return reason
            else:
                for lane in range(2):
                    if lane in workers:continue
                    ready=next_trials(plan,rows,selected,{w['task']['id'] for w in workers.values()})
                    if not ready:continue
                    task=ready[0];out=stage/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
                    if (out/'completion.json').exists():
                        prepare_completed(stage,task);rows=records(stage,plan)
                        selected=selected_seeds(stage,rows,plan);changed=True
                        continue
                    if quota()['free_gib']<plan['storage']['minimum_free_gib']:
                        reason='waiting_for_quota';changed=True;break
                    (out/'requested_stop.json').unlink(missing_ok=True)
                    stream=(out/'train.log').open('a')
                    try:process=subprocess.Popen(study.command(stage,plan,task,base_path,lane),cwd=stage/'source',stdout=stream,stderr=subprocess.STDOUT)
                    except Exception:stream.close();raise
                    workers[lane]=dict(task=task,output=out,phase='running',process=process,stream=stream);changed=True
                if not workers and reason is None and not next_trials(plan,rows,selected,set()):
                    # Decide/clean again after any recovered completion above.
                    rows=records(stage,plan);selected=selected_seeds(stage,rows,plan)
                    retire_archived(stage,plan,rows,selected,set())
                    publish('complete',{})
                    return 'complete'
            if changed:publish(reason or 'running',workers)
            time.sleep(2)
    finally:
        pool.signal_workers(workers)
        for w in workers.values():
            if w['process'].poll() is None:
                try:w['process'].wait(timeout=300)
                except subprocess.TimeoutExpired:w['process'].terminate();w['process'].wait(timeout=30)
            w['stream'].close()


def run(stage):
    plan=verify(stage)
    controller.summarize=summarize
    pool.dispatch=dispatch
    pool.gpu_preflight=lambda stage,plan,base,publish:True
    pool.run(stage,plan,lambda:None,lambda:None)


def preflight(stage):
    plan=verify(stage);config=read(OPS/'config.json')
    if digest(Path(config['base_source']))!=plan['base_sha256']:
        raise ValueError('Persistent pretrained model changed')
    usage=quota()
    if usage['free_gib']<config['offline_peak_gib']+plan['storage']['minimum_free_gib']:
        raise RuntimeError('Insufficient headroom for worst-case offline retention')
    subprocess.run(['bash','-n',str(OPS/'run.sbatch')],check=True)
    check=subprocess.run(['sbatch','--test-only',str(OPS/'run.sbatch'),str(stage)],capture_output=True,text=True,check=True)
    write(OPS/'preflight.json',dict(status='passed',operational_manifest_sha256=digest(OPS/'manifest.json'),
        scientific_manifest_sha256=digest(stage/'manifest.json'),quota=usage,
        offline_peak_gib=config['offline_peak_gib'],scheduler=check.stdout+check.stderr,time=time.time()))


def submit(stage, plan):
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        control=read(stage/'control.json')
        if control.get('pause') or control.get('controller_owner')!='offline_v1':
            raise RuntimeError('Offline controller is not enabled')
        current=controller.queue();own={p.stem for p in (stage/'submissions').glob('*.json')}
        if any(r[0] in own for r in current):return None
        if len(current)>1 or any(r[1] not in ('RUNNING','COMPLETING','CONFIGURING') for r in current):return None
        if quota()['free_gib']<plan['storage']['minimum_free_gib']:return None
        receipt=read(OPS/'preflight.json')
        if receipt['status']!='passed' or receipt['operational_manifest_sha256']!=digest(OPS/'manifest.json'):
            raise ValueError('No matching operational preflight')
        command=['sbatch','--parsable','--chdir='+str(stage),'--output='+str(stage/'slurm-%j.out'),'--error='+str(stage/'slurm-%j.err')]
        dependency=current[0][0] if current else None
        if dependency:command+=['--dependency=afterany:'+dependency]
        command += [str(OPS/'run.sbatch'),str(stage)]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not job.isdigit():raise ValueError('Unexpected Slurm job id')
        record=dict(job_id=job,dependency=dependency,command=command,submitted_at=time.time(),
            manifest_sha256=digest(stage/'manifest.json'),operational_manifest_sha256=digest(OPS/'manifest.json'),laptop_required=False)
        write(stage/'submissions'/f'{job}.json',record);write(stage/'submission.json',record)
        current=controller.queue()
        if len(current)>2 and any(r[0]==job and r[1]=='PENDING' for r in current):
            subprocess.run(['scancel',job],check=True)
            write(stage/'offline_state/submission_race.json',record)
        return record


def deliver_base(stage, plan, queue, verified_source):
    request=stage/'base_request.json'
    if not request.exists():return
    r=read(request);job=str(r['job_id'])
    own={p.stem for p in (stage/'submissions').glob('*.json')}
    if job not in own or not any(x[0]==job and x[1]=='RUNNING' for x in queue):return
    target=f'/tmp/hippo-sam3-{os.getuid()}-{job}.pt'
    if not job.isdigit() or r['destination']!=target or r['expected_sha256']!=plan['base_sha256']:
        raise ValueError('Unexpected model delivery request')
    marker=stage/f'base_ready_{job}.json'
    if marker.exists() and read(marker).get('sha256')==plan['base_sha256']:return
    subprocess.run(['sbcast','--jobid='+job,'--force',str(verified_source),target],check=True,timeout=240)
    write(marker,dict(job_id=job,sha256=plan['base_sha256'],delivered_at=time.time(),
        delivered_by='cluster_side_service',source=str(verified_source)))


def serve(stage):
    plan=verify(stage);config=read(OPS/'config.json')
    source=Path(config['base_source'])
    if digest(source)!=plan['base_sha256']:raise ValueError('Persistent base changed')
    with (stage/'offline_service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Login nodes cannot access old compute-node scratch. The archive gate
        # still verifies and retires shared artifacts with the original code.
        study.base.retire=retire
        while True:
            try:
                control=read(stage/'control.json')
                if control.get('pause') or control.get('controller_owner')!='offline_v1':return
                current=controller.queue()
                own={p.stem for p in (stage/'submissions').glob('*.json')}
                active=[r for r in current if r[0] in own]
                deliver_base(stage,plan,current,source)
                state=read(stage/'status.json') if (stage/'status.json').exists() else {}
                if not active:
                    rows=records(stage,plan);selected=selected_seeds(stage,rows,plan)
                    retire_archived(stage,plan,rows,selected,set())
                    if state.get('status') in ('complete','complete_pending_archive'):
                        summarize(stage,plan,'complete',[])
                        if read(stage/'status.json')['status']=='complete':return
                    elif state.get('status') in ('failed','paused'):
                        return
                    else:
                        submit(stage,plan)
                write(stage/'offline_service.json',dict(status='running',pid=os.getpid(),
                    active_jobs=active,last_check=time.time(),laptop_required=False,
                    operational_manifest_sha256=digest(OPS/'manifest.json')))
            except Exception:
                write(stage/'offline_service_error.json',dict(time=time.time(),traceback=traceback.format_exc()))
            time.sleep(15)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','run','serve','verify'])
    parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args()
    globals()[args.command](args.stage.resolve())


if __name__=='__main__':main()
