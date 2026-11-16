"""Resume fixed conventional recipes after VM loss; preserve all prior evidence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.colab_hfh.bands_campaign import read, write, sha, extract_archive


def verify_restart(work, current, prior, payload):
    """Permit only an explicitly audited, separately accounted infrastructure retry."""
    audit = read(work/'vm_loss_restart_audit.json')
    if (audit['run'] != current or audit['prior_status_sha256'] != sha(prior)
            or audit['payload_sha256'] != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            or audit['reason'] != 'Colab VM lost; no durable checkpoint or ledger'
            or audit['new_attempt'] != 2
            or audit['model'] not in ('basic_unet16', 'unetr_vit_b16', 'swinunetr')
            or not current.startswith(audit['model']+'_f')
            or not audit['prior_status_archived_and_verified']):
        raise ValueError('Invalid VM-loss restart audit')
    target = work/'runs'/current
    if (target/'budget_ledger.json').exists():
        raise ValueError('Resume the existing ledger instead of restarting')
    target.mkdir(parents=True, exist_ok=True)
    write(target/'vm_loss_restart_audit.json', audit)


def run(root, hours=10):
    from scripts.colab_hfh.constraint_drive import Drive
    from scripts.colab_hfh.complete_selected_matrix import prepare, MODELS
    from scripts.colab_hfh.constraint_schedule import make_payload
    from scripts.colab_hfh.constraint_report import write_report
    from scripts.colab_hfh.runtime import verify_archive
    source = root/'bundle/source/scripts/colab_hfh'
    plan, base = read(root/'constraint_matrix_plan.json'), read(root/'bands_v1_payload.json')
    for name, expected in plan['calibration_implementation']['source_sha256'].items():
        if sha(source/name) != expected:
            raise ValueError('Frozen scientific implementation differs: '+name)
    # Only the transport/recovery guard changes; trainer, losses and recipes do not.
    path = source/'bands_campaign.py'
    old = "raise RuntimeError('Unfinished charged BAND attempt on previous VM requires recovery audit: '+current)"
    new = "__import__('scripts.colab_hfh.three_model_recovery', fromlist=['verify_restart']).verify_restart(work,current,prior,payload)"
    text = path.read_text()
    if new not in text:
        if text.count(old) != 1: raise ValueError('Recovery integration anchor differs')
        backup = root/'bands_campaign_before_vm_recovery.py'
        if not backup.exists(): shutil.copyfile(path, backup)
        path.write_text(text.replace(old, new))
    work = root/'constraint_search'; work.mkdir(exist_ok=True)
    drive = Drive('results_constraint_search_v1')
    state_path = work/'search_status.json'
    if not state_path.exists():
        drive.call('copyto', drive.path('search_status.json'), str(state_path), '--checksum')
    original = read(state_path)
    write(work/'before_vm_recovery_search_status.json', original)
    state, queue = prepare(original, plan, base)
    state['fixed_completion_scope'] = dict(models=list(MODELS), excluded_models=['nnunet'],
        policy='All selected pairs, including negative results; independent of nnU-Net outcome')
    deadline = time.time()+hours*3600
    failures = []
    with (root/'bands_recipe_search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in queue:
            if item['complete']: continue
            if time.time() >= deadline: break
            name, model, arm = item['recipe'], item['model'], item['objective']
            trial = work/name; trial.mkdir(exist_ok=True)
            record = state['attempts'][name]
            payload = make_payload(base, model, arm, plan['cost_references'][model]['canonical_costs'],
                                   **record['specification'], screening=False)
            payload_path = work/(name+'_payload.json'); write(payload_path, payload)
            try:
                remote = Drive('results_constraints_'+name)
                downloads = trial/'remote_archives'; downloads.mkdir(exist_ok=True)
                remote.call('copy', remote.folder, str(downloads), '--include', '*.zip',
                            '--include', 'campaign_status.json', '--include', 'vm_loss_restart_audit.json', '--checksum')
                for zipped in downloads.glob('*.zip'):
                    if zipped.stem not in payload['references']: raise ValueError('Unexpected run archive')
                    durable = trial/'archives'/zipped.stem
                    extract_archive(zipped, durable); verify_archive(durable)
                prior = downloads/'campaign_status.json'
                if prior.exists():
                    saved = read(prior); current = saved.get('current')
                    if (current and saved['status'] in ('running','runtime_error')
                            and not (trial/'archives'/current/'cleanup.json').exists()
                            and not (trial/'runs'/current/'budget_ledger.json').exists()):
                        # This is a separate full-budget replacement, never a claim
                        # that unknown earlier work consumed zero FLOPs.
                        if name != 'basic_unet16__dice_bands_edge__r03' or current != 'basic_unet16_f2_s0':
                            raise RuntimeError('Unexpected lost run requires a separate audit: '+str(current))
                        if (downloads/'vm_loss_restart_audit.json').exists():
                            raise RuntimeError('An infrastructure replacement was already attempted; preserve it')
                        inventory = json.loads(remote.call('lsjson', remote.folder, '--recursive'))
                        if any(r['Path'].endswith(('.pt','.pth')) or 'recovery/' in r['Path'] for r in inventory):
                            raise RuntimeError('Remote recovery artifacts exist; restore them instead of restarting')
                        preserved = trial/('vm_loss_prior_status_'+sha(prior)+'.json')
                        shutil.copyfile(prior, preserved); remote.upload(preserved)
                        audit = dict(reason='Colab VM lost; no durable checkpoint or ledger',
                            run=current, recipe=name, model=model, new_attempt=2,
                            prior_status_sha256=sha(prior), prior_status_archived_and_verified=True,
                            payload_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
                            prior_optimization_spend='unknown; not counted as zero',
                            prior_optimization_upper_bound=base['protocols'][model]['protocol.json']['budget_flops'],
                            replacement_optimization_cap=base['protocols'][model]['protocol.json']['budget_flops'],
                            authorization='User requested completion of the other three models on 2026-11-12',
                            scientific_recipe='unchanged', recorded_at=time.time())
                        write(trial/'vm_loss_restart_audit.json', audit); remote.upload(trial/'vm_loss_restart_audit.json')
                state.update(status='running', current=name, model_scope=list(MODELS))
                record['status']='running'; write(state_path,state); drive.upload(state_path)
                print('THREE_MODEL_RESUME',name,flush=True)
                result = subprocess.run([sys.executable,'-m','scripts.colab_hfh.constraint_campaign',
                    '--root',str(root),'--payload',str(payload_path),'--work',str(trial),
                    '--folder','results_constraints_'+name,'--hours',str(max(.05,(deadline-time.time())/3600))])
                if result.returncode: raise RuntimeError('Recipe process failed: '+str(result.returncode))
                completed=read(trial/'campaign_status.json')
                if completed['status']!='complete_requested_matrix': raise RuntimeError('Recipe remains incomplete')
                rows=completed['results']
                if len(rows)!=9 or {(r['fold'],r['seed']) for r in rows}!={(f,s) for f in range(3) for s in range(3)}:
                    raise ValueError('Incomplete or duplicate matrix pairs')
                record.update(results=rows, full_matrix_complete=True, final_matrix_frozen=True,
                              status='complete_fixed_recipe')
                print('THREE_MODEL_VERIFIED',name,'mean_pp',sum(r['delta_dice_pp'] for r in rows)/9,flush=True)
            except Exception as error:
                failures.append(dict(recipe=name,error=str(error)))
                record['status']='recovery_error'; record['recovery_error']=str(error)
                print('THREE_MODEL_ERROR_CONTINUING',name,str(error),flush=True)
            write(state_path,state); drive.upload(state_path)
            summary,csv,report=write_report(root)
            drive.upload_report(csv);drive.upload_report(report)
        pending=[r['recipe'] for r in queue if not state['attempts'][r['recipe']].get('full_matrix_complete')]
        state.update(status='three_models_complete' if not pending else 'three_models_incomplete',
                     recovery_failures=failures, recovery_pending=pending)
        state.pop('current',None); write(state_path,state);drive.upload(state_path)
        print('THREE_MODEL_QUEUE_FINISHED',state['status'],'remaining',len(pending),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.root)
