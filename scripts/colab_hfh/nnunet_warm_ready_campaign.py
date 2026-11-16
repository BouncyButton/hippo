"""Three native controls followed by three readiness BAND runs; no seed pruning."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from scripts.colab_hfh.bands_campaign import read,write,sha
from scripts.colab_hfh.nnunet_warm_ready import NAME,FILES,payload,readiness


def validate_readiness(output,p):
    rows=[json.loads(x) for x in (output/'readiness_trace.jsonl').read_text().splitlines()]
    done=read(output/'completion.json');config=read(output/'config.json')
    if [r['update'] for r in rows]!=list(range(1,done['updates']+1)):
        raise ValueError('Missing or repeated readiness decisions')
    ready=False
    for i,row in enumerate(rows):
        if row['case_name'] not in config['train_cases'] or row['ready_before_step']!=ready:
            raise ValueError('Invalid readiness source or state')
        if row['band_call'] and not ready:raise ValueError('Premature BAND')
        if row['update']%10==0:ready=ready or readiness(rows[i-9:i+1])
        if row['ready_after_step']!=ready:raise ValueError('Readiness gate differs')
    cal=read(output/'calibration.json')
    first=next((r['update'] for r in rows if r['ready_after_step']),None)
    trace_path=output/'residual_band_trace.jsonl'
    trace=[json.loads(x) for x in trace_path.read_text().splitlines()] if trace_path.exists() else []
    if [r['charged_attempt'] for r in trace]!=[r['update'] for r in rows if r['band_call']]:
        raise ValueError('Auxiliary trace differs from actual calls')
    count=0
    for row in trace:
        count+=row['selected_errors']>0
        if row['active_update_count']!=count or row['applied_scale']!=(min(count/9,1.) if row['selected_errors'] else 0.):
            raise ValueError('Active ramp differs')
    if p['active_objective']!='dice' and trace:
        s=p['models']['nnunet']['objectives']['dice_bands']
        if cal['anchor_successful_updates']!=first or done['calibration_attempts']!=1 or done['calibration_flops']!=s['calibration_flops']:
            raise ValueError('Readiness calibration anchor or allowance differs')
        if [x['case_name'] for x in cal['cases']]!=config['train_cases']:
            raise ValueError('Calibration is not training-only')
        stats=cal['statistics']['joint']
        if stats['worst_ramp_median_ratio']>.1+1e-9 or stats['worst_ramp_p95_ratio']>.5+1e-9:
            raise ValueError('Calibration envelope exceeded')
    report=dict(first_ready_after_update=first,active_band_updates=count,
                first_band_update=trace[0]['charged_attempt'] if trace else None,
                readiness_verified=True,training_only=True,trace_count=len(rows))
    write(output/'readiness_audit.json',report)
    return report


def archive(output,durable,protocol,original,matched,seed):
    from scripts.colab_hfh import runtime
    from scripts.colab_hfh.constraint_schedule import training_flops
    p=read(protocol/'protocol.json');arm=p['active_objective']
    config=read(output/'config.json');done=read(output/'completion.json')
    s=p['models']['nnunet']['objectives'][arm]
    if done['updates']!=s['max_steps'] or done['discarded_attempts'] or done['consumed_training_flops']!=training_flops(done['updates'],s):
        raise ValueError('Incomplete or incorrectly charged run')
    if not done['consumed_training_flops']<=p['budget_flops']<training_flops(done['updates']+1,s):
        raise ValueError('Original optimization cap violated or underused')
    for key in ('seed','fold','dataset_sha256','train_cases','validation_cases','precision','lr_horizon_updates','augmentation'):
        if config[key]!=original['configuration'][key]:raise ValueError('Original setting differs: '+key)
    if config['initial_lr']!=.03 or config['nnunet_warmup_fraction']!=5/90 or config['nnunet_sgd_momentum']!=.95:
        raise ValueError('New native optimization recipe differs')
    if read(output/'initialization.json')!=original['initialization']:
        raise ValueError('Original seeded initialization differs')
    audit=validate_readiness(output,p)
    val=read(output/'validation_latest.json');dice=val['summary']['macro_hard_dice']
    row=dict(seed=seed,fold=0,arm=arm,updates=done['updates'],dice=dice,
             original_dice=original['result']['validation_dice'],
             delta_vs_original_pp=100*(dice-original['result']['validation_dice']),
             fp=sum(r['foreground_fp'] for r in val['cases']),fn=sum(r['foreground_fn'] for r in val['cases']),
             optimization_flops=done['consumed_training_flops'],optimization_cap=p['budget_flops'],
             calibration_flops=done['calibration_flops'],**audit)
    if matched is not None:
        for key in ('initial_lr','nnunet_warmup_fraction','nnunet_sgd_momentum','native_loss','train_cases','validation_cases','precision'):
            if config[key]!=matched['configuration'][key]:raise ValueError('Matched native setting differs: '+key)
        row.update(new_baseline_dice=matched['result']['validation_dice'],
                   delta_vs_new_baseline_pp=100*(dice-matched['result']['validation_dice']),
                   delta_vs_update89_baseline_pp=100*(dice-matched['validation_update89']['summary']['macro_hard_dice']),
                   fp_delta_vs_new=sum(r['foreground_fp'] for r in val['cases'])-sum(r['foreground_fp'] for r in matched['validation']['cases']),
                   fn_delta_vs_new=sum(r['foreground_fn'] for r in val['cases'])-sum(r['foreground_fn'] for r in matched['validation']['cases']))
    write(output/'comparison.json',row)
    source=Path(runtime.__file__).read_text().replace("schedule = p['models'][p['trainer_alias']]['objectives']['dice']",
                  "schedule = p['models'][p['trainer_alias']]['objectives'][p['active_objective']]")
    ns={'__name__':'warm_ready_archive','__file__':__file__};exec(compile(source,__file__,'exec'),ns)
    ns['archive'](output,durable,'baselines','nnunet',0,seed,protocol)
    return row


def baseline_reference(folder,protocol,archive_sha):
    return dict(result=read(folder/'result.json'),initialization=read(folder/'initialization.json'),
        configuration=read(folder/'config.json'),validation=read(folder/'validation_latest.json'),
        validation_update89=read(folder/'validation_update89.json'),training=read(folder/'training_final.json'),
        archive_sha256=archive_sha,baseline_protocol_sha256=sha(protocol/'protocol.json'))


def run(root,hours):
    from scripts.colab_hfh.constraint_drive import Drive
    from scripts.colab_hfh.runtime import verify_archive
    work=root/NAME;work.mkdir(exist_ok=True);bundle=root/'bundle'
    # Serialize with other conventional training. Never interrupt a live model.
    with (root/'bands_recipe_search.lock').open('a') as controller_lock, (root/'bands_campaign.lock').open('a') as lock:
        fcntl.flock(controller_lock,fcntl.LOCK_EX)
        fcntl.flock(lock,fcntl.LOCK_EX)
        drive=Drive('results_'+NAME)
        downloads=work/'remote';downloads.mkdir(exist_ok=True)
        drive.call('copy',drive.folder,str(downloads),'--include','*.zip','--include','campaign_status.json',
                   '--transfers','2','--checkers','2','--checksum')
        for path in downloads.glob('*.zip'):
            if path.stem not in {f'{arm}_f0_s{seed}' for arm in ('dice','dice_bands') for seed in (0,1,2)}:
                raise ValueError('Unexpected remote run identity')
            arm,seed=path.stem.rsplit('_f0_s',1);durable=work/arm/'archives'/f'nnunet_f0_s{seed}'
            durable.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(path) as z:
                manifest=json.loads(z.read('results/archive_manifest.json'))
                for name,expected in manifest.items():
                    if Path(name).name!=name or hashlib.sha256(z.read('results/'+name)).hexdigest()!=expected:
                        raise ValueError('Remote result archive integrity failed')
                for name in z.namelist():
                    parts=Path(name).parts
                    if len(parts)==2 and parts[0]=='results':
                        target=durable/parts[1];data=z.read(name)
                        if target.exists() and target.read_bytes()!=data:raise ValueError('Remote/local evidence differs')
                        target.write_bytes(data)
            verify_archive(durable);shutil.copyfile(path,work/path.name)
        prior=downloads/'campaign_status.json'
        if prior.exists():
            previous=read(prior);current=previous.get('current')
            if current and previous['status'] in ('running','runtime_error'):
                arm,seed=current.rsplit('_f0_s',1);key=f'nnunet_f0_s{seed}'
                if not (work/arm/'archives'/key/'cleanup.json').exists() and not (work/arm/'runs'/key/'budget_ledger.json').exists():
                    raise RuntimeError('Prior VM lost an unfinished charged run; recovery audit required: '+current)
        base=read(root/'bands_v1_payload.json');plan=read(root/'constraint_matrix_plan.json')
        if read(bundle/'gpu_overlay.json')['precision']!=base['precision']:
            raise ValueError('Precision differs from the frozen controls')
        state=dict(status='running',seeds=[0,1,2],folds=[0],results=[],
                   interpretation='Original control vs warmup0.03 control vs readiness BAND; no pruning; same cap',
                   hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip())
        write(work/'campaign_status.json',state);drive.upload(work/'campaign_status.json')
        env=dict(os.environ,PYTHONPATH=':'.join(map(str,[bundle/'compatibility',bundle/'source',bundle/'source/scripts',bundle/'source/baselines/nnUNet'])),
                 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',HIPPO_HFH_GPU_CACHE='1')
        deadline=time.time()+hours*3600;matched={}
        try:
            for arm in ('dice','dice_bands'):
                pack=payload(base,plan,arm)
                if arm!='dice':pack['references'].update(matched)
                protocol=work/arm/'protocol';protocol.mkdir(parents=True,exist_ok=True)
                for name,value in pack['protocols']['nnunet'].items():
                    text=value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
                    target=protocol/name
                    if target.exists() and target.read_text()!=text:raise ValueError('Frozen protocol changed')
                    target.write_text(text)
                write(protocol/'baseline_references.json',pack['references'])
                for seed in (0,1,2):
                    key=f'nnunet_f0_s{seed}';name=f'{arm}_f0_s{seed}'
                    output=work/arm/'runs'/key;durable=work/arm/'archives'/key
                    output.mkdir(parents=True,exist_ok=True);durable.mkdir(parents=True,exist_ok=True)
                    if (durable/'cleanup.json').exists():
                        verify_archive(durable);row=read(durable/'comparison.json')
                        if read(durable/'runtime_recipe.json')['protocol_sha256']!=sha(protocol/'protocol.json'):
                            raise ValueError('Cached run belongs to another protocol')
                    else:
                        if time.time()>=deadline or (work/'STOP').exists():
                            state['status']='paused';write(work/'campaign_status.json',state);drive.upload(work/'campaign_status.json');return
                        state.update(current=name);write(work/'campaign_status.json',state);drive.upload(work/'campaign_status.json')
                        print('WARM_READY_START',name,flush=True)
                        write(output/'execution_environment.json',dict(gpu=state['hardware'],precision=base['precision']))
                        if not (output/'completion.json').exists():
                            with (output/'worker.log').open('a') as log:
                                subprocess.run([sys.executable,'-m','scripts.colab_hfh.nnunet_warm_ready_worker',
                                    '--protocol-dir',str(protocol),'--output',str(output),'--durable',str(durable),
                                    '--pkl',str(bundle/'data_index.json'),'--fold','0','--seed',str(seed)],
                                    cwd=bundle/'source',env={**env,'HIPPO_DEADLINE':str(deadline)},stdout=log,stderr=subprocess.STDOUT,check=True)
                        row=archive(output,durable,protocol,base['references'][key],matched.get(key),seed)
                    archive_path=work/(name+'.zip')
                    if not archive_path.exists():
                        with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as z:
                            for path in sorted(durable.iterdir()):
                                if path.is_file():z.write(path,'results/'+path.name)
                            for path in sorted(protocol.iterdir()):z.write(path,'protocol/'+path.name)
                            for file in FILES+('residual_band.py','residual_band_v2.py','baseline.py','runtime.py','constraint_schedule.py','residual_band_v2_worker.py'):
                                z.write(bundle/'source/scripts/colab_hfh'/file,'source/'+file)
                    drive.upload(archive_path)
                    if arm=='dice':matched[key]=baseline_reference(durable,protocol,sha(archive_path))
                    state['results'].append(row);write(work/'campaign_status.json',state);drive.upload(work/'campaign_status.json')
                    print('WARM_READY_VERIFIED',json.dumps(row),flush=True)
            state.update(status='complete_six_runs',current=None)
            write(work/'campaign_status.json',state);drive.upload(work/'campaign_status.json')
            print('WARM_READY_COMPLETE',flush=True)
        except Exception as error:
            state.update(status='runtime_error',error=str(error))
            write(work/'campaign_status.json',state)
            try:drive.upload(work/'campaign_status.json')
            finally:raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--hours',type=float,default=8)
    a=p.parse_args();run(a.root,a.hours)
