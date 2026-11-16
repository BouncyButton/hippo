"""One matched doubled-budget pair without augmentation; never launches another seed or fold."""
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from scripts.colab_hfh.bands_campaign import read,write,sha
from scripts.colab_hfh.nnunet_warm_ready import payload
from scripts.colab_hfh import nnunet_warm_ready_campaign as campaign
from scripts.colab_hfh.nnunet_seed0_long_worker import SCHEDULE

FILES=('nnunet_seed0_long.py','nnunet_seed0_long_worker.py')


def trace(path):return [json.loads(line) for line in path.read_text().splitlines()]


def longer_archive():
    ns=dict(campaign.__dict__)
    validation=inspect.getsource(campaign.validate_readiness)
    validation=validation.replace("count+=row['selected_errors']>0","count+=row['recall_balance_active']")
    validation=validation.replace("if row['selected_errors'] else 0.","if row['recall_balance_active'] else 0.")
    exec(compile(validation,'long_readiness_audit','exec'),ns)
    code=inspect.getsource(campaign.archive)
    code=code.replace("config['initial_lr']!=.03","config['initial_lr']!=p['initial_lrs']['nnunet']")
    code=code.replace("config['nnunet_warmup_fraction']!=5/90","config['nnunet_warmup_fraction']!=0")
    code=code.replace("'precision','lr_horizon_updates','augmentation'","'precision','augmentation'")
    code=code.replace('update89','update179')
    old="    write(output/'comparison.json',row)"
    new="""    if config['lr_horizon_updates']!=180:raise ValueError('Long horizon differs')
    train=read(output/'training_final.json')
    row.update(training_dice=train['summary']['macro_hard_dice'],
               training_fp=sum(x['foreground_fp'] for x in train['cases']),
               training_fn=sum(x['foreground_fn'] for x in train['cases']))
    write(output/'comparison.json',row)"""
    if code.count(old)!=1:raise ValueError('Archive integration changed')
    exec(compile(code.replace(old,new),'long_archive','exec'),ns)
    return ns['archive']


def make_pack(base,plan,arm,source):
    pack=payload(base,plan,arm);p=pack['protocols']['nnunet']['protocol.json']
    assert p['budget_flops']==31976450667090
    p.update(name='nnunet_seed0_long_v1_'+arm,augmentation='none',seeds=[0],folds=[0],
             budget_flops=2*p['budget_flops'],lr_horizon_updates=180,
             nnunet_warmup_fraction=0,nnunet_warmup_updates=0,long_schedule=SCHEDULE,
             long_source_sha256={n:sha(source/n) for n in FILES+('nnunet_seed0_balance_worker.py','residual_band_recall_balance.py')},
             constraint_definition='Doubled optimization cap; unchanged native loss and balanced readiness BAND',
             endpoint='Final doubled optimization cap endpoint; no automatic seed expansion',
             detached_head_diagnostic_flops_allowance=20*4*64*64**3)
    p['initial_lrs']['nnunet']=.1
    schedule=p['models']['nnunet']['objectives'][arm]
    extra=16*64**3 if arm!='dice' else 0
    for key in ('baseline_flops_per_update','early_flops_per_update','flops_per_update'):schedule[key]+=extra
    schedule['recall_balance_flops_per_update']=extra
    steps=p['budget_flops']//schedule['flops_per_update']
    assert steps==(180 if arm=='dice' else 179)
    used=steps*schedule['flops_per_update']
    schedule.update(max_steps=steps,consumed_training_flops=used,consumed_flops=used,
        consumed_total_flops=used+schedule['calibration_flops'],unused_flops=p['budget_flops']-used,patient_pass_epochs=steps/10)
    p['diagnostics']['logit_probe_max_flops']*=2
    p['diagnostics']['parameter_probe_max_flops']*=2
    p['diagnostics']['equal_update_clean_training_evaluation']='At update179, clean training inference; no fitting'
    pack.update(folds=[0],seeds=[0],rates={'nnunet':.1})
    return pack


def run(root):
    import psutil
    from scripts.colab_hfh.constraint_drive import Drive
    from scripts.colab_hfh.runtime import verify_archive
    work=root/'nnunet_seed0_long_v1';work.mkdir(exist_ok=True)
    bundle=root/'bundle';source=bundle/'source/scripts/colab_hfh'
    base=read(root/'bands_v1_payload.json');plan=read(root/'constraint_matrix_plan.json')
    original=base['references']['nnunet_f0_s0']
    unaug=root/'nnunet_seed0_plateau/kickstart_plateau007/archive'
    verify_archive(unaug);unaug_row=read(unaug/'comparison.json')
    dispatchers=[p for p in psutil.process_iter(['cmdline','status']) if p.info['status']!='zombie' and str(root/'resume_three_models.py') in (p.info['cmdline'] or [])]
    if len(dispatchers)>1:raise RuntimeError('Multiple matrix dispatchers; refuse ambiguity')
    dispatcher=dispatchers[0] if dispatchers else None
    state=dict(status='waiting_for_current_recipe',seed=0,fold=0,results=[],
               unaugmented_native_dice=unaug_row['dice'],policy='Run only native and balanced BAND seed0; no automatic expansion')
    write(work/'status.json',state)
    if dispatcher:dispatcher.suspend()
    env=dict(os.environ,PYTHONPATH=':'.join(str(bundle/p) for p in ('compatibility','source','source/scripts','source/baselines/nnUNet')),
             PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',HIPPO_HFH_GPU_CACHE='1')
    try:
        with (root/'bands_campaign.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            drive=Drive('results_nnunet_seed0_long_v1');archive=longer_archive();matched=None
            for name,arm in [('native','dice'),('balanced_band','dice_bands')]:
                pack=make_pack(base,plan,arm,source);p=pack['protocols']['nnunet']['protocol.json']
                if matched:pack['references']['nnunet_f0_s0']=matched
                protocol=work/name/'protocol';protocol.mkdir(parents=True,exist_ok=True)
                for filename,value in pack['protocols']['nnunet'].items():
                    content=value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
                    target=protocol/filename
                    if target.exists() and target.read_text()!=content:raise ValueError('Frozen pilot protocol changed')
                    target.write_text(content)
                write(protocol/'baseline_references.json',pack['references'])
                output=work/name/'run';durable=work/name/'archive';output.mkdir(exist_ok=True);durable.mkdir(exist_ok=True)
                if (durable/'cleanup.json').exists():verify_archive(durable);row=read(durable/'comparison.json')
                else:
                    state.update(status='running',current=name);write(work/'status.json',state);drive.upload(work/'status.json')
                    print('LONG_START',name,flush=True)
                    write(output/'execution_environment.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()))
                    if (output/'completion.json').exists():
                        raise RuntimeError('Completed unarchived output requires archive-only recovery')
                    with (output/'worker.log').open('a') as log:
                        subprocess.run([sys.executable,'-m','scripts.colab_hfh.nnunet_seed0_long_worker','--protocol-dir',str(protocol),
                            '--output',str(output),'--durable',str(durable),'--pkl',str(bundle/'data_index.json'),'--fold','0','--seed','0'],
                            cwd=bundle/'source',env={**env,'HIPPO_DEADLINE':str(time.time()+1800)},stdout=log,stderr=subprocess.STDOUT,check=True)
                    row=archive(output,durable,protocol,original,matched,0)
                zipped=work/(name+'.zip')
                if not zipped.exists():
                    with zipfile.ZipFile(zipped,'w',zipfile.ZIP_DEFLATED) as z:
                        for folder,prefix in ((durable,'results'),(protocol,'protocol')):
                            for f in sorted(folder.iterdir()):
                                if f.is_file():z.write(f,prefix+'/'+f.name)
                        for filename in FILES:z.write(source/filename,'source/'+filename)
                drive.upload(zipped)
                row=dict(row,recipe=name,delta_vs_unaugmented_native_pp=100*(row['dice']-unaug_row['dice']))
                state['results'].append(row);write(work/'status.json',state);drive.upload(work/'status.json')
                print('LONG_VERIFIED',json.dumps(row),flush=True)
                if name=='native':matched=dict(result=read(durable/'result.json'),initialization=read(durable/'initialization.json'),configuration=read(durable/'config.json'),validation=read(durable/'validation_latest.json'),validation_update179=read(durable/'validation_update179.json'),training=read(durable/'training_final.json'),archive_sha256=sha(zipped),baseline_protocol_sha256=sha(protocol/'protocol.json'))
            native,band=state['results']
            ntrace=trace(work/'native/archive/readiness_trace.jsonl');btrace=trace(work/'balanced_band/archive/readiness_trace.jsonl')
            first=native['first_ready_after_update']
            if first is not None and ntrace[:first]!=btrace[:first]:raise ValueError('Native-only training prefix differs')
            ready_paths=[work/name/'archive/readiness_model.json' for name in ('native','balanced_band')]
            if ready_paths[0].exists()!=ready_paths[1].exists():raise ValueError('Readiness prefix differs')
            if all(path.exists() for path in ready_paths) and read(ready_paths[0])!=read(ready_paths[1]):
                raise ValueError('Native-only prefix differs before constraint activation')
            state.update(status='complete_seed0_pair_no_expansion',paired_native_prefix_verified=first,
                         baseline_non_regression=native['dice']>=unaug_row['dice'],
                         constraint_improves_matched=band['dice']>native['dice'])
            state.pop('current',None);write(work/'status.json',state);drive.upload(work/'status.json')
            print('LONG_PAIR_COMPLETE',json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='error_no_seed_expansion',error=str(error));write(work/'status.json',state)
        raise
    finally:
        if dispatcher and dispatcher.is_running():dispatcher.resume()
        print('OTHER_MODEL_QUEUE_RELEASED',dispatcher.pid if dispatcher else 'already finished',flush=True)


if __name__=='__main__':run(Path('/content/hippo_hfh_baselines_v2'))
