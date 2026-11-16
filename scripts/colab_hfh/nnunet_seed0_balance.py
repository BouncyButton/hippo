"""Bounded seed-zero diagnosis; never expand automatically to other seeds."""
import fcntl
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

from scripts.colab_hfh.bands_campaign import read,write,sha
from scripts.colab_hfh.nnunet_warm_ready import payload
from scripts.colab_hfh import nnunet_warm_ready_campaign as campaign


def run(root):
    import psutil
    from scripts.colab_hfh.constraint_drive import Drive
    work=root/'nnunet_seed0_balance';work.mkdir(exist_ok=True)
    base=read(root/'bands_v1_payload.json');plan=read(root/'constraint_matrix_plan.json')
    bundle=root/'bundle';ref=base['references']['nnunet_f0_s0']
    control=root/'nnunet_seed0_plateau/kickstart_plateau007'
    matched=campaign.baseline_reference(control/'archive',control/'protocol',sha(root/'nnunet_seed0_plateau/kickstart_plateau007.zip'))
    control_row=read(control/'archive/comparison.json')
    if control_row['dice']<ref['result']['validation_dice']-.01:raise ValueError('Matched baseline did not recover')
    env=dict(os.environ,PYTHONPATH=':'.join(str(bundle/p) for p in ('compatibility','source','source/scripts','source/baselines/nnUNet')),
             PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',HIPPO_HFH_GPU_CACHE='1')
    # Pause only the queue dispatcher. Its current complete recipe may finish.
    # Always release it, including on diagnostic/transport exceptions.
    dispatchers=[p for p in psutil.process_iter(['cmdline']) if str(root/'resume_three_models.py') in (p.info['cmdline'] or [])]
    if len(dispatchers)!=1:raise RuntimeError('Expected exactly one three-model queue dispatcher')
    dispatcher=dispatchers[0]
    state=dict(status='waiting_for_current_recipe',seed=0,fold=0,results=[],dispatcher_pid=dispatcher.pid,
               policy='No seed1/2 expansion; baseline must recover before BAND')
    write(work/'status.json',state)
    dispatcher.suspend()
    try:
        with (root/'bands_campaign.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            drive=Drive('results_nnunet_seed0_balance')
            # Reuse the archive verification with the declared LR and warm-up,
            # rather than the old ablation's hard-coded0.03/five-step recipe.
            code=inspect.getsource(campaign.archive).replace("config['initial_lr']!=.03", "config['initial_lr']!=p['initial_lrs']['nnunet']")
            code=code.replace("config['nnunet_warmup_fraction']!=5/90", "config['nnunet_warmup_fraction']!=p['nnunet_warmup_fraction']")
            ns=dict(campaign.__dict__)
            validation=inspect.getsource(campaign.validate_readiness)
            validation=validation.replace("count+=row['selected_errors']>0","count+=row['recall_balance_active']")
            validation=validation.replace("if row['selected_errors'] else 0.","if row['recall_balance_active'] else 0.")
            exec(compile(validation,'recall_balance_readiness_audit','exec'),ns)
            exec(compile(code,'seed0_repair_archive','exec'),ns)
            for name,arm,warmup in [('kickstart_plateau007_recall_balance','dice_bands',0)]:
                pack=payload(base,plan,arm);p=pack['protocols']['nnunet']['protocol.json']
                p.update(name='seed0_repair_'+name,nnunet_warmup_fraction=warmup/90,
                    nnunet_warmup_updates=warmup,constraint_definition='Original native objective; seed0 diagnosis '+name,
                    detached_head_diagnostic_flops_allowance=11*4*64*64**3)
                p['initial_lrs']['nnunet']=.1
                p['recall_balance_source_sha256']={n:sha(bundle/'source/scripts/colab_hfh'/n) for n in ('residual_band_recall_balance.py','nnunet_seed0_balance_worker.py')}
                p['recall_balance_policy']='outer=min(1,selected_FN/max(1,selected_FP)); inner=1; detached counts; training only'
                schedule=p['models']['nnunet']['objectives'][arm]
                extra=16*64**3
                for key in ('baseline_flops_per_update','early_flops_per_update','flops_per_update'):schedule[key]+=extra
                schedule['recall_balance_flops_per_update']=extra
                steps=p['budget_flops']//schedule['flops_per_update'];assert steps==89
                used=steps*schedule['flops_per_update']
                schedule.update(max_steps=steps,consumed_training_flops=used,consumed_flops=used,
                    consumed_total_flops=used+schedule['calibration_flops'],unused_flops=p['budget_flops']-used)
                p['constraint_definition']='Readiness residual BAND; selected-error-count attenuation of outer term'
                p['seed0_plateau_schedule']=dict(original_prefix_updates=20,plateau_lr=.07,decay_start=60,horizon=90,power=.9)
                if matched is not None and arm!='dice':pack['references']['nnunet_f0_s0']=matched
                protocol=work/name/'protocol';protocol.mkdir(parents=True,exist_ok=True)
                for filename,value in pack['protocols']['nnunet'].items():
                    text=value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
                    path=protocol/filename
                    if path.exists() and path.read_text()!=text:raise ValueError('Diagnostic protocol changed')
                    path.write_text(text)
                write(protocol/'baseline_references.json',pack['references'])
                out=work/name/'run';durable=work/name/'archive';out.mkdir(exist_ok=True);durable.mkdir(exist_ok=True)
                if (durable/'cleanup.json').exists():
                    from scripts.colab_hfh.runtime import verify_archive
                    verify_archive(durable);row=read(durable/'comparison.json')
                else:
                    state.update(status='running',current=name);write(work/'status.json',state);drive.upload(work/'status.json')
                    print('SEED0_DIAG_START',name,flush=True)
                    write(out/'execution_environment.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()))
                    with (out/'worker.log').open('a') as log:
                        subprocess.run([sys.executable,'-m','scripts.colab_hfh.nnunet_seed0_balance_worker',
                            '--protocol-dir',str(protocol),'--output',str(out),'--durable',str(durable),
                            '--pkl',str(bundle/'data_index.json'),'--fold','0','--seed','0'],
                            cwd=bundle/'source',env={**env,'HIPPO_DEADLINE':str(time.time()+1800)},stdout=log,stderr=subprocess.STDOUT,check=True)
                    row=ns['archive'](out,durable,protocol,ref,matched if arm!='dice' else None,0)
                path=work/(name+'.zip')
                if not path.exists():
                    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
                        for folder,prefix in ((durable,'results'),(protocol,'protocol')):
                            for f in sorted(folder.iterdir()):
                                if f.is_file():z.write(f,prefix+'/'+f.name)
                        for n in ('nnunet_seed0_balance.py','nnunet_seed0_balance_worker.py','residual_band_recall_balance.py'):
                            z.write(bundle/'source/scripts/colab_hfh'/n,'source/'+n)
                drive.upload(path)
                row['recipe']=name;state['results'].append(row);write(work/'status.json',state);drive.upload(work/'status.json')
                print('SEED0_DIAG_VERIFIED',json.dumps(row),flush=True)
                if arm=='dice' and row['dice']<.5:
                    state['status']='native_failure_no_band_or_seed_expansion';break
                if name=='original_lr_replay' and abs(row['dice']-ref['result']['validation_dice'])>.001:
                    state['status']='original_replay_differs_no_recipe_expansion';break
                if name=='kickstart_plateau007':
                    matched=campaign.baseline_reference(durable,protocol,sha(path))
                    if row['dice']<ref['result']['validation_dice']-.01 or row['first_ready_after_update'] is None or row['first_ready_after_update']>60:
                        state['status']='warmup_not_adequate_no_band_or_seed_expansion';break
            else:state['status']='seed0_diagnosis_complete_no_expansion'
            write(work/'status.json',state);drive.upload(work/'status.json')
    except BaseException as error:
        state.update(status='error_no_seed_expansion',error=str(error));write(work/'status.json',state)
        raise
    finally:
        if dispatcher.is_running():dispatcher.resume()
        print('THREE_MODEL_DISPATCHER_RESUMED',dispatcher.pid,flush=True)


if __name__=='__main__':run(Path('/content/hippo_hfh_baselines_v2'))
