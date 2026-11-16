"""Three concurrent A100 controls, then the two preregistered composition tests."""
import argparse
from copy import deepcopy
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from scripts.colab_hfh import nnunet_composition_calibration as composition
from scripts.colab_hfh import runtime
from scripts.colab_hfh.constraint_schedule import training_flops
from scripts.colab_hfh.nnunet_composition_campaign import (
    SDKDrive, check_campaign_completion, check_no_workers, check_prior_attempt,
    verify_package, retire, zip_archive)

NAME = 'nnunet_composition_a100_v1'
read, write, digest = runtime.read, runtime.write, runtime.digest


def archive_without_retirement(output, durable, protocol):
    source = inspect.getsource(runtime.archive)
    old = "schedule = p['models'][p['trainer_alias']]['objectives']['dice']"
    if source.count(old) != 1 or source.count('    deleted = []') != 1:
        raise ValueError('Archive integration changed')
    source = source.replace(old,"schedule = p['models'][p['trainer_alias']]['objectives'][p['active_objective']]")
    source = source[:source.index('    deleted = []')]+'    return result\n'
    namespace = dict(runtime.__dict__)
    exec(compile(source,__file__,'exec'),namespace)
    return namespace['archive'](output,durable,'baselines','nnunet',0,0,protocol)


def validate_result(root, package, name):
    output = root/'runs'/name/'run'
    protocol = root/'protocols'/name
    p, c, done = (read(protocol/'protocol.json'),read(output/'config.json'),read(output/'completion.json'))
    ref = package/'controls/native/results'
    original = read(ref/'config.json')
    for key in ('fold','seed','dataset_sha256','train_cases','validation_cases','precision',
                'initial_lr','lr_horizon_updates','nnunet_warmup_fraction','nnunet_sgd_momentum','augmentation','native_loss'):
        if c[key] != original[key]:
            raise ValueError('Native matching failed: '+key)
    if read(output/'initialization.json') != read(ref/'initialization.json'):
        raise ValueError('Native initialization changed')
    arm = p['active_objective']
    schedule = p['models']['nnunet']['objectives'][arm]
    used = training_flops(done['updates'],schedule)
    expected_steps = 180 if name == 'native' else 179
    if (done['status'] != 'complete' or done['discarded_attempts'] or done['updates'] != expected_steps or
            done['updates'] != schedule['max_steps'] or done['consumed_training_flops'] != used or
            not used <= composition.CAP < training_flops(done['updates']+1,schedule) or
            done['calibration_flops'] != schedule['calibration_flops'] or
            done['calibration_attempts'] != (0 if name == 'native' else 1)):
        raise ValueError('Optimization or calibration budget differs')
    native_anchor = read(root/'runs/native/run/native_update60.json')
    if name != 'native':
        calibration = read(output/'calibration.json')
        if (calibration['anchor_successful_updates'] != 60 or
                calibration['anchor_model_sha256'] != native_anchor['model_state_sha256']):
            raise ValueError('Native prefix differs on A100')
        if name in composition.ARMS:
            if not calibration.get('composition_anchor_verified'):
                raise ValueError('Composition calibration was not verified')
            composition.ANCHOR = native_anchor
            composition.check_anchor(calibration,p['composition'])
    val, train = read(output/'validation_latest.json'),read(output/'training_final.json')
    row = dict(variant=name,arm=arm,fold=0,seed=0,updates=done['updates'],
        validation_dice=val['summary']['macro_hard_dice'],training_dice=train['summary']['macro_hard_dice'],
        optimization_flops=used,optimization_cap=composition.CAP,calibration_flops=done['calibration_flops'],
        native_update60=native_anchor,ap_swaps_applicable=False,
        gpu=read(root/'launch_provenance.json')['hardware']['gpu'],primary_comparison='same_A100_native_final_budget')
    if name != 'native':
        row['weights'] = read(output/'calibration.json')['weights']
    for split, values in (('validation',val),('training',train)):
        for error in ('fp','fn'):
            row[split+'_'+error] = sum(r['foreground_'+error] for r in values['cases'])
    if name != 'native':
        n = read(root/'runs/native/archive/validation_latest.json')
        by_case = {r['case_name']:r for r in n['cases']}
        deltas = [100*(r['macro_hard_dice']-by_case[r['case_name']]['macro_hard_dice']) for r in val['cases']]
        row.update(native_gain_pp=100*(row['validation_dice']-n['summary']['macro_hard_dice']),
            improved_crops=sum(d>0 for d in deltas),worst_crop_pp=min(deltas),per_crop_delta_pp=deltas,
            native_equal_update_gain_pp=100*(row['validation_dice']-read(root/'runs/native/archive/validation_update179.json')['summary']['macro_hard_dice']))
    if name in composition.ARMS:
        row['reference_gains_pp'] = {control:100*(row['validation_dice']-
            read(root/'runs'/control/'archive/validation_latest.json')['summary']['macro_hard_dice'])
            for control in ('band_edge','band_edge_depth')}
    row['historical_L4_native_gain_pp_descriptive_only'] = 100*(row['validation_dice']-
        read(ref/'validation_latest.json')['summary']['macro_hard_dice'])
    write(output/'comparison.json',row)
    archive_without_retirement(output,root/'runs'/name/'archive',protocol)
    return row


def prepare_protocols(root, package, controls_only):
    if controls_only:
        for name in ('native','band_edge','band_edge_depth'):
            target = root/'protocols'/name
            if target.exists():
                continue
            shutil.copytree(package/'controls'/name/'protocol',target)
            p = read(target/'protocol.json')
            p.update(a100_native_anchor=str(root/'runs/native/run/native_update60.json'),
                     hardware_campaign=NAME)
            write(target/'protocol.json',p)
    else:
        composition.ANCHOR = read(root/'runs/native/archive/native_update60.json')
        parent = read(root/'runs/band_edge/archive/calibration.json')
        triple = read(root/'runs/band_edge_depth/archive/calibration.json')
        for name,arm in composition.ARMS.items():
            reference = root/'protocols'/('band_edge' if arm == 'dice_bands_edge' else 'band_edge_depth')
            specification = composition.derive_specification(parent,triple,name)
            p = read(reference/'protocol.json')
            p['matched_readiness_anchor'] = deepcopy(composition.ANCHOR)
            p = composition.make_protocol(p,specification)
            p['name'] = NAME+'_'+name
            target = root/'protocols'/name
            if target.exists():
                if read(target/'protocol.json') != p:
                    raise ValueError('Frozen A100 protocol changed')
            else:
                shutil.copytree(reference,target)
                write(target/'protocol.json',p)


def run(root, package):
    verify_package(package)
    root.mkdir(exist_ok=True,parents=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_no_workers()
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
        if 'A100' not in gpu or int(gpu.rsplit(',',1)[1]) < 39000:
            raise RuntimeError('At least a 40 GB A100 required for three workers')
        drive = SDKDrive('results_'+NAME)
        drive.api.download('hfh_storage:results_medsam3_component_constraints125_f0s0_v1/monitor/live_status.json',root/'medsam_completion.json')
        # User explicitly authorized this separate A100 to run independently of
        # the remaining MedSAM job. Record its status without a completion gate.
        unetr = read(package/'unetr_verification.json')
        if unetr.get('status') != 'complete' or unetr.get('unetr_target_pairs') != 9:
            raise ValueError('UNETR reference completion is not verified')
        quota = json.loads(drive.call('about','hfh_storage:','--json'))
        if min(quota['free'],shutil.disk_usage(root).free) < 12*1024**3:
            raise RuntimeError('Three workers require 12 GiB free local/Drive storage; recovery retained')
        write(root/'storage_preflight.json',dict(drive=quota,local_free=shutil.disk_usage(root).free,
            peak_generated_estimate_bytes=8*1024**3,workers=3,checked_at=time.time(),gpu=gpu))
        for relative, expected in read(root/'execution_source_manifest.json').items():
            if digest(root/'bundle/source'/relative) != expected:
                raise ValueError('Execution source changed: '+relative)
        state_path = root/'status.json'
        previous = read(state_path) if state_path.exists() else None
        check_prior_attempt(previous,drive.previous_state(root/'previous_remote_status.json'))
        state = previous or dict(run_id=uuid.uuid4().hex,status='ready',results=[],workers=[],max_gpu_workers=3)
        write(state_path,state)
        for path in ('execution_source.zip','launch_provenance.json','storage_preflight.json'):
            drive.upload(root/path)
        for name in ('native','band_edge','band_edge_depth'):
            runtime.verify_archive(package/'controls'/name/'results')
        env = dict(os.environ,PYTHONPATH=':'.join([str(root)]+[str(root/'bundle'/p) for p in
            ('compatibility','source','source/scripts','source/baselines/nnUNet')]),
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',
            HIPPO_HFH_GPU_CACHE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        active = {}
        def publish():
            state['workers'] = [dict(name=name,pid=proc.pid) for name,(proc,_) in active.items()
                                if proc is not None and proc.poll() is None]
            state['updated_at'] = time.time()
            write(state_path,state);drive.upload(state_path)
        try:
            for phase,names in [('controls',('native','band_edge','band_edge_depth')),('tests',tuple(composition.ARMS))]:
                prepare_protocols(root,package,phase == 'controls')
                state['phase'] = phase
                for name in names:
                    output, durable = root/'runs'/name/'run',root/'runs'/name/'archive'
                    output.mkdir(parents=True,exist_ok=True);durable.mkdir(parents=True,exist_ok=True)
                    if (durable/'cleanup.json').exists():
                        runtime.verify_archive(durable)
                        continue
                    marker = output/'controller_attempt.json'
                    if marker.exists() and not (output/'budget_ledger.json').exists() and not (output/'completion.json').exists():
                        raise RuntimeError('Missing charged ledger for prior attempt: '+name)
                    if (output/'completion.json').exists():
                        active[name] = (None,None)
                        continue
                    write(marker,dict(run_id=state['run_id'],started_at=time.time()))
                    write(output/'execution_environment.json',read(root/'launch_provenance.json')['hardware'])
                    state.update(status='launching',current=name);publish()
                    log = (output/'worker.log').open('a')
                    proc = subprocess.Popen([sys.executable,'-m','scripts.colab_hfh.nnunet_composition_a100_worker',
                        '--protocol-dir',str(root/'protocols'/name),'--output',str(output),'--durable',str(durable),
                        '--pkl',str(root/'bundle/data_index.json'),'--fold','0','--seed','0'],
                        cwd=root/'bundle/source',env={**env,'HIPPO_DEADLINE':str(time.time()+3600)},
                        stdout=log,stderr=subprocess.STDOUT)
                    active[name] = (proc,log)
                state.update(status='running',current=None);publish()
                last_publish = time.monotonic()
                while active:
                    for name,(proc,log) in list(active.items()):
                        if proc is not None and proc.poll() is None:
                            continue
                        if proc is not None and proc.returncode:
                            raise RuntimeError('Worker failed; no replacement attempt: '+name)
                        if name != 'native' and not (root/'runs/native/archive/cleanup.json').exists():
                            continue
                        if log is not None:log.close()
                        row = validate_result(root,package,name)
                        durable, output = root/'runs'/name/'archive',root/'runs'/name/'run'
                        zipped = root/(name+'.zip')
                        zip_archive(zipped,durable,root/'protocols'/name,package)
                        drive.upload(zipped)
                        retire(output,durable)
                        zip_archive(zipped,durable,root/'protocols'/name,package);drive.upload(zipped)
                        state['results'] = [r for r in state['results'] if r['variant'] != name]+[row]
                        del active[name]
                        publish()
                        print('A100_ARCHIVED',json.dumps(row),flush=True)
                    if active:
                        if time.monotonic()-last_publish > 30:publish();last_publish=time.monotonic()
                        time.sleep(1)
            state.update(status='complete_five_arms',current=None,workers=[],
                storage_after=json.loads(drive.call('about','hfh_storage:','--json')))
            publish()
        except BaseException as error:
            # Do not kill another worker or erase its charged work/recovery.
            state.update(status='error_no_expansion',error=str(error))
            for proc,log in active.values():
                if proc is not None:proc.wait()
                if log is not None:log.close()
            active.clear()
            publish()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    args = parser.parse_args()
    run(args.root,args.package)
