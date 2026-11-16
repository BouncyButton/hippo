"""Two sequential, source-frozen long-budget tests after both campaigns finish.

The worker runs in an isolated bundle. No dispatcher is stopped or modified.
Result archives are uploaded and verified before checkpoint retirement.
"""
import argparse
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from scripts.colab_hfh.nnunet_composition_calibration import NAME, CAP, ANCHOR, ARMS, check_anchor
from scripts.colab_hfh.constraint_schedule import training_flops
from scripts.colab_hfh import runtime

read, write, digest = runtime.read, runtime.write, runtime.digest


class SDKDrive:
    """Use the already authenticated Second-account SDK without changing it."""
    def __init__(self, folder):
        from storage_sdk import Drive
        self.api = Drive()
        self.folder = 'hfh_storage:'+folder
        self.api.resolve(self.folder, create=True)

    def call(self, *args):
        if args[0] == 'about':
            self.api.quota = self.api.service.about().get(fields='storageQuota').execute(num_retries=5)['storageQuota']
        return self.api.command(*args)

    def upload(self, path):
        self.api.upload(path, self.folder+'/'+path.name)

    def previous_state(self, destination):
        try:
            self.api.download(self.folder+'/status.json', destination)
        except FileNotFoundError:
            return None
        return read(destination)


def check_prior_attempt(local, remote):
    """A replacement VM must not silently reset an already charged attempt."""
    if remote is not None and (local is None or local.get('run_id') != remote.get('run_id')):
        raise RuntimeError('Prior remote attempt exists; restore and audit its charged work before proceeding')
    if local is not None and not local.get('run_id'):
        raise ValueError('Local attempt has no identity; inspect before resuming')


def check_campaign_completion(unetr, medsam):
    if unetr.get('status') != 'complete' or unetr.get('unetr_target_pairs') != 9 or unetr.get('three_model_verified_pairs') != 378:
        raise ValueError('UNETR completion not verified')
    expected = {(c,a) for c in ('noise','gain') for a in ('dice_bands','dice_edge_depth')}
    rows = medsam.get('results', [])
    if (medsam.get('status') != 'complete_4_component_constraints' or medsam.get('workers') or
            len(rows) != 4 or {(r['component'],r['arm']) for r in rows} != expected or
            any((r['fold'],r['seed'],r['updates']) != (0,0,125) for r in rows)):
        raise ValueError('MedSAM is not complete with all four archived outcomes')


def check_no_workers():
    import psutil
    busy = []
    for process in psutil.process_iter(['pid','cmdline','status']):
        if process.info['status'] == 'zombie' or process.pid == os.getpid():
            continue
        command = ' '.join(process.info['cmdline'] or [])
        if any(name in command for name in (
                'medsam_component_constraints_campaign', 'medsam_augmentation_components_worker',
                'complete_missing.py', 'resume_three_models.py', 'nnunet_composition_worker',
                'scripts.colab_hfh.constraint_campaign', 'scripts.colab_hfh.constraint_worker')):
            busy.append(process.pid)
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    if busy or gpu:
        raise RuntimeError('Training/runtime processes still active; no takeover: '+str(busy))


def verify_package(package):
    manifest = read(package/'manifest.json')
    for relative, expected in manifest.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or digest(package/path) != expected:
            raise ValueError('Package mismatch: '+relative)
    return manifest


def stage_archive(output, durable, protocol, controls):
    p = read(protocol/'protocol.json')
    c = read(output/'config.json')
    n = read(controls/'native/results/config.json')
    for key in ('fold','seed','dataset_sha256','train_cases','validation_cases','precision',
                'initial_lr','lr_horizon_updates','nnunet_warmup_fraction','nnunet_sgd_momentum','augmentation','native_loss'):
        if c[key] != n[key]:
            raise ValueError('Native matching failed: '+key)
    if read(output/'initialization.json') != read(controls/'native/results/initialization.json'):
        raise ValueError('Initialization differs')
    done = read(output/'completion.json')
    arm = p['active_objective']
    schedule = p['models']['nnunet']['objectives'][arm]
    used = training_flops(done['updates'], schedule)
    if (done['status'] != 'complete' or done['discarded_attempts'] or done['updates'] != 179 or
            done['updates'] != schedule['max_steps'] or done['consumed_training_flops'] != used or
            not used <= CAP < training_flops(done['updates']+1, schedule) or
            done['calibration_flops'] != schedule['calibration_flops'] or done['calibration_attempts'] != 1):
        raise ValueError('Endpoint or budget mismatch')
    cal = read(output/'calibration.json')
    if cal['anchor_successful_updates'] != 60 or not cal.get('composition_anchor_verified'):
        raise ValueError('Native prefix was not verified')
    check_anchor(cal, p['composition'])
    val = read(output/'validation_latest.json')
    train = read(output/'training_final.json')
    dice = val['summary']['macro_hard_dice']
    comparisons = {}
    for name in ('native','band_edge','band_edge_depth'):
        original = read(controls/name/'results/validation_latest.json')
        comparisons[name] = 100*(dice-original['summary']['macro_hard_dice'])
    equal = read(controls/'native/results/validation_update179.json')
    by_case = {r['case_name']:r for r in read(controls/'native/results/validation_latest.json')['cases']}
    deltas = [100*(r['macro_hard_dice']-by_case[r['case_name']]['macro_hard_dice']) for r in val['cases']]
    row = dict(variant=p['composition']['variant'], arm=arm, fold=0, seed=0, updates=done['updates'],
        validation_dice=dice, training_dice=train['summary']['macro_hard_dice'],
        gains_pp=comparisons, equal_update_native_gain_pp=100*(dice-equal['summary']['macro_hard_dice']),
        improved_crops=sum(d>0 for d in deltas), worst_crop_pp=min(deltas),
        weights=cal['weights'], optimization_flops=used, optimization_cap=CAP,
        calibration_flops=done['calibration_flops'], active_updates=119, matched_prefix_verified=True,
        ap_swaps_applicable=False)
    for split, values in (('validation',val),('training',train)):
        for error in ('fp','fn'):
            row[split+'_'+error] = sum(r['foreground_'+error] for r in values['cases'])
    write(output/'comparison.json', row)
    # Retain all standard completion/metric/split checks, but defer deletion until
    # the zipped archive has been independently verified on Drive.
    source = inspect.getsource(runtime.archive)
    old = "schedule = p['models'][p['trainer_alias']]['objectives']['dice']"
    if source.count(old) != 1 or source.count('    deleted = []') != 1:
        raise ValueError('Archive integration changed')
    source = source.replace(old, "schedule = p['models'][p['trainer_alias']]['objectives'][p['active_objective']]")
    source = source[:source.index('    deleted = []')]+'    return result\n'
    ns = dict(runtime.__dict__)
    exec(compile(source, __file__, 'exec'), ns)
    ns['archive'](output,durable,'baselines','nnunet',0,0,protocol)
    return row


def retire(output, durable):
    done = read(output/'completion.json')
    pending = []
    for folder in (output,durable):
        for path in folder.glob('checkpoint_*.pt'):
            key = 'checkpoint_latest_sha256' if path.name == 'checkpoint_latest.pt' else 'checkpoint_best_sha256'
            sha = digest(path)
            if sha != done[key]:
                raise ValueError('Checkpoint changed; retained')
            pending.append((path,sha,path.stat().st_size))
    marker = durable/'recovery/latest.json'
    if marker.exists():
        value = read(marker)
        if Path(value['file']).name != value['file']:
            raise ValueError('Unsafe recovery filename')
        path = marker.parent/value['file']
        if digest(path) != value['sha256']:
            raise ValueError('Recovery changed; retained')
        pending.append((path,value['sha256'],path.stat().st_size))
    # Every prospective deletion is verified before the first deletion occurs.
    deleted = []
    for path,sha,size in pending:
        path.unlink()
        deleted.append(dict(path=str(path),sha256=sha,bytes=size))
    if marker.exists():
        marker.unlink()
    write(durable/'cleanup.json',dict(status='complete',verified_remote_before_deletion=True,
                                    verified_at=time.time(),deleted=deleted))


def zip_archive(path, durable, protocol, package):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for directory,prefix in ((durable,'results'),(protocol,'protocol'),(package/'source','source')):
            for item in sorted(directory.rglob('*')):
                if item.is_file() and 'recovery' not in item.parts and item.suffix not in ('.pt','.pyc'):
                    archive.write(item,prefix+'/'+str(item.relative_to(directory)))
        archive.write(package/'manifest.json','package_manifest.json')
        archive.write(path.parent/'execution_source_manifest.json','execution_source_manifest.json')
        archive.write(path.parent/'launch_provenance.json','launch_provenance.json')
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('results/archive_manifest.json'))
        for name,sha in manifest.items():
            if hashlib.sha256(archive.read('results/'+name)).hexdigest() != sha:
                raise ValueError('Zipped result verification failed')


def run(root, package, clearance):
    verify_package(package)
    gate = read(clearance)
    if time.time()-gate['checked_at'] > 600 or gate['checked_at'] > time.time()+30:
        raise ValueError('Completion clearance is stale')
    check_campaign_completion(read(package/'unetr_verification.json'),gate['medsam'])
    root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_no_workers()
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
        if gpu != 'NVIDIA L4':
            raise RuntimeError('Exact L4 native comparison required; no silent hardware change')
        drive = SDKDrive('results_'+NAME)
        quota = json.loads(drive.call('about','hfh_storage:','--json'))
        if min(quota.get('free',0),shutil.disk_usage(root).free) < 8*1024**3:
            raise RuntimeError('Need 8 GiB local and Second-account Drive headroom')
        write(root/'storage_preflight.json',dict(drive=quota,local_free=shutil.disk_usage(root).free,
            peak_generated_estimate_bytes=4*1024**3,checked_at=time.time(),gpu=gpu))
        controls = package/'controls'
        for name in ('native','band_edge','band_edge_depth'):
            runtime.verify_archive(controls/name/'results')
        if read(controls/'native/results/readiness_model.json') != ANCHOR:
            raise ValueError('Native anchor reference changed')
        source = root/'bundle/source'
        for relative,sha in read(root/'execution_source_manifest.json').items():
            if digest(source/relative) != sha:
                raise ValueError('Isolated execution source changed: '+relative)
        state_path = root/'status.json'
        previous = read(state_path) if state_path.exists() else None
        check_prior_attempt(previous,drive.previous_state(root/'previous_remote_status.json'))
        state = previous or dict(status='ready',results=[],arms=list(ARMS),run_id=uuid.uuid4().hex)
        write(state_path,state)
        drive.upload(root/'execution_source.zip')
        drive.upload(root/'launch_provenance.json')
        env = dict(os.environ, PYTHONPATH=':'.join([str(root)]+[str(root/'bundle'/p) for p in
            ('compatibility','source','source/scripts','source/baselines/nnUNet')]),
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',
            HIPPO_HFH_GPU_CACHE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        try:
            for variant in ARMS:
                output, durable = root/variant/'run', root/variant/'archive'
                protocol = package/'protocols'/variant
                output.mkdir(parents=True,exist_ok=True);durable.mkdir(parents=True,exist_ok=True)
                if (durable/'cleanup.json').exists():
                    runtime.verify_archive(durable)
                    row = read(durable/'comparison.json')
                else:
                    check_no_workers()
                    if (state.get('current') == variant and not (output/'budget_ledger.json').exists()
                            and not (output/'completion.json').exists()):
                        raise RuntimeError('Prior attempt has no local ledger; charged-work audit required')
                    state.update(status='running',current=variant);write(state_path,state);drive.upload(state_path)
                    if not (output/'completion.json').exists():
                        write(output/'execution_environment.json',dict(gpu=gpu))
                        with (output/'worker.log').open('a') as log:
                            subprocess.run([sys.executable,'-m','scripts.colab_hfh.nnunet_composition_worker',
                                '--protocol-dir',str(protocol),'--output',str(output),'--durable',str(durable),
                                '--pkl',str(root/'bundle/data_index.json'),'--fold','0','--seed','0'],
                                cwd=source,env={**env,'HIPPO_DEADLINE':str(time.time()+3600)},
                                stdout=log,stderr=subprocess.STDOUT,check=True)
                    row = stage_archive(output,durable,protocol,controls)
                    zipped = root/(variant+'.zip')
                    zip_archive(zipped,durable,protocol,package)
                    drive.upload(zipped)  # verifies persisted size/checksum before deletion
                    retire(output,durable)
                zipped = root/(variant+'.zip')
                zip_archive(zipped,durable,protocol,package);drive.upload(zipped)
                state['results'] = [r for r in state['results'] if r['variant'] != variant]+[row]
                state.update(status='archived',current=None);write(state_path,state);drive.upload(state_path)
                print('COMPOSITION_ARCHIVED',json.dumps(row),flush=True)
            state.update(status='complete_two_arms',current=None,
                         storage_after=json.loads(drive.call('about','hfh_storage:','--json')))
            write(state_path,state);drive.upload(state_path)
        except BaseException as error:
            state.update(status='error_no_expansion',error=str(error))
            write(state_path,state)
            try:drive.upload(state_path)
            except Exception:pass
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root','package','clearance'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    run(args.root,args.package,args.clearance)
