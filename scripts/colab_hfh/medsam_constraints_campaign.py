"""Sequential BAND exploration followed by a strictly gated 108-run matrix."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import zipfile

from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh.medsam_recovery import copy_verified, rclone, remote_path
from scripts.colab_hfh.runtime import read, write, digest
from scripts.colab_hfh.medsam_constraints_policy import ARMS, PAIRS, RECIPES, STRICT_RECIPES, STRICT_CONTROL, make_protocol, qualifies, matrix_tasks


def zip_read(path, name):
    with zipfile.ZipFile(path) as stream:
        return json.loads(stream.read('results/'+name))


def archived_control(path):
    archives.verify_zip(path)
    control = {name: zip_read(path, name+'.json') for name in
               ('protocol', 'training_history', 'slab_schedule', 'training_summary',
                'completion', 'evaluation', 'training_inventory', 'environment')}
    control['archive_sha256'] = digest(path)
    return control


def read_control(root, seed, fold, inventory):
    name = f'medsam3_f{fold}_s{seed}.zip'
    row = inventory[name]
    local = root/'work_medsam3/compact_archives'/name
    local.parent.mkdir(parents=True, exist_ok=True)
    expected = {k.lower(): v for k, v in row['Hashes'].items()}['md5']
    if not local.exists() or hashlib.md5(local.read_bytes()).hexdigest() != expected:
        rclone('copyto', 'hfh_storage:results_medsam3_v1/compact_archives/'+name, str(local), '--checksum')
    if hashlib.md5(local.read_bytes()).hexdigest() != expected:
        raise ValueError('Control archive remote hash mismatch')
    result = archives.verify_zip(local)
    if (result['seed'], result['fold'], result['updates']) != (seed, fold, 200):
        raise ValueError('Wrong control identity')
    control = {name: zip_read(local, name+'.json') for name in
               ('protocol', 'training_history', 'slab_schedule', 'training_summary',
                'completion', 'evaluation', 'training_inventory', 'environment')}
    control['archive_sha256'] = digest(local)
    if control['completion']['validation_dice'] < .5:
        raise ValueError('Control does not qualify')
    return control


def paired_result(output, protocol, control, arm, recipe):
    done, actual = read(output/'completion.json'), read(output/'protocol.json')
    if actual != {**protocol, 'arm': arm} or done['protocol_sha256'] != digest(output/'protocol.json'):
        raise ValueError('Constraint protocol binding changed')
    if done['arm'] != arm or done['updates'] != 200 or done['primary_checkpoint'] != 'final_update':
        raise ValueError('Incomplete final-budget constraint run')
    history = read(output/'training_history.json')
    if len(history) != 200 or [r['lr'] for r in history] != protocol['frozen_learning_rates']:
        raise ValueError('Learning-rate trajectory differs from control')
    if read(output/'slab_schedule.json') != protocol['frozen_sample_schedule']:
        raise ValueError('Sample trajectory differs from control')
    training = read(output/'training_summary.json')
    for key in ('frozen_before', 'adapter_before', 'schedule_sha256'):
        if training[key] != control['training_summary'][key]:
            raise ValueError('Paired initialization/sampling differs: '+key)
    if training['frozen_before'] != training['frozen_after'] or training['adapter_before'] == training['adapter_after']:
        raise ValueError('Invalid adapter-only training')
    if read(output/'training_inventory.json') != control['training_inventory']:
        raise ValueError('Training images or annotations changed')
    if read(output/'environment.json')['torch'] != control['environment']['torch']:
        raise ValueError('PyTorch changed since control')
    calibration = read(output/'calibration.json')
    active = {name for name in ('bands', 'edge', 'sdf', 'depth') if name in arm}
    if not active:
        if calibration['status'] != 'not_required' or any(calibration['weights'].values()):
            raise ValueError('Native control unexpectedly has calibrated constraints')
    elif calibration['status'] != 'complete' or set(calibration['statistics']) != active:
        raise ValueError('Incomplete active-term calibration')
    if active and len(calibration['rows']) != 4 * len(protocol['train_cases']):
        raise ValueError('Calibration does not cover all support crops')
    if protocol.get('training_numerics') == 'strict_empty_grid_v1':
        if not read(output/'runtime_validation.json')['deterministic_algorithms']:
            raise ValueError('Strict campaign ran without deterministic algorithms')
        if arm != 'dice' and control['protocol'].get('training_numerics') != 'strict_empty_grid_v1':
            raise ValueError('Strict constraints require fresh matching strict controls')
    for name in active:
        weight = calibration['weights'][name]
        if not math.isfinite(weight) or weight <= 0 or calibration['statistics'][name]['weighted_p95'] > .50000001:
            raise ValueError('Invalid calibrated weight')
    evaluation = read(output/'evaluation.json')
    if digest(output/'evaluation.json') != done['evaluation_sha256']:
        raise ValueError('Evaluation hash differs')
    expected = {(split, case) for split, names in
                [('training_support', protocol['train_cases']), ('validation', protocol['evaluation_cases'])]
                for case in names}
    records = evaluation['records']
    if len(records) != 20 or {(r['split'], r['case_name']) for r in records} != expected:
        raise ValueError('Incomplete paired evaluation')
    for split, key in [('training_support', 'training_dice'), ('validation', 'validation_dice')]:
        values = [r['macro_hard_dice'] for r in records if r['split'] == split]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError('Nonfinite Dice')
        if not math.isclose(statistics.mean(values), done[key], abs_tol=1e-12):
            raise ValueError('Case scores disagree with completion')
    reference = control['completion']
    row = dict(seed=protocol['seed'], fold=protocol['fold'], arm=arm, recipe=recipe['name'],
               updates=200, training_dice=done['training_dice'], validation_dice=done['validation_dice'],
               baseline_training_dice=reference['training_dice'], baseline_dice=reference['validation_dice'],
               delta_dice_pp=100*(done['validation_dice']-reference['validation_dice']),
               weights=calibration['weights'], baseline_archive_sha256=control['archive_sha256'],
               calibration_pairs=len(calibration.get('rows', [])),
               training_seconds=training['elapsed_seconds'],
               evaluation_scope='exploratory_validation_selection', optimization_flops=None,
               budget='200 matched optimizer updates; auxiliary/calibration overhead is not FLOP matched')
    if recipe.get('delay', 0):
        from scripts.colab_hfh.medsam_constraints_schedule import constraint_ramp
        if [h['ramp'] for h in history] != [constraint_ramp(i, protocol) for i in range(200)]:
            raise ValueError('Delayed activation trajectory differs from declared schedule')
        prefix = history[:recipe['delay']]
        if any(h['total'] != h['native'] for h in prefix):
            raise ValueError('Constraint affected the native-only prefix objective')
        reference_history = control['training_history'][:recipe['delay']]
        row['native_prefix_audit'] = dict(updates=len(prefix), objective_verified=True,
            max_native_loss_difference=max(abs(a['native']-b['native'])
                                          for a,b in zip(prefix,reference_history)),
            max_gradient_norm_difference=max(abs(a['gradient_norm']-b['gradient_norm'])
                                            for a,b in zip(prefix,reference_history)))
    for key in ('foreground_fp', 'foreground_fn'):
        row[key] = sum(r[key] for r in records if r['split'] == 'validation')
        row['baseline_'+key] = sum(r[key] for r in control['evaluation']['records'] if r['split'] == 'validation')
    return row


class Publisher:
    """Keep network I/O off the worker-supervision path."""
    def __init__(self, work, durable):
        self.work, self.durable = work, durable
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def snapshot(self):
        state = read(self.work/'campaign_status.json')
        active = state.get('active')
        if active:
            output = self.work/'runs'/active['name']
            for name in ('checkpoint_ledger', 'calibration_partial', 'completion'):
                path = output/(name+'.json')
                if path.exists():
                    item = read(path)
                    if name == 'calibration_partial':
                        item = dict(rows=len(item['rows']), names=item['names'])
                    active[name] = item
            log = output/'worker.log'
            if log.exists():
                active['log_tail'] = log.read_text(errors='replace')[-4500:]
        state['observed_at'] = time.time()
        path = self.work/'live_status.json'
        write(path, state)
        copy_verified(path, self.durable/'monitor/live_status.json')

    def loop(self):
        while not self.stop.is_set():
            try:
                self.snapshot()
            except Exception as error:
                print('STATUS_UPLOAD_RETRY', type(error).__name__, str(error)[-400:], flush=True)
            self.stop.wait(90)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.snapshot()
        copy_verified(self.work/'campaign_status.json', self.durable/'monitor/campaign_status.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/content/hippo_hfh'))
    parser.add_argument('--hours', type=float, default=20)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    if not 0 < args.hours <= 22:
        parser.error('Session limit must be in (0,22] hours')
    root, bundle = args.root, args.root/'bundle'
    campaign = 'medsam_constraints_strict_v1' if args.strict else 'medsam_constraints_v1'
    work = root/campaign
    durable = Path('/content/hfh_storage')/('results_'+campaign.replace('medsam_', 'medsam3_', 1))
    recipes = STRICT_RECIPES if args.strict else RECIPES
    if args.strict:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    work.mkdir(exist_ok=True); durable.mkdir(exist_ok=True)
    locks = []
    for path in (work/'controller.lock', root/'work_medsam3/controller.lock'):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open('a'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    if shutil.disk_usage(root).free < 8*1024**3:
        raise RuntimeError('Need 8 GiB free local storage')
    quota = json.loads(rclone('about', 'hfh_storage:', '--json'))
    if quota.get('free', 8*1024**3) < 8*1024**3:
        raise RuntimeError('Need 8 GiB free second-account storage')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This paired BF16 campaign needs a BF16-capable GPU')
    gpu = torch.cuda.get_device_name()
    # Controls were trained on this exact GPU family. A changed GPU needs new controls.
    if gpu != 'NVIDIA A100-SXM4-40GB':
        raise RuntimeError('Controls used A100-SXM4-40GB; rerun matched controls before comparing on '+gpu)
    inventory = {r['Name']: r for r in json.loads(rclone('lsjson',
        'hfh_storage:results_medsam3_v1/compact_archives', '--files-only', '--hash'))}
    controls = {(seed, fold): read_control(root, seed, fold, inventory) for seed, fold in PAIRS}
    archives.configure(bundle, work, durable)
    state = dict(status='running_controls' if args.strict else 'running_bands', gpu=gpu,
                 recipes=list(recipes), results=[], controls=[], rejected=[],
                 gate='All nine paired validation gains strictly positive; every Dice >=50%',
                 matrix_arms=list(ARMS), matrix_additional_runs=108, winner=None,
                 warning='Validation-selected exploratory study. Preserve all attempts; no test-set claim.')
    write(work/'campaign_status.json', state)
    publisher = Publisher(work, durable)
    deadline = time.time()+args.hours*3600
    stop, process = False, None
    def interrupt(*unused):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupt); signal.signal(signal.SIGTERM, interrupt)
    def save():
        write(work/'campaign_status.json', state)
    def stopping():
        return stop or time.time() >= deadline or (work/'STOP').exists()

    def run(recipe, arm, seed, fold):
        nonlocal process
        name = f"{recipe['name']}__{arm}__f{fold}_s{seed}"
        output = work/'runs'/name
        remote_run = durable/'runs'/name
        output.mkdir(parents=True, exist_ok=True)
        control = controls[seed, fold]
        p = make_protocol(control['protocol'], control['slab_schedule'], control['training_history'],
                          recipe, arm, control['archive_sha256'])
        protocol_path = work/'protocols'/(name+'.json')
        if protocol_path.exists() and read(protocol_path) != p:
            raise ValueError('Existing campaign protocol changed')
        write(protocol_path, p)
        zip_name = name+'.zip'
        if zip_name in archives.CONTEXT['inventory']:
            remote = archives.CONTEXT['inventory'][zip_name]
            local = archives.CONTEXT['cache']/zip_name
            md5 = {k.lower(): v for k, v in remote['Hashes'].items()}['md5']
            if not local.exists() or hashlib.md5(local.read_bytes()).hexdigest() != md5:
                rclone('copyto', remote_path(archives.CONTEXT['remote']/zip_name), str(local), '--checksum')
            if hashlib.md5(local.read_bytes()).hexdigest() != md5:
                raise ValueError('Completed archive differs')
            archives.verify_zip(local)
            if zip_read(local, 'protocol.json') != {**p, 'arm': arm}:
                raise ValueError('Completed arm protocol differs')
            return zip_read(local, 'paired_result.json')
        if stopping():
            raise SystemExit(75)
        state['active'] = dict(name=name, recipe=recipe['name'], arm=arm, seed=seed, fold=fold)
        save()
        if not (output/'completion.json').exists():
            remote_run.mkdir(parents=True, exist_ok=True)
            provenance = {str(path.relative_to(bundle/'source')): path.read_text()
                          for path in sorted((bundle/'source/scripts/colab_hfh').glob('medsam_constraints_*.py'))}
            for relative in ('scripts/cross_dataset_campaign/foundation.py',
                             'scripts/cross_dataset_campaign/adni_foundation_full_runtime.py'):
                provenance[relative] = (bundle/'source'/relative).read_text()
            write(output/'constraint_sources.json', provenance)
            env = dict(os.environ, HIPPO_DEADLINE=str(deadline),
                       MNI_MANIFEST_SHA=digest(bundle/'manifest.json'), HFH_CONTROLLER_ARCHIVES='1')
            command = [sys.executable, '-m', 'scripts.colab_hfh.medsam_constraints_worker',
                       '--protocol', str(protocol_path), '--output', str(output), '--durable', str(remote_run),
                       '--base-weights', str(root/'sam3.pt'), '--arm', arm]
            (output/'supervisor_ready.json').unlink(missing_ok=True)
            print('START', name, flush=True)
            with (output/'worker.log').open('a') as log:
                process = subprocess.Popen(command, cwd=bundle/'source', env=env, stdout=log, stderr=subprocess.STDOUT)
                sent = False
                last_report = 0.
                while process.poll() is None:
                    if stopping() and not sent and (output/'supervisor_ready.json').exists():
                        process.send_signal(signal.SIGUSR1); sent = True
                    if time.monotonic()-last_report >= 60:
                        last_report = time.monotonic()
                        ledger, calibration = output/'checkpoint_ledger.json', output/'calibration_partial.json'
                        if ledger.exists():
                            print('PROGRESS', name, 'saved_updates', read(ledger)['step'], flush=True)
                        elif calibration.exists():
                            print('PROGRESS', name, 'calibration_slabs', len(read(calibration)['rows']), '/40', flush=True)
                        else:
                            print('PROGRESS', name, 'loading/initial loss checks', flush=True)
                    try: process.wait(timeout=2)
                    except subprocess.TimeoutExpired: pass
            if process.returncode == 75:
                raise SystemExit(75)
            if process.returncode:
                raise RuntimeError(f'{name} exited {process.returncode}: '+(output/'worker.log').read_text()[-3500:])
        paired = paired_result(output, p, control, arm, recipe)
        write(output/'paired_result.json', paired)
        archives.archive(output, remote_run, 'medsam3', 'medsam3', fold, seed, protocol_path)
        print('PAIRED_FINAL', name, json.dumps(paired), flush=True)
        return paired

    try:
        if args.strict:
            for seed, fold in PAIRS:
                row = run(STRICT_CONTROL, 'dice', seed, fold)
                state['controls'].append(row); state.pop('active', None); save()
                if row['validation_dice'] < .5:
                    state['status'] = 'control_recipe_failed'; save()
                    print('STRICT_CONTROL_FAILED: revise native training before BAND.', flush=True)
                    return
                name = f"{STRICT_CONTROL['name']}__dice__f{fold}_s{seed}.zip"
                controls[seed, fold] = archived_control(archives.CONTEXT['cache']/name)
            state['status'] = 'running_bands'; save()
        winner = None
        for index, recipe in enumerate(recipes):
            rows = []
            for seed, fold in PAIRS:
                row = run(recipe, 'dice_bands', seed, fold)
                rows.append(row); state['results'].append(row); state.pop('active', None); save()
                # Once any pair fails, this recipe cannot satisfy the all-nine gate.
                # Initial canonical pilot changed to fail-fast after its first
                # verified -5.10 pp result; all completed evidence is retained.
                if row['delta_dice_pp'] <= 0 or row['validation_dice'] < .5:
                    break
            if qualifies(rows):
                winner = recipe; state['winner'] = recipe; break
            state['rejected'].append(dict(recipe=recipe, completed_pairs=len(rows),
                reason='At least one pair failed strict improvement or 50% threshold', results=rows))
            save()
        if winner is None:
            state['status'] = 'needs_next_hypothesis'; save()
            print('BAND_BANK_EXHAUSTED: full matrix withheld; inspect preserved outcomes.', flush=True)
            return
        state['status'] = 'running_full_matrix'; save()
        for recipe, arm, seed, fold in matrix_tasks(winner):
            row = run(recipe, arm, seed, fold)
            state['results'].append(row); state.pop('active', None); save()
        state['status'] = 'complete_full_matrix'; save()
    except SystemExit as error:
        if error.code != 75: raise
        state['status'] = 'paused'; save()
        print('PAUSED: verified recovery retained. Resume the same campaign.', flush=True)
    except Exception as error:
        state.update(status='error', error_type=type(error).__name__, error=str(error)[-4000:]); save()
        raise
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGUSR1)
            process.wait()
        publisher.close()


if __name__ == '__main__':
    main()
