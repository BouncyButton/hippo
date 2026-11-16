"""Bounded, validation-selected stopping-budget pilot; original campaign untouched."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_for(reference, updates):
    if reference['updates'] != 200 or reference['fold'] != 0 or reference['seed'] != 0:
        raise ValueError('Expected original 200-update fold0 seed0 control')
    if not reference['constraint_recipe'].get('native_control') or not 0 < updates < 200:
        raise ValueError('Only a shorter native control is allowed')
    result = copy.deepcopy(reference)
    result.pop('arm', None)
    result['updates'] = updates
    result['learning_rate_horizon_updates'] = 200
    result['frozen_sample_schedule'] = reference['frozen_sample_schedule'][:updates * reference['accumulation']]
    # Retain the complete 200-entry LR tape required by the unchanged worker;
    # only its prefix is consumed. This is early stopping, not a faster decay.
    result['comparison'] = 'Exploratory stopping-budget selection near 70% validation Dice; fixed original LR/sample prefix'
    return result


def verify_prefix(source, target, step):
    allowed = {'updates', 'frozen_sample_schedule', 'arm'}
    if {k:v for k,v in source.items() if k not in allowed} != {k:v for k,v in target.items() if k not in allowed}:
        raise ValueError('Checkpoint import changes scientific recipe')
    n = step * target['accumulation']
    if (step > target['updates'] or step > source['updates']
            or source['frozen_sample_schedule'][:n] != target['frozen_sample_schedule'][:n]):
        raise ValueError('Checkpoint import changes consumed samples')


def retain_checkpoint(path, state):
    """Called after an ordinary save, without touching training or random state."""
    if state['step'] % 25:
        return
    folder = Path(os.environ['HFH_BUDGET_CHECKPOINTS'])
    destination = folder / f"u{state['step']:03d}.pt"
    if not destination.exists():
        shutil.copyfile(path, destination)
        shutil.copyfile(Path(path).parent/'protocol.json', destination.with_suffix('.json'))
        write(destination.with_suffix('.receipt.json'), dict(sha256=digest(destination), step=state['step']))


def worker():
    from scripts.colab_hfh.medsam_constraints_worker import worker_source
    source = worker_source()
    anchor = '        save(path, state)\n'
    if source.count(anchor) != 1:
        raise ValueError('Audited checkpoint hook changed')
    source = source.replace(anchor, anchor + '        retain_checkpoint(path, state)\n')
    source = source.replace('Native plus calibrated constraints; exact paired samples/LR; 200 updates',
                            'Native-only stopping-budget pilot; exact original sample/LR prefix')
    namespace = dict(__name__='__main__', __file__=__file__, retain_checkpoint=retain_checkpoint)
    sys.argv.remove('--worker')
    exec(compile(source, __file__, 'exec'), namespace)


def main():
    import psutil
    from scripts.colab_hfh.medsam_recovery import copy_verified
    root = Path('/content/hippo_hfh')
    work = root/'medsam_budget_pilot_f0s0_v1'
    remote = Path('/content/hfh_storage/results_medsam3_budget_pilot_f0s0_v1')
    monitor = Path('/content/hfh_storage/results_medsam3_constraints_strict_v1/monitor/short_budget_status.json')
    work.mkdir(exist_ok=True)
    (work/'checkpoints').mkdir(exist_ok=True)
    locks = []
    for path in (root/'medsam_constraints_strict_v1/controller.lock', root/'work_medsam3/controller.lock'):
        handle = path.open('a'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    for process in psutil.process_iter(['cmdline']):
        if 'scripts.colab_hfh.medsam_constraints_worker' in (process.info['cmdline'] or []):
            raise RuntimeError('BAND worker still active; wait for checkpoint-safe stop')
    if not (root/'medsam_constraints_strict_v1/STOP').exists():
        raise RuntimeError('Original BAND campaign must remain stopped')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True).strip()
    if gpu != 'NVIDIA A100-SXM4-40GB' or shutil.disk_usage(root).free < 8*1024**3:
        raise RuntimeError('Original A100 and 8GiB free disk required')
    reference = read(root/'medsam_constraints_strict_v1/protocols/strict_native__dice__f0_s0.json')
    state = read(work/'status.json') if (work/'status.json').exists() else dict(
        status='running', results=[], target_validation_dice=[.68,.72], maximum_evaluations=5,
        seed=0, fold=0, baseline_200_validation_dice=.7743168448729005,
        interpretation='Validation-selected stopping-budget pilot, not evidence of improved generalization or constraint benefit')
    if state.get('status') == 'complete':
        print(json.dumps(state), flush=True); return
    state.pop('error', None)
    lower, upper = 0, 200
    count = 100
    child = None

    def publish():
        state['observed_at'] = time.time()
        write(work/'status.json', state)
        copy_verified(work/'status.json', monitor)

    try:
        for attempt in range(5):
            name = f'native_stop_u{count:03d}_f0_s0'
            output = work/'runs'/name
            output.mkdir(parents=True, exist_ok=True)
            protocol = protocol_for(reference, count)
            path = work/'protocols'/(name+'.json')
            if path.exists() and read(path) != protocol:
                raise ValueError('Pilot protocol changed')
            write(path, protocol)
            checkpoint = output/'checkpoint_latest.pt'
            if not checkpoint.exists() and not (output/'completion.json').exists():
                candidates = sorted(p for p in (work/'checkpoints').glob('u*.pt') if int(p.stem[1:]) <= count)
                if candidates:
                    import torch
                    source = candidates[-1]
                    previous = read(source.with_suffix('.json'))
                    saved = torch.load(source, map_location='cpu', weights_only=False)
                    if (digest(source) != read(source.with_suffix('.receipt.json'))['sha256']
                            or saved['protocol_sha256'] != digest(source.with_suffix('.json'))
                            or len(saved['history']) != saved['step']
                            or [h['lr'] for h in saved['history']] != protocol['frozen_learning_rates'][:saved['step']]):
                        raise ValueError('Retained checkpoint provenance differs')
                    verify_prefix(previous, protocol, saved['step'])
                    write(output/'protocol.json', {**protocol, 'arm':'dice'})
                    original_hash = saved['protocol_sha256']
                    saved['protocol_sha256'] = digest(output/'protocol.json')
                    torch.save(saved, checkpoint)
                    write(output/'checkpoint_ledger.json', dict(step=saved['step'],
                        protocol_sha256=saved['protocol_sha256']))
                    write(output/'checkpoint_import.json', dict(source=str(source), source_sha256=digest(source),
                        original_protocol_sha256=original_hash, destination_protocol_sha256=saved['protocol_sha256'],
                        restored_updates=saved['step'], reason='Explicit stopping-budget branch; model, optimizer and RNG unchanged'))
                    del saved
            if not (output/'completion.json').exists():
                env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', HFH_CONTROLLER_ARCHIVES='1',
                    HFH_BUDGET_CHECKPOINTS=str(work/'checkpoints'),
                    MNI_MANIFEST_SHA=digest(root/'bundle/manifest.json'), HIPPO_DEADLINE=str(time.time()+3*3600))
                with (output/'worker.log').open('a') as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
                        '--protocol',str(path),'--output',str(output),'--durable',str(remote/'runs'/name),
                        '--base-weights',str(root/'sam3.pt'),'--arm','dice'],env=env,
                        cwd=root/'bundle/source',stdout=log,stderr=subprocess.STDOUT)
                last = 0
                while child.poll() is None:
                    if time.monotonic()-last > 60:
                        ledger = read(output/'checkpoint_ledger.json') if (output/'checkpoint_ledger.json').exists() else {}
                        state.update(status='running', active=dict(updates=count,pid=child.pid,step=ledger.get('step',0),
                            phase='evaluation' if (output/'training_summary.json').exists() else 'training',
                            log_tail=(output/'worker.log').read_text(errors='replace')[-1500:]))
                        publish(); last=time.monotonic()
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError('Pilot worker failed: '+(output/'worker.log').read_text()[-3000:])
                child = None
            done = read(output/'completion.json')
            # Repair an evaluation-only import created by the first pilot
            # version: evaluation completed, but final recovery lacked a ledger.
            # Validate the completed artifact before synthesizing that sidecar;
            # no model loading/evaluation/training is repeated.
            if not (output/'checkpoint_ledger.json').exists():
                import torch
                from scripts.colab_hfh.medsam_recovery import RecoveryWriter
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if (saved['step'] != count or len(saved['history']) != count
                        or saved['protocol_sha256'] != digest(output/'protocol.json')
                        or done['checkpoint_sha256'] != digest(checkpoint)):
                    raise ValueError('Cannot repair an unverified completed checkpoint ledger')
                write(output/'checkpoint_ledger.json', dict(step=count,protocol_sha256=saved['protocol_sha256']))
                del saved
                recovery = RecoveryWriter(output, remote/'runs'/name/'recovery', interval=180)
                recovery.close(force=True)
                write(output/'pilot_recovery_repair.json', dict(status='verified',optimizer_updates_performed=0,
                    evaluation_repeated=False, reason='Created missing imported-checkpoint ledger; uploaded recovery'))
            history = read(output/'training_history.json')
            if (done['updates'] != count or len(history) != count
                    or [h['lr'] for h in history] != protocol['frozen_learning_rates'][:count]
                    or read(output/'slab_schedule.json') != protocol['frozen_sample_schedule']
                    or done['checkpoint_sha256'] != digest(checkpoint)
                    or done['evaluation_sha256'] != digest(output/'evaluation.json')
                    or any(read(output/'calibration.json')['weights'].values())):
                raise ValueError('Pilot completion differs from frozen native prefix')
            row = {k:done[k] for k in ('updates','training_dice','validation_dice','checkpoint_sha256')}
            state['results'] = [r for r in state['results'] if r['updates'] != count] + [row]
            archive = work/(name+'.zip')
            files = [p for p in output.iterdir() if p.suffix in ('.json','.log','.csv')]
            manifest = {p.name:digest(p) for p in files}
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
                for p in files:zipped.write(p,'results/'+p.name)
                zipped.writestr('manifest.json',json.dumps(manifest))
                zipped.write(__file__,'source/medsam_budget_pilot.py')
            with zipfile.ZipFile(archive) as zipped:
                assert all(hashlib.sha256(zipped.read('results/'+n)).hexdigest()==h for n,h in manifest.items())
            copy_verified(archive,remote/'compact_archives'/archive.name)
            print('BUDGET_PILOT_RESULT',json.dumps(row),flush=True)
            if .68 <= row['validation_dice'] <= .72:
                state.update(status='complete', selected=row, target_met=True, active=None); publish(); return
            if row['validation_dice'] > .72:upper=count
            else:lower=count
            next_count = int(round(((lower+upper)/2)/5)*5)
            if next_count <= 0 or next_count >= 200 or next_count == count:break
            count=next_count
        state.update(status='complete',target_met=False,active=None,
            selected=min(state['results'],key=lambda r:abs(r['validation_dice']-.70)))
        publish()
    except Exception as error:
        state.update(status='error',error=str(error)); publish(); raise
    finally:
        if child is not None and child.poll() is None:
            import signal
            ready=output/'supervisor_ready.json'
            while not ready.exists() and child.poll() is None:time.sleep(2)
            if child.poll() is None:child.send_signal(signal.SIGUSR1)
            child.wait()


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()
