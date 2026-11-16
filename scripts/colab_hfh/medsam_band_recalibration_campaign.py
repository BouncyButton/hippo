"""Exactly three BAND seed-zero runs; saved gain baselines remain immutable."""
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import textwrap
import time
import zipfile

from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh import medsam_constraints_parallel as parallel
from scripts.colab_hfh.medsam_component_constraints_campaign import ThreeWorkers, portable_gpu_sample
from scripts.colab_hfh.medsam_band_recalibration import CAMPAIGN, POLICY, check_protocol, summarize
from scripts.colab_hfh.medsam_gain_replication_campaign import phase_source
from scripts.colab_hfh.medsam_mask_only_campaign import aggregate
from scripts.colab_hfh.medsam_recovery import copy_verified, rclone
from scripts.colab_hfh.runtime import read, write, digest

SOURCES = ('medsam_band_recalibration.py', 'medsam_band_recalibration_worker.py',
           'medsam_band_recalibration_campaign.py', 'medsam_gain_replication.py',
           'medsam_augmentation_components.py', 'medsam_intensity_augmentation.py')


def validator():
    source = inspect.getsource(serial.paired_result)
    for old, new in (
        ("done['updates'] != 200", "done['updates'] != 125"),
        ('len(history) != 200', 'len(history) != 125'),
        ("[r['lr'] for r in history] != protocol['frozen_learning_rates']",
         "[r['lr'] for r in history] != protocol['frozen_learning_rates'][:125]"),
        ('updates=200, training_dice=', 'updates=125, training_dice='),
        ('200 matched optimizer updates;', '125 matched optimizer updates;'),
        ('range(200)', 'range(125)'),
        ("4 * len(protocol['train_cases'])", "20 * len(protocol['train_cases'])"),
        ('> .50000001', '> .05000001')):
        if source.count(old) != 1:
            raise ValueError('Paired validator changed: ' + old)
        source = source.replace(old, new)
    namespace = dict(serial.__dict__)
    exec(compile(source, __file__, 'exec'), namespace)
    return namespace['paired_result']


class BandRecalibration(parallel.Campaign):
    def __init__(self, root):
        self.root, self.bundle = root, root / 'bundle'
        self.work = root / CAMPAIGN
        self.durable = Path('/content/hfh_storage/results_' + CAMPAIGN)
        self.work.mkdir(exist_ok=True)
        self.handles = []
        for path in (self.work/'controller.lock', root/'work_medsam3/controller.lock'):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handles.append(handle)
        parallel.gpu_sample = portable_gpu_sample
        self.hardware = portable_gpu_sample()
        quota = json.loads(rclone('about', 'hfh_storage:', '--json'))
        if min(quota.get('free', 0), shutil.disk_usage(root).free) < 12 * 1024**3:
            raise RuntimeError('Need12GiB headroom locally and on second-account Drive')
        write(self.work/'storage_quota.json', dict(checked_at=time.time(), **quota))
        self.control_paths, self.controls = {}, {}
        for fold, entry in read(self.work/'controls_manifest.json').items():
            path = self.work/'controls'/entry['name']
            if digest(path) != entry['sha256']:
                raise ValueError('Saved native archive changed')
            archives.verify_zip(path)
            self.control_paths[int(fold)] = path
            self.controls[0, int(fold)] = serial.archived_control(path)
        import torch
        if any(c['environment']['torch'] != torch.__version__ for c in self.controls.values()):
            raise ValueError('Scientific torch version differs from saved controls')
        self.deadline = time.time() + 18 * 3600
        self.maximum, self.stopping, self.active = 3, False, {}
        self.tuner, self.phase_failure, self.last_publish = ThreeWorkers(), None, 0
        archives.configure(self.bundle, self.work, self.durable)
        self.validate = validator()
        self.state = dict(status='running_recalibrated_bands', updates=125, seed=0, folds=[0,1,2],
            controls=[archives.verify_zip(p) for p in self.control_paths.values()], results=[],
            policy=POLICY, requested_workers=3, hardware=self.hardware,
            scope='Exactly three BAND runs; gain-only; all existing native controls reused; no performance pruning',
            hardware_policy='User accepts L4; three-worker target subject to memory guard, one on L4',
            warning='Exploratory calibration-policy change; all outcomes retained')
        namespace = dict(parallel.__dict__, ThreeWorkers=ThreeWorkers)
        exec(compile(phase_source(), __file__, 'exec'), namespace)
        self.phase = namespace['phase'].__get__(self, type(self))
        source = textwrap.dedent(inspect.getsource(parallel.Campaign.launch))
        source = source.replace("'scripts.colab_hfh.medsam_constraints_worker'",
                                "'scripts.colab_hfh.medsam_band_recalibration_worker'")
        source = source.replace('for relative in (', 'for relative in (' +
                                ','.join(repr('scripts/colab_hfh/'+x) for x in SOURCES) + ',')
        exec(compile(source, __file__, 'exec'), namespace)
        self.launch = namespace['launch'].__get__(self, type(self))
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

    def prepare(self, recipe, arm, seed, fold):
        if arm != 'dice_bands' or seed != 0 or fold not in (0,1,2):
            raise ValueError('Only BAND seed0 folds0/1/2 authorized')
        name = f'recal125__dice_bands__f{fold}_s0'
        p = read(self.work/'protocol_templates.json')[str(fold)]
        check_protocol(p)
        control = self.controls[seed,fold]
        if p['baseline_archive_sha256'] != control['archive_sha256']:
            raise ValueError('Baseline binding changed')
        output = self.work/'runs'/name
        output.mkdir(parents=True, exist_ok=True)
        path = self.work/'protocols'/(name+'.json')
        if path.exists() and read(path) != p:
            raise ValueError('Frozen recalibration protocol changed')
        write(path, p)
        return dict(name=name, protocol=p, path=path, output=output, remote=self.durable/'runs'/name,
                    control=control, recipe=p['constraint_recipe'], arm=arm, seed=seed, fold=fold)

    def workers(self, gpu):
        rows = super().workers(gpu)
        for row in rows:
            path = self.active[row['name']]['output']/'calibration_partial.json'
            if row['step'] == 0 and path.exists():
                row.update(phase='calibration', calibration_slabs=len(read(path)['rows']), calibration_total=200)
        return rows

    def commit(self, job):
        output, p, control = job['output'], job['protocol'], job['control']
        row = self.validate(output, p, control, job['arm'], job['recipe'])
        if digest(output/'checkpoint_latest.pt') != read(output/'completion.json')['checkpoint_sha256']:
            raise ValueError('Checkpoint hash differs')
        cal = read(output/'calibration.json')
        if cal != summarize(cal['rows']):
            raise ValueError('Calibration statistics do not reconstruct')
        if any(r['case'] not in p['train_cases'] or r['case'] in p['evaluation_cases'] for r in cal['rows']):
            raise ValueError('Calibration leakage')
        if read(output/'augmentation_schedule.json') != serial.zip_read(self.control_paths[job['fold']], 'augmentation_schedule.json'):
            raise ValueError('Gain augmentation tape changed')
        audit = read(output/'loss_logging_preflight.json')
        if not all(audit[k] for k in ('exact_loss','exact_gradients','exact_rng')):
            raise ValueError('Logging preflight failed')
        history = read(output/'training_history.json')
        for i, h in enumerate(history, 1):
            if not all(math.isfinite(v) for v in h.values()) or h['ramp'] != min(i/20, 1.):
                raise ValueError('Invalid history or training ramp')
            if not math.isclose(h['total'], h['native'] + h['ramp']*cal['weights']['bands']*h['bands'], abs_tol=1e-4):
                raise ValueError('Weighted objective differs')
            if not math.isclose(sum(v for k,v in h.items() if k.startswith('native_term/weighted/')), h['native'], abs_tol=1e-3):
                raise ValueError('Native loss reconstruction differs')
        records = read(output/'evaluation.json')['records']
        truth = {(r['split'],r['case_name']):r['truth_sha256'] for r in control['evaluation']['records']}
        for r in records:
            if r['truth_sha256'] != truth[r['split'],r['case_name']]:
                raise ValueError('Evaluation truth differs')
            (_,fp),(fn,tp) = r['confusion']
            if abs(2*tp/(2*tp+fp+fn)-r['macro_hard_dice']) > 1e-12 or r['spatial']['fp'] != fp or r['spatial']['fn'] != fn:
                raise ValueError('Evaluation metrics differ from counts')
        row.update(component='gain', hardware=read(output/'execution_hardware.json'),
            calibration_policy=POLICY, calibration_stability=cal['stability'],
            training_gain_pp=100*(row['training_dice']-row['baseline_training_dice']),
            spatial=aggregate({(r['split'],r['case_name']):r['spatial'] for r in records}))
        write(output/'paired_result.json', row)
        write(output/'result.json', row)
        files = [p for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.log','.csv') and p.name!='archive_manifest.json']
        write(output/'archive_manifest.json', {p.name:digest(p) for p in files})
        path = archives.CONTEXT['cache']/(job['name']+'.zip')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in files+[output/'archive_manifest.json']:
                z.write(p, 'results/'+p.name)
            for name in SOURCES:
                z.write(Path(__file__).with_name(name), 'source/'+name)
        if archives.verify_zip(path) != row:
            raise ValueError('Archive verification failed')
        copy_verified(path, archives.CONTEXT['remote']/path.name)
        archives.CONTEXT['inventory'][path.name] = dict(Name=path.name, Hashes={'md5':hashlib.md5(path.read_bytes()).hexdigest()})
        print('RECALIBRATION_FINAL', json.dumps(row), flush=True)
        return row

    def run(self):
        try:
            self.phase({'name':POLICY['name']}, 'dice_bands', [(0,0),(0,1),(0,2)], 'results', False)
            if len(self.state['results']) != 3:
                raise ValueError('Incomplete three-fold comparison')
            self.state['status'] = 'complete_three_bands'
        except SystemExit as error:
            if error.code != 75:
                raise
            self.state['status'] = 'paused'
        except Exception as error:
            self.state.update(status='error', error=str(error)[-3500:])
            raise
        finally:
            for job in self.active.values():
                while job['process'].poll() is None and not (job['output']/'supervisor_ready.json').exists():
                    time.sleep(2)
                if job['process'].poll() is None:
                    job['process'].send_signal(signal.SIGUSR1)
            for job in self.active.values():
                job['process'].wait()
            (self.work/'STOP').touch()
            self.publish(portable_gpu_sample(), [], True)


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    BandRecalibration(Path('/content/hippo_hfh')).run()
