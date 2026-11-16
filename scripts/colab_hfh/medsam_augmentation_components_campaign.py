"""Three native-only component ablations, paired with the saved125 control."""
from copy import deepcopy
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
from scripts.colab_hfh import medsam_constraints_parallel as parallel
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh.medsam_constraints_capacity import Capacity
from scripts.colab_hfh.medsam_125_campaign import validator125
from scripts.colab_hfh.medsam_augmentation_components import COMPONENTS, component_spec
from scripts.colab_hfh.medsam_mask_only_campaign import aggregate
from scripts.colab_hfh.medsam_recovery import copy_verified, rclone
from scripts.colab_hfh.runtime import digest, read, write

CAMPAIGN = 'medsam_augmentation_components125_f0s0_v1'


def make_protocol(control, path, component):
    p = deepcopy(control['protocol'])
    arm = p.pop('arm')
    if arm != 'dice' or (p['fold'], p['seed'], p['updates']) != (0, 0, 125):
        raise ValueError('Wrong unaugmented control')
    if p['augmentation'] != 'none':
        raise ValueError('Control must be unaugmented')
    p.update(augmentation=component_spec(component), augmentation_control_archive=str(path),
             augmentation_control_sha256=control['archive_sha256'])
    return p


def pilot_phase_source():
    source = textwrap.dedent(inspect.getsource(parallel.Campaign.phase))
    head = 'def phase(self, recipe, arm, pairs, bucket, fail_fast=False):'
    jobs = 'jobs=[self.prepare(recipe,arm,*pair) for pair in pairs]'
    if source.count(head) != 1 or source.count(jobs) != 1:
        raise ValueError('Audited scheduler changed')
    source = source.replace(head, "def phase_pilot(self):\n    recipe,arm,bucket,fail_fast={'name':'components125'},'components','results',False")
    return source.replace(jobs, 'jobs=[self.prepare(None,component,0,0) for component in COMPONENTS]')


class Components125(parallel.Campaign):
    def __init__(self, root):
        self.root, self.bundle = root, root/'bundle'
        self.work = root/CAMPAIGN
        self.durable = Path('/content/hfh_storage/results_medsam3_augmentation_components125_f0s0_v1')
        self.work.mkdir(exist_ok=True)
        self.handles = []
        for path in (self.work/'controller.lock', root/'work_medsam3/controller.lock'):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handles.append(handle)
        parallel.gpu_sample()  # Exact GPU family used by the saved controls.
        quota = json.loads(rclone('about', 'hfh_storage:', '--json'))
        if min(quota.get('free', 0), shutil.disk_usage(root).free) < 8*1024**3:
            raise RuntimeError('Need 8 GiB local and Drive headroom')
        self.control_paths, self.controls = {}, {}
        manifest = read(self.work/'control_manifest.json')
        for arm in ('dice',):
            entry = manifest[arm]
            path = self.work/'controls'/entry['name']
            if digest(path) != entry['sha256']:
                raise ValueError('Saved control changed: '+arm)
            self.control_paths[arm] = path
            self.controls[arm] = serial.archived_control(path)
        import torch
        if any(c['environment']['torch'] != torch.__version__ for c in self.controls.values()):
            raise ValueError('PyTorch differs from saved controls')
        self.deadline = time.time()+4*3600
        self.maximum, self.stopping, self.active = 3, False, {}
        self.tuner, self.phase_failure, self.last_publish = Capacity(maximum=3), None, 0
        archives.configure(self.bundle, self.work, self.durable)
        self.validate = validator125()
        self.state = dict(status='running_augmentation_pilot', updates=125, fold=0, seed=0,
            controls=[archives.verify_zip(self.control_paths['dice'])], results=[],
            components=list(COMPONENTS), augmentation={c:component_spec(c) for c in COMPONENTS},
            scope='Only gamma-only, gain-only and noise-only native runs; saved unaugmented control reused',
            comparison='Against the same arm without augmentation; unchanged 125-update LR/sample prefix',
            warning='Exploratory fold0 validation; equal optimizer updates, augmentation overhead separate')
        source = textwrap.dedent(inspect.getsource(parallel.Campaign.launch))
        old = "'scripts.colab_hfh.medsam_constraints_worker'"
        if source.count(old) != 1:
            raise ValueError('Audited worker launcher changed')
        source = source.replace(old, "'scripts.colab_hfh.medsam_augmentation_components_worker'")
        needle = 'for relative in ('
        if source.count(needle) != 1:
            raise ValueError('Source provenance hook changed')
        source = source.replace(needle, needle +
            "'scripts/colab_hfh/medsam_augmentation_components.py', 'scripts/colab_hfh/medsam_intensity_augmentation.py', "
            "'scripts/colab_hfh/medsam_augmentation_components_worker.py', "
            "'scripts/colab_hfh/medsam_augmentation_components_campaign.py', ")
        ns = dict(parallel.__dict__, COMPONENTS=COMPONENTS)
        exec(compile(source, __file__, 'exec'), ns)
        self.launch = ns['launch'].__get__(self, type(self))
        exec(compile(pilot_phase_source(), __file__, 'exec'), ns)
        self.phase_pilot = ns['phase_pilot'].__get__(self, type(self))
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

    def prepare(self, recipe, component, seed, fold):
        if (seed, fold) != (0, 0) or component not in COMPONENTS:
            raise ValueError('Pilot scope changed')
        arm = 'dice'
        control = self.controls[arm]
        p = make_protocol(control, self.control_paths[arm], component)
        name = f'aug125_{component}__dice__f0_s0'
        output = self.work/'runs'/name
        output.mkdir(parents=True, exist_ok=True)
        path = self.work/'protocols'/(name+'.json')
        if path.exists() and read(path) != p:
            raise ValueError('Existing augmentation protocol changed')
        write(path, p)
        return dict(name=name, output=output, remote=self.durable/'runs'/name, control=control,
                    protocol=p, path=path, recipe=p['constraint_recipe'], arm=arm, component=component, seed=0, fold=0)

    def commit(self, job):
        output, protocol, control = job['output'], job['protocol'], job['control']
        row = self.validate(output, protocol, control, job['arm'], job['recipe'])
        if read(output/'calibration.json') != serial.zip_read(self.control_paths[job['arm']], 'calibration.json'):
            raise ValueError('Augmentation changed constraint calibration')
        if read(output/'training_summary.json')['augmentation'] != protocol['augmentation']:
            raise ValueError('Augmentation provenance missing')
        tape = read(output/'augmentation_schedule.json')
        if len(tape) != 250 or [(r['case'], r['anchor']) for r in tape] != [tuple(x) for x in protocol['frozen_sample_schedule']]:
            raise ValueError('Augmentation sampling differs')
        if digest(output/'checkpoint_latest.pt') != read(output/'completion.json')['checkpoint_sha256']:
            raise ValueError('Final checkpoint differs')
        preflight = read(output/'loss_logging_preflight.json')
        if preflight['status'] != 'passed' or not all(preflight[k] for k in ('exact_loss','exact_gradients','exact_rng')):
            raise ValueError('Loss-logging preflight missing')
        history = read(output/'training_history.json')
        required = ['native_term/weighted/'+k for k in preflight['weights']]
        if any(not all(k in h for k in required) for h in history):
            raise ValueError('Individual losses missing from training history')
        if any(not all(math.isfinite(float(h[k])) for k in required) or
               not math.isclose(sum(h[k] for k in required), h['native'], rel_tol=1e-5, abs_tol=1e-3)
               for h in history):
            raise ValueError('Logged weighted losses do not reconstruct native history')
        reference_truth = {(r['split'],r['case_name']):r['truth_sha256'] for r in control['evaluation']['records']}
        records = read(output/'evaluation.json')['records']
        if any(r['truth_sha256'] != reference_truth[r['split'],r['case_name']] for r in records):
            raise ValueError('Evaluation truth changed')
        for record in records:
            spatial = record['spatial']
            if spatial['fp'] != record['foreground_fp'] or spatial['fn'] != record['foreground_fn']:
                raise ValueError('Spatial counts disagree with evaluation')
        spatial = aggregate({(r['split'], r['case_name']):r['spatial'] for r in records})
        row.update(augmentation=protocol['augmentation'], component=job['component'], control_arm=job['arm'],
            training_gain_pp=100*(row['training_dice']-row['baseline_training_dice']),
            generalization_gap_pp=100*(row['training_dice']-row['validation_dice']),
            spatial_validation=spatial['validation'], augmentation_schedule_sha256=digest(output/'augmentation_schedule.json'))
        write(output/'paired_result.json', row)
        write(output/'result.json', row)
        files = [p for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.log','.csv')
                 and p.name != 'archive_manifest.json']
        write(output/'archive_manifest.json', {p.name:digest(p) for p in files})
        path = archives.CONTEXT['cache']/(job['name']+'.zip')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for p in files+[output/'archive_manifest.json']:
                zipped.write(p, 'results/'+p.name)
            for name in ('medsam_intensity_augmentation.py','medsam_augmentation_components.py','medsam_augmentation_components_worker.py','medsam_augmentation_components_campaign.py'):
                zipped.write(Path(__file__).with_name(name), 'source/'+name)
        if archives.verify_zip(path) != row:
            raise ValueError('Archive verification failed')
        copy_verified(path, archives.CONTEXT['remote']/path.name)
        archives.CONTEXT['inventory'][path.name] = dict(Name=path.name, Hashes={'md5':hashlib.md5(path.read_bytes()).hexdigest()})
        print('AUGMENTATION_PAIRED_FINAL', json.dumps(row), flush=True)
        return row

    def publish(self, gpu, workers, force=False):
        previous = self.last_publish
        super().publish(gpu, workers, force)
        rows = []
        for component in COMPONENTS:
            path = self.work/'runs'/f'aug125_{component}__dice__f0_s0'/'training_history.json'
            if path.exists():
                history = read(path)
                rows.append(dict(component=component, updates=len(history), latest=[{
                    k:v for k,v in h.items() if k in ('update','lr','native','gradient_norm') or
                    k.startswith(('native_term/weighted/','augmented_training/'))} for h in history[-5:]]))
        path = self.work/'loss_progress.json'
        write(path, dict(observed_at=time.time(), runs=rows))
        if force or self.last_publish != previous:
            copy_verified(path, self.durable/'monitor/loss_progress.json')

    def run(self):
        try:
            rows = self.phase_pilot()
            if len(rows) != 3 or {r['component'] for r in rows} != set(COMPONENTS):
                raise ValueError('Incomplete augmentation pilot')
            self.state.update(status='complete_3_components',
                              next_action='Review paired Dice and FP/FN; no automatic expansion')
            write(self.work/'augmentation_results.json', self.state)
            copy_verified(self.work/'augmentation_results.json', self.durable/'augmentation_results.json')
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
            self.publish(parallel.gpu_sample(), [], force=True)


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    Components125(Path('/content/hippo_hfh')).run()
