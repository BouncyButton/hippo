"""Four serial adaptive BAND experiments, reusing all existing controls."""
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import textwrap
import time
import zipfile

from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh import medsam_constraints_parallel as parallel
from scripts.colab_hfh.medsam_gain_replication_campaign import GainReplication, portable_gpu_sample
from scripts.colab_hfh.medsam_adaptive_band import (CAMPAIGN, POLICY, ARMS, check_protocol,
    empty_state, transition, band_factor, calibration_summary)
from scripts.colab_hfh.medsam_mask_only_campaign import aggregate
from scripts.colab_hfh.medsam_recovery import copy_verified
from scripts.colab_hfh.runtime import read, write, digest

RUNS = ((2, 'dice_bands'), (2, 'dice_bands_sdf'), (1, 'dice_bands'), (1, 'dice_bands_sdf'))
SOURCES = ('medsam_adaptive_band.py', 'medsam_adaptive_band_worker.py',
           'medsam_adaptive_band_recovery.py', 'medsam_adaptive_band_campaign.py')


def validate_adaptive(output, protocol, arm, history, calibration):
    if calibration['weights']['bands'] != 0 or calibration['weights']['edge'] != 0:
        raise ValueError('Initial BAND must remain off')
    final = read(output/'adaptive_band_state.json')
    if [p['update'] for p in final['probes']] != list(range(0, 125, POLICY['probe_every'])):
        raise ValueError('Missing full-support gate observation')
    cal_by_update = {c['identity']['update']:c for c in final['calibrations']}
    for c in ([calibration['sdf_calibration']] if 'sdf' in arm else [])+final['calibrations']:
        term = c['term']
        recomputed = calibration_summary(c['rows'], term)
        if any(c[k] != v for k,v in recomputed.items()):
            raise ValueError('Calibration does not reconstruct')
        if any(r['case'] not in protocol['train_cases'] or r['case'] in protocol['evaluation_cases'] for r in c['rows']):
            raise ValueError('Calibration leakage')
    state = empty_state()
    probe_index = {r['update']:r for r in final['probes']}
    for i,h in enumerate(history):
        if i in probe_index:
            observation = {k:v for k,v in probe_index[i].items() if k not in ('update','ready','event')}
            if {c['case'] for c in observation['crops']} != set(protocol['train_cases']):
                raise ValueError('Gate did not cover exactly training crops')
            from scripts.colab_hfh.medsam_adaptive_band import summarize_errors
            if summarize_errors(observation['crops']) != observation:
                raise ValueError('Gate error counts do not reconstruct')
            state,event = transition(state,observation,i)
            if event == 'activate':
                cal = cal_by_update[i]
                state['weight'] = cal['weight']; state['calibrations'].append(cal)
        expected = dict(active=state['active'], band_ramp=band_factor(state,i), band_weight=state['weight'],
                        last_probe=state['last_probe'],activation_update=state['activation_update'])
        if h['adaptive_band'] != expected or h['ramp'] != min((i+1)/20,1.):
            raise ValueError('Adaptive timing or recorded weight differs')
        value = h['native']+h['ramp']*sum(w*h[k] for k,w in calibration['weights'].items())
        value += expected['band_ramp']*expected['band_weight']*h['bands']
        if not math.isclose(h['total'],value,rel_tol=2e-5,abs_tol=2e-4):
            raise ValueError('Applied objective differs from logged weights')
        if not math.isclose(sum(v for k,v in h.items() if k.startswith('native_term/weighted/')),
                            h['native'],rel_tol=1e-5,abs_tol=1e-3):
            raise ValueError('Native objective accounting differs')
    if state != final:
        raise ValueError('Final gate state does not reconstruct from observations')


def validator():
    source = inspect.getsource(serial.paired_result)
    begin = source.index("    active = {name for name in ('bands'")
    end = source.index("    evaluation = read(output/'evaluation.json')",begin)
    source = source[:begin]+"    validate_adaptive(output,protocol,arm,history,calibration)\n"+source[end:]
    for old,new in (
        ("done['updates'] != 200", "done['updates'] != 125"),
        ('len(history) != 200','len(history) != 125'),
        ("[r['lr'] for r in history] != protocol['frozen_learning_rates']",
         "[r['lr'] for r in history] != protocol['frozen_learning_rates'][:125]"),
        ('updates=200, training_dice=','updates=125, training_dice='),
        ('200 matched optimizer updates;','125 matched optimizer updates;'),
        ('range(200)','range(125)')):
        if source.count(old) != 1:
            raise ValueError('Paired validator hook changed: '+old)
        source = source.replace(old,new)
    namespace = dict(serial.__dict__,validate_adaptive=validate_adaptive)
    exec(compile(source,__file__,'exec'),namespace)
    return namespace['paired_result']


class AdaptiveCampaign(GainReplication):
    def __init__(self,root):
        super().__init__(root)
        old_work = self.work
        if read(old_work/'campaign_status.json')['status'] != 'paused' or not (old_work/'STOP').exists():
            raise ValueError('Previous campaign must remain paused')
        if 'L4' not in str(self.hardware):
            raise ValueError('Requested L4 required')
        self.control_paths = {}
        for fold in (1,2):
            self.load_control(0,fold)
            self.control_paths[fold] = archives.CONTEXT['cache']/f'gain125__dice__f{fold}_s0.zip'
        self.work = root/CAMPAIGN; self.work.mkdir(exist_ok=True)
        self.durable = Path('/content/hfh_storage/results_'+CAMPAIGN)
        archives.configure(self.bundle,self.work,self.durable)
        self.validate = validator()
        self.deadline = time.time()+16*3600
        self.state = dict(status='starting_adaptive_band', results=[], controls=[], workers=[],
            scope='Exactly BAND alone and SDF+BAND, folds1/2 seed0; then stop',
            policy=POLICY, maximum_active_workers=1, maximum_queued_jobs=1,
            baseline_reruns=0, automatic_expansion=False, warning='Exploratory; no test-set claims')
        # Parent binds launch on the instance. Replace that exact binding.
        source = textwrap.dedent(inspect.getsource(parallel.Campaign.launch))
        source = source.replace("'scripts.colab_hfh.medsam_constraints_worker'",
                                "'scripts.colab_hfh.medsam_adaptive_band_worker'")
        source = source.replace('for relative in (','for relative in ('+
                                ','.join(repr('scripts/colab_hfh/'+n) for n in SOURCES)+',')
        namespace = dict(parallel.__dict__)
        exec(compile(source,__file__,'exec'),namespace)
        self.launch = namespace['launch'].__get__(self,type(self))

    def prepare(self,recipe,arm,seed,fold):
        if seed != 0 or (fold,arm) not in RUNS:
            raise ValueError('Outside bounded adaptive experiment')
        p = deepcopy(self.templates[f'dice_bands__f{fold}_s0'])
        p.update(requested_arms=[arm],adaptive_band=POLICY,
            adaptive_source_sha256=digest(Path(__file__).with_name('medsam_adaptive_band.py')),
            baseline_archive_sha256=self.controls[0,fold]['archive_sha256'],
            constraint_recipe=dict(name=POLICY['name'],target=.01,ramp=20,numerics='strict_empty_grid_v1'),
            calibration_policy='Current-model native-relative training-only calibration at BAND activation; SDF at initialization',
            comparison='Gain-only adaptive BAND +/- SDF; reuse saved baseline; final125; original200 LR horizon')
        check_protocol(p)
        name = f'adaptive125__{arm}__f{fold}_s0'
        output = self.work/'runs'/name; output.mkdir(parents=True,exist_ok=True)
        path = self.work/'protocols'/(name+'.json')
        if path.exists() and read(path) != p:
            raise ValueError('Frozen adaptive protocol changed')
        write(path,p)
        return dict(name=name,protocol=p,path=path,output=output,remote=self.durable/'runs'/name,
                    control=self.controls[0,fold],recipe=p['constraint_recipe'],arm=arm,seed=seed,fold=fold)

    def commit(self,job):
        output,p,control = job['output'],job['protocol'],job['control']
        row = self.validate(output,p,control,job['arm'],job['recipe'])
        if digest(output/'checkpoint_latest.pt') != read(output/'completion.json')['checkpoint_sha256']:
            raise ValueError('Checkpoint hash differs')
        if read(output/'augmentation_schedule.json') != serial.zip_read(self.control_paths[job['fold']],'augmentation_schedule.json'):
            raise ValueError('Gain tape differs from saved native control')
        audit = read(output/'loss_logging_preflight.json')
        if not all(audit[k] for k in ('exact_loss','exact_gradients','exact_rng')):
            raise ValueError('Logging preflight failed')
        if not read(output/'runtime_validation.json')['deterministic_algorithms']:
            raise ValueError('Strict numerics required')
        records = read(output/'evaluation.json')['records']
        truth = {(r['split'],r['case_name']):r['truth_sha256'] for r in control['evaluation']['records']}
        for r in records:
            if r['truth_sha256'] != truth[r['split'],r['case_name']]:
                raise ValueError('Evaluation truth differs')
            (_,fp),(fn,tp) = r['confusion']
            if abs(2*tp/(2*tp+fp+fn)-r['macro_hard_dice']) > 1e-12 or r['spatial']['fp'] != fp or r['spatial']['fn'] != fn:
                raise ValueError('Evaluation counts differ')
        gate = read(output/'adaptive_band_state.json')
        row.update(component='gain',hardware=read(output/'execution_hardware.json'),adaptive_gate=gate,
            calibration_policy=POLICY,band_active_updates=sum(h['adaptive_band']['band_ramp']>0 for h in read(output/'training_history.json')),
            band_weights=[c['weight'] for c in gate['calibrations']],
            spatial=aggregate({(r['split'],r['case_name']):r['spatial'] for r in records}))
        # Exact unchanged prefix is checked against already completed native history.
        if job['arm'] == 'dice_bands':
            prefix = [h for h in read(output/'training_history.json') if h['adaptive_band']['band_ramp'] == 0]
            first_activation = next((e['update'] for e in gate['events'] if e['action']=='activate'),125)
            native = control['training_history'][:first_activation]
            candidate = read(output/'training_history.json')[:first_activation]
            row['native_prefix_audit'] = dict(updates=len(native),
                max_native_loss_difference=max((abs(a['native']-b['native']) for a,b in zip(native,candidate)),default=0),
                max_gradient_norm_difference=max((abs(a['gradient_norm']-b['gradient_norm']) for a,b in zip(native,candidate)),default=0))
        write(output/'paired_result.json',row); write(output/'result.json',row)
        files = [q for q in output.iterdir() if q.is_file() and q.suffix in ('.json','.log','.csv') and q.name!='archive_manifest.json']
        write(output/'archive_manifest.json',{q.name:digest(q) for q in files})
        path = archives.CONTEXT['cache']/(job['name']+'.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for q in files+[output/'archive_manifest.json']:
                z.write(q,'results/'+q.name)
            for name in SOURCES:
                z.write(Path(__file__).with_name(name),'source/'+name)
        if archives.verify_zip(path) != row:
            raise ValueError('Archive verification failed')
        copy_verified(path,archives.CONTEXT['remote']/path.name)
        archives.CONTEXT['inventory'][path.name] = dict(Name=path.name,Hashes={'md5':hashlib.md5(path.read_bytes()).hexdigest()})
        print('ADAPTIVE_FINAL',job['name'],row['validation_dice'],row['delta_dice_pp'],row['band_active_updates'],flush=True)
        return row

    def publish(self,gpu,workers,force=False):
        # Keep status compact; complete gate/gradient evidence lives in archives.
        parallel.Campaign.publish(self,gpu,workers,force)

    def run(self):
        try:
            for i,(fold,arm) in enumerate(RUNS):
                self.state.update(status='running_adaptive_band',current=dict(fold=fold,seed=0,arm=arm),
                    queued_job=dict(fold=RUNS[i+1][0],arm=RUNS[i+1][1],seed=0) if i+1<len(RUNS) else None)
                self.phase({'name':POLICY['name']},arm,[(0,fold)],'results',False)
            if len(self.state['results']) != 4:
                raise ValueError('Incomplete bounded experiment')
            self.state.update(status='complete_adaptive_four_runs',queued_job=None,
                              next_action='Stop; compare all four results. No automatic additional runs.')
        except SystemExit as error:
            if error.code != 75:
                raise
            self.state.update(status='paused',queued_job=None)
        except Exception as error:
            self.state.update(status='error',error=str(error)[-3000:],queued_job=None)
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
            self.publish(portable_gpu_sample(),[],True)


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    AdaptiveCampaign(Path('/content/hippo_hfh')).run()
