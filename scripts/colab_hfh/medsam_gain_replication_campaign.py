"""Complete every gain-only seed/fold triple without a performance stopping gate."""
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
from scripts.colab_hfh.medsam_component_constraints_campaign import portable_gpu_sample,ThreeWorkers
from scripts.colab_hfh.medsam_gain_replication import CAMPAIGN,ARMS,PAIRS
from scripts.colab_hfh.medsam_125_campaign import validator125,local_control
from scripts.colab_hfh.medsam_mask_only_campaign import aggregate
from scripts.colab_hfh.medsam_recovery import copy_verified,rclone
from scripts.colab_hfh.runtime import read,write,digest


def phase_source():
    source=textwrap.dedent(inspect.getsource(parallel.Campaign.phase))
    old='self.tuner=Capacity(maximum=self.maximum)'
    if source.count(old)!=1:raise ValueError('Audited admission hook changed')
    return source.replace(old,'self.tuner=ThreeWorkers()')


class GainReplication(parallel.Campaign):
    def __init__(self,root):
        self.root,self.bundle=root,root/'bundle'
        self.work=root/CAMPAIGN;self.work.mkdir(exist_ok=True)
        self.durable=Path('/content/hfh_storage/results_medsam3_gain125_3fold3seed_v1')
        self.handles=[]
        for path in (self.work/'controller.lock',root/'work_medsam3/controller.lock'):
            path.parent.mkdir(parents=True,exist_ok=True);h=path.open('a')
            fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);self.handles.append(h)
        parallel.gpu_sample=portable_gpu_sample
        self.hardware=portable_gpu_sample()
        quota=json.loads(rclone('about','hfh_storage:','--json'))
        if min(quota.get('free',0),shutil.disk_usage(root).free)<12*1024**3:
            raise RuntimeError('Need12GiB local and durable headroom for active recovery')
        self.pilot_paths={};self.pilots={}
        for arm,entry in read(self.work/'pilot_manifest.json').items():
            path=self.work/'pilots'/entry['name']
            if digest(path)!=entry['sha256']:raise ValueError('Pilot changed: '+arm)
            self.pilot_paths[arm]=path;self.pilots[arm]=serial.archived_control(path)
        import torch
        if any(c['environment']['torch']!=torch.__version__ for c in self.pilots.values()):
            raise ValueError('Scientific torch version differs from pilot')
        self.templates=read(self.work/'protocol_templates.json')
        self.controls={(0,0):self.pilots['dice']}
        self.maximum,self.stopping,self.active=3,False,{}
        self.tuner,self.phase_failure,self.last_publish=ThreeWorkers(),None,0
        self.deadline=time.time()+18*3600
        self.validate=validator125()
        archives.configure(self.bundle,self.work,self.durable)
        self.state=dict(status='running_gain_replication',updates=125,components=['gain'],arms=list(ARMS),
            pairs=[list(x) for x in PAIRS],controls=[],results=[],reused_pilot=[],requested_workers=3,
            new_runs=24,scope='Only gain-only native,BAND,EDGE+DEPTH;threefolds,threeseeds;all outcomes retained',
            calibration='Fresh training-only calibration for every new run;pilot targets and20-update ramp frozen',
            hardware_policy='Use available BF16 GPU; user accepts L4/A100 comparison, hardware recorded',
            warning='Exploratory validation-selected pilot replication; no held-out tuning or negative-result deletion')
        ns=dict(parallel.__dict__,ThreeWorkers=ThreeWorkers)
        exec(compile(phase_source(),__file__,'exec'),ns);self.phase=ns['phase'].__get__(self,type(self))
        source=textwrap.dedent(inspect.getsource(parallel.Campaign.launch))
        source=source.replace("'scripts.colab_hfh.medsam_constraints_worker'","'scripts.colab_hfh.medsam_gain_replication_worker'")
        needle='for relative in ('
        if source.count(needle)!=1:raise ValueError('Provenance hook changed')
        extra=('medsam_gain_replication.py','medsam_gain_replication_worker.py','medsam_gain_replication_campaign.py',
               'medsam_augmentation_components.py','medsam_intensity_augmentation.py')
        source=source.replace(needle,needle+','.join(repr('scripts/colab_hfh/'+x) for x in extra)+',')
        exec(compile(source,__file__,'exec'),ns);self.launch=ns['launch'].__get__(self,type(self))
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)

    def prepare(self,recipe,arm,seed,fold):
        if arm not in ARMS or (seed,fold) not in PAIRS or (seed,fold)==(0,0):raise ValueError('Unauthorized or reused job')
        name=f'gain125__{arm}__f{fold}_s{seed}'
        p=read(self.work/'protocol_templates.json')[f'{arm}__f{fold}_s{seed}']
        control=self.controls.get((seed,fold))
        if arm!='dice':
            if control is None:raise ValueError('Matched native must finish before constraints')
            p['baseline_archive_sha256']=control['archive_sha256']
        output=self.work/'runs'/name;output.mkdir(parents=True,exist_ok=True)
        path=self.work/'protocols'/(name+'.json')
        if path.exists() and read(path)!=p:raise ValueError('Frozen replication protocol changed')
        write(path,p)
        return dict(name=name,protocol=p,path=path,output=output,remote=self.durable/'runs'/name,
                    control=control,recipe=p['constraint_recipe'],arm=arm,seed=seed,fold=fold)

    def commit(self,job):
        output,p=job['output'],job['protocol']
        control=job['control']
        if job['arm']=='dice':
            control=local_control(output);control['archive_sha256']=None
        row=self.validate(output,p,control,job['arm'],job['recipe'])
        if digest(output/'checkpoint_latest.pt')!=read(output/'completion.json')['checkpoint_sha256']:
            raise ValueError('Final checkpoint changed')
        audit=read(output/'loss_logging_preflight.json')
        if not all(audit[k] for k in ('exact_loss','exact_gradients','exact_rng')):raise ValueError('Logging preflight failed')
        history=read(output/'training_history.json')
        for h in history:
            if not math.isclose(sum(h['native_term/weighted/'+k] for k in audit['weights']),h['native'],rel_tol=1e-5,abs_tol=1e-3):
                raise ValueError('Native loss terms do not reconstruct total')
        tape=read(output/'augmentation_schedule.json')
        if len(tape)!=250 or [(r['case'],r['anchor']) for r in tape]!=[tuple(x) for x in p['frozen_sample_schedule']]:
            raise ValueError('Augmentation sample tape differs')
        if job['arm']!='dice':
            native_path=archives.CONTEXT['cache']/f"gain125__dice__f{job['fold']}_s{job['seed']}.zip"
            if tape!=serial.zip_read(native_path,'augmentation_schedule.json'):raise ValueError('Paired gain draws differ')
            cal=read(output/'calibration.json')
            if any(r['case'] not in p['train_cases'] or r['case'] in p['evaluation_cases'] for r in cal['rows']):
                raise ValueError('Calibration contains held-out data')
        records=read(output/'evaluation.json')['records']
        before={(r['split'],r['case_name']):r for r in control['evaluation']['records']}
        if any(r['truth_sha256']!=before[r['split'],r['case_name']]['truth_sha256'] for r in records):
            raise ValueError('Paired evaluation labels changed')
        spatial=aggregate({(r['split'],r['case_name']):r['spatial'] for r in records})
        hardware=read(output/'execution_hardware.json')
        row.update(component='gain',gpu=hardware['gpu'],hardware=hardware,augmentation=p['augmentation'],
            training_gain_pp=100*(row['training_dice']-row['baseline_training_dice']),
            spatial_validation=spatial['validation'],augmentation_schedule_sha256=digest(output/'augmentation_schedule.json'),
            reused=False,calibration_policy=p['calibration_policy'])
        write(output/'paired_result.json',row);write(output/'result.json',row)
        files=[q for q in output.iterdir() if q.is_file() and q.suffix in ('.json','.log','.csv') and q.name!='archive_manifest.json']
        write(output/'archive_manifest.json',{q.name:digest(q) for q in files})
        path=archives.CONTEXT['cache']/(job['name']+'.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for q in files+[output/'archive_manifest.json']:z.write(q,'results/'+q.name)
            for name in ('medsam_gain_replication.py','medsam_gain_replication_worker.py','medsam_gain_replication_campaign.py'):
                z.write(Path(__file__).with_name(name),'source/'+name)
        if archives.verify_zip(path)!=row:raise ValueError('Archive validation failed')
        copy_verified(path,archives.CONTEXT['remote']/path.name)
        archives.CONTEXT['inventory'][path.name]=dict(Name=path.name,Hashes={'md5':hashlib.md5(path.read_bytes()).hexdigest()})
        print('GAIN_REPLICATION_FINAL',json.dumps(row),flush=True)
        return row

    def load_control(self,seed,fold):
        path=archives.CONTEXT['cache']/f'gain125__dice__f{fold}_s{seed}.zip'
        self.controls[seed,fold]=serial.archived_control(path)

    def reuse_pilot(self):
        native=self.pilots['dice']
        for arm in ARMS:
            path=self.pilot_paths[arm];control=self.pilots[arm]
            # Original archives are preserved byte-for-byte with their historical GPU caveat.
            output=self.work/'reused'/arm;output.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.startswith('results/') and name.endswith('.json'):(output/Path(name).name).write_bytes(z.read(name))
            p=read(output/'protocol.json');p.pop('arm')
            row=self.validate(output,p,native,arm,p['constraint_recipe'])
            if serial.zip_read(path,'augmentation_schedule.json')!=serial.zip_read(self.pilot_paths['dice'],'augmentation_schedule.json'):
                raise ValueError('Pilot gain draws differ')
            result=archives.verify_zip(path)
            row.update(component='gain',reused=True,gpu=result.get('gpu','NVIDIA A100-SXM4-40GB'),
                cross_gpu_comparison=arm!='dice',source_archive_sha256=digest(path),
                source_archive_name=path.name,calibration_policy='Saved pilot training-only calibration; retained without retraining')
            self.state['controls' if arm=='dice' else 'results'].append(row)
            self.state['reused_pilot'].append(row)
            copy_verified(path,self.durable/'reused_pilot'/path.name)

    def publish(self,gpu,workers,force=False):
        super().publish(gpu,workers,force)
        rows=self.state.get('results',[])
        summary={}
        for arm in ARMS[1:]:
            rr=[r for r in rows if r['arm']==arm]
            if rr:summary[arm]=dict(completed=len(rr),mean_delta_pp=sum(r['delta_dice_pp'] for r in rr)/len(rr),
                worst_delta_pp=min(r['delta_dice_pp'] for r in rr),positive=sum(r['delta_dice_pp']>0 for r in rr),negative=sum(r['delta_dice_pp']<0 for r in rr))
        self.state['paired_summary']=summary
        write(self.work/'replication_results.json',self.state)
        if force:copy_verified(self.work/'replication_results.json',self.durable/'replication_results.json')

    def run(self):
        try:
            self.reuse_pilot()
            remaining=[pair for pair in sorted(PAIRS) if pair!=(0,0)]
            # Complete one triple at a time: a reset cannot strand all controls without comparisons.
            for seed,fold in remaining:
                self.state.update(status='running_native',current_pair=dict(seed=seed,fold=fold))
                self.phase({'name':'gain_native'},'dice',[(seed,fold)],'controls',False)
                self.load_control(seed,fold)
                self.state['status']='running_constraints'
                # The two constraint arms share the same native control and frozen sample/gain tape.
                self.phase_pair(seed,fold)
            if len(self.state['controls'])!=9 or len(self.state['results'])!=18:raise ValueError('Incomplete27-condition study')
            self.state.update(status='complete_9_triples',next_action='Report all nine effects; no additional combinations launched')
        except SystemExit as error:
            if error.code!=75:raise
            self.state['status']='paused'
        except Exception as error:
            self.state.update(status='error',error=str(error)[-3500:]);raise
        finally:
            for job in self.active.values():
                while job['process'].poll() is None and not (job['output']/'supervisor_ready.json').exists():time.sleep(2)
                if job['process'].poll() is None:job['process'].send_signal(signal.SIGUSR1)
            for job in self.active.values():job['process'].wait()
            self.publish(portable_gpu_sample(),[],True)

    def phase_pair(self,seed,fold):
        source=phase_source()
        source=source.replace('def phase(self, recipe, arm, pairs, bucket, fail_fast=False):',
            "def phase_pair(self,seed,fold):\n    recipe,arm,bucket,fail_fast={'name':'gain_constraints'},'paired_constraints','results',False")
        source=source.replace('jobs=[self.prepare(recipe,arm,*pair) for pair in pairs]',
            "jobs=[self.prepare(None,arm,seed,fold) for arm in ('dice_bands','dice_edge_depth')]")
        ns=dict(parallel.__dict__,ThreeWorkers=ThreeWorkers)
        exec(compile(source,__file__,'exec'),ns)
        return ns['phase_pair'](self,seed,fold)

if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    GainReplication(Path('/content/hippo_hfh')).run()
