"""Exactly one fresh native/EDGE+DEPTH pair, with no automatic seed expansion."""
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

import numpy as np
from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_parallel as parallel
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh.medsam_constraints_capacity import Capacity
from scripts.colab_hfh.medsam_component_constraints_campaign import portable_gpu_sample
from scripts.colab_hfh.medsam_125_campaign import validator125, local_control
from scripts.colab_hfh.medsam_coverage125 import CAMPAIGN, RECIPE, make_protocol, hardware_identity
from scripts.colab_hfh.medsam_recovery import copy_verified, rclone
from scripts.colab_hfh.runtime import read, write, digest


class CoveragePair(parallel.Campaign):
    def __init__(self, root):
        self.root, self.bundle = root, root/'bundle'
        self.work = root/CAMPAIGN
        self.durable = Path('/content/hfh_storage/results_medsam3_coverage125_f0s0_v1')
        self.work.mkdir(exist_ok=True)
        self.handles = []
        for path in (self.work/'controller.lock', root/'work_medsam3/controller.lock'):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handles.append(handle)
        parallel.gpu_sample = portable_gpu_sample
        portable_gpu_sample()
        self.hardware = hardware_identity()
        quota = json.loads(rclone('about','hfh_storage:','--json'))
        if min(quota.get('free',0),shutil.disk_usage(root).free) < 8*1024**3:
            raise RuntimeError('Need 8 GiB free local and second-account storage')
        reference = self.work/'controls/reference.zip'
        if digest(reference) != read(self.work/'reference_manifest.json')['sha256']:
            raise ValueError('Historical recipe reference changed')
        self.reference = serial.archived_control(reference)
        foreground = {}
        for case in self.reference['protocol']['train_cases']:
            with np.load(self.bundle/'data'/(case+'.npz')) as data:
                foreground[case] = (data['label']>0).any(axis=(0,2)).tolist()
        self.foreground = foreground
        self.deadline = time.time()+4*3600
        self.maximum,self.stopping,self.active = 1,False,{}
        self.tuner,self.phase_failure,self.last_publish = Capacity(maximum=1),None,0
        self.validate = validator125()
        self.control = None
        archives.configure(self.bundle,self.work,self.durable)
        self.state = dict(status='running_coverage_native',updates=125,fold=0,seed=0,
            controls=[],results=[],hardware=self.hardware,augmentation='none',
            scope='One fresh matched coverage-first native/EDGE+DEPTH pair; no other folds or seeds',
            warning='Exploratory validation; equal optimizer updates, auxiliary/calibration costs separate',
            historical_native=dict(train=self.reference['completion']['training_dice'],val=self.reference['completion']['validation_dice'],
                comparison='Historical reference only; hardware differs, not the matched baseline'))
        source = textwrap.dedent(inspect.getsource(parallel.Campaign.launch))
        old = "'scripts.colab_hfh.medsam_constraints_worker'"
        assert source.count(old)==1
        source = source.replace(old,"'scripts.colab_hfh.medsam_coverage125_worker'")
        extra = ('scripts/colab_hfh/medsam_coverage125.py','scripts/colab_hfh/medsam_coverage125_worker.py',
            'scripts/colab_hfh/medsam_coverage125_campaign.py','scripts/colab_hfh/medsam_native125_logged.py',
            'scripts/colab_hfh/medsam_augmentation_components.py','scripts/medsam3_supervised_constraints.py',
            'scripts/depth_supervision/geometry.py')
        source = source.replace('for relative in (','for relative in ('+','.join(repr(x) for x in extra)+',')
        ns = dict(parallel.__dict__)
        exec(compile(source,__file__,'exec'),ns)
        self.launch = ns['launch'].__get__(self,type(self))
        signal.signal(signal.SIGTERM,self.stop)
        signal.signal(signal.SIGINT,self.stop)

    def prepare(self, recipe, arm, seed, fold):
        if (seed,fold)!=(0,0):
            raise ValueError('No additional seeds/folds authorized for this pilot')
        p = make_protocol(self.reference['protocol'],self.foreground,arm,self.hardware)
        control = self.control
        if arm!='dice':
            if control is None:
                raise ValueError('Fresh same-GPU native result required first')
            p['baseline_archive_sha256'] = control['archive_sha256']
        name = 'coverage125__'+arm+'__f0_s0'
        output=self.work/'runs'/name
        output.mkdir(parents=True,exist_ok=True)
        path=self.work/'protocols'/(name+'.json')
        if path.exists() and read(path)!=p:
            raise ValueError('Existing coverage protocol differs; preserve it')
        write(path,p)
        return dict(name=name,output=output,remote=self.durable/'runs'/name,control=control,
            protocol=p,path=path,recipe=p['constraint_recipe'],arm=arm,seed=0,fold=0)

    def commit(self,job):
        output,p=job['output'],job['protocol']
        control=job['control']
        if job['arm']=='dice':
            control=local_control(output)
            control['archive_sha256']=None
        row=self.validate(output,p,control,job['arm'],job['recipe'])
        if read(output/'execution_hardware.json')!=self.hardware:
            raise ValueError('Actual worker hardware differs from matched protocol')
        if job['arm']!='dice' and control['protocol']['paired_execution']!=self.hardware:
            raise ValueError('Native ran on different hardware')
        if digest(output/'checkpoint_latest.pt')!=read(output/'completion.json')['checkpoint_sha256']:
            raise ValueError('Final checkpoint integrity failed')
        preflight=read(output/'loss_logging_preflight.json')
        if preflight['status']!='passed' or not all(preflight[k] for k in ('exact_loss','exact_gradients','exact_rng')):
            raise ValueError('Loss logging changed optimization')
        history=read(output/'training_history.json')
        keys=['native_term/weighted/'+k for k in preflight['weights']]
        if any(not all(k in h and math.isfinite(h[k]) for k in keys) or
            not math.isclose(sum(h[k] for k in keys),h['native'],rel_tol=1e-5,abs_tol=1e-3) for h in history):
            raise ValueError('Native components do not reconstruct training loss')
        records=read(output/'evaluation.json')['records']
        truth={(r['split'],r['case_name']):r['truth_sha256'] for r in self.reference['evaluation']['records']}
        if any(r['truth_sha256']!=truth[r['split'],r['case_name']] for r in records):
            raise ValueError('Evaluation truth differs from original split')
        if read(output/'coverage_sampling.json')!=p['coverage_sampling']:
            raise ValueError('Coverage audit changed')
        row.update(hardware=self.hardware,cross_gpu_comparison=False,augmentation='none',
            training_gain_pp=100*(row['training_dice']-row['baseline_training_dice']),
            coverage_sampling=p['coverage_sampling'],fresh_calibration=True)
        row['metrics_by_split']={}
        for split in ('training_support','validation'):
            rr=[r for r in records if r['split']==split]
            row['metrics_by_split'][split]=dict(dice=sum(r['macro_hard_dice'] for r in rr)/len(rr),
                fp=sum(r['foreground_fp'] for r in rr),fn=sum(r['foreground_fn'] for r in rr))
        write(output/'result.json',row)
        write(output/'paired_result.json',row)
        files=[f for f in output.iterdir() if f.is_file() and f.suffix in ('.json','.log','.csv') and f.name!='archive_manifest.json']
        write(output/'archive_manifest.json',{f.name:digest(f) for f in files})
        path=archives.CONTEXT['cache']/(job['name']+'.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for f in files+[output/'archive_manifest.json']:
                z.write(f,'results/'+f.name)
            for name in ('medsam_coverage125.py','medsam_coverage125_worker.py','medsam_coverage125_campaign.py'):
                z.write(Path(__file__).with_name(name),'source/'+name)
        if archives.verify_zip(path)!=row:
            raise ValueError('Local archive validation failed')
        copy_verified(path,archives.CONTEXT['remote']/path.name)
        archives.CONTEXT['inventory'][path.name]=dict(Name=path.name,Hashes={'md5':hashlib.md5(path.read_bytes()).hexdigest()})
        print('COVERAGE125_FINAL',json.dumps(row),flush=True)
        return row

    def run(self):
        try:
            self.phase(RECIPE,'dice',[(0,0)],'controls')
            path=archives.CONTEXT['cache']/'coverage125__dice__f0_s0.zip'
            self.control=serial.archived_control(path)
            self.state['status']='running_coverage_edge_depth'
            self.phase(RECIPE,'dice_edge_depth',[(0,0)],'results')
            self.state.update(status='complete_coverage_pair',next_action='Review clean train/val Dice and FP/FN before authorizing any expansion')
            native=self.state['controls'][0]['metrics_by_split']
            constrained=self.state['results'][0]['metrics_by_split']
            self.state['comparison']={s:dict(native=native[s],edge_depth=constrained[s],
                gain_pp=100*(constrained[s]['dice']-native[s]['dice']),
                fp_delta=constrained[s]['fp']-native[s]['fp'],fn_delta=constrained[s]['fn']-native[s]['fn']) for s in native}
            write(self.work/'pair_results.json',self.state)
            copy_verified(self.work/'pair_results.json',self.durable/'pair_results.json')
        except SystemExit as error:
            if error.code!=75:raise
            self.state['status']='paused'
        except Exception as error:
            self.state.update(status='error',error=str(error)[-3500:])
            raise
        finally:
            for job in self.active.values():
                while job['process'].poll() is None and not (job['output']/'supervisor_ready.json').exists():
                    time.sleep(2)
                if job['process'].poll() is None:job['process'].send_signal(signal.SIGUSR1)
            for job in self.active.values():job['process'].wait()
            self.publish(portable_gpu_sample(),[],force=True)


if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    CoveragePair(Path('/content/hippo_hfh')).run()
