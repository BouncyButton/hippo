"""All16 original semantic-constraint combinations, fixed HFH fold0 seed0."""
import hashlib
import inspect
from itertools import combinations
import os
from pathlib import Path
import shutil
import signal
import textwrap
import time

from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_parallel as parallel
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh.medsam_125_campaign import Fixed125,short_protocol
from scripts.colab_hfh.medsam_constraints_policy import STRICT_RECIPES
from scripts.colab_hfh.medsam_mask_only_campaign import aggregate,spatial_rows
from scripts.colab_hfh.medsam_recovery import copy_verified
from scripts.colab_hfh.runtime import read,write,digest

TERMS=('bands','edge','sdf','depth')
ALL_ARMS=tuple('dice_'+'_'.join(c) if c else 'dice'
               for n in range(5) for c in combinations(TERMS,n))
RECIPE=dict(STRICT_RECIPES[0])


def matrix_phase_source():
    source=textwrap.dedent(inspect.getsource(parallel.Campaign.phase))
    head='def phase(self, recipe, arm, pairs, bucket, fail_fast=False):'
    jobs='jobs=[self.prepare(recipe,arm,*pair) for pair in pairs]'
    if source.count(head)!=1 or source.count(jobs)!=1:raise ValueError('Audited scheduler changed')
    source=source.replace(head,"def phase_matrix(self):\n    recipe,arm,bucket,fail_fast=RECIPE,'full16','results',False")
    return source.replace(jobs,"jobs=[self.prepare(RECIPE,arm,0,0) for arm in ALL_ARMS if arm!='dice']")


class Matrix125(Fixed125):
    def __init__(self,root):
        super().__init__(root,hours=8)
        self.maximum=3
        self.work=root/'medsam_matrix125_f0s0_v1'
        self.durable=Path('/content/hfh_storage/results_medsam3_matrix125_f0s0_v1')
        self.work.mkdir(exist_ok=True)
        archives.configure(self.bundle,self.work,self.durable)
        baseline_name='u125_strict_native__dice__f0_s0'
        band_name='u125_strict_gentle_t001_r20__dice_bands__f0_s0'
        for name in (baseline_name,band_name):
            path=root/'medsam_fixed125_v1/compact_archives'/(name+'.zip')
            row=archives.verify_zip(path)
            if (row['seed'],row['fold'],row['updates'])!=(0,0,125):raise ValueError('Wrong reusable run')
            target=archives.CONTEXT['cache']/path.name
            if target.exists() and digest(target)!=digest(path):raise ValueError('Existing imported archive differs')
            if not target.exists():shutil.copyfile(path,target)
            if path.name not in archives.CONTEXT['inventory']:
                copy_verified(target,archives.CONTEXT['remote']/target.name)
                archives.CONTEXT['inventory'][path.name]=dict(Name=path.name,Hashes={'md5':hashlib.md5(target.read_bytes()).hexdigest()})
        self.controls={(0,0):serial.archived_control(archives.CONTEXT['cache']/(baseline_name+'.zip'))}
        control=self.controls[0,0]
        self.baseline_spatial=spatial_rows(control,root)
        old_band=root/'medsam_fixed125_v1/runs'/band_name
        protocol=read(old_band/'protocol.json');protocol.pop('arm')
        verified=self.validate(old_band,protocol,control,'dice_bands',RECIPE)
        archived=archives.verify_zip(archives.CONTEXT['cache']/(band_name+'.zip'))
        if verified!=archived:raise ValueError('Reusable original BAND scientific checks differ')
        if read(old_band/'completion.json')['checkpoint_sha256']!='4c4c9bdef6dc80076c3d05825fe6b39532cfcaa38dee23ed3e5e1f49d2fb5330':
            raise ValueError('Original BAND spatial reference checkpoint differs')
        prior=read(root/'medsam_spatial125_v1/analysis.json')
        self.band_spatial={(r['split'],r['case']):r['band'] for r in prior['records']}
        old_eval=read(old_band/'evaluation.json')['records']
        for r in old_eval:
            s=self.band_spatial[r['split'],r['case_name']]
            if s['fp']!=r['foreground_fp'] or s['fn']!=r['foreground_fn'] or abs(s['dice']-r['macro_hard_dice'])>1e-12:
                raise ValueError('Original BAND spatial metrics differ')
        validate=self.validate

        def checked(output,protocol,control,arm,recipe):
            row=validate(output,protocol,control,arm,recipe)
            new={(r['split'],r['case_name']):r['spatial'] for r in read(output/'evaluation.json')['records']}
            if set(new)!=set(self.baseline_spatial):raise ValueError('Incomplete spatial cases')
            comparison=dict(baseline=aggregate(self.baseline_spatial),constraint=aggregate(new),
                cases=[dict(split=s,case=c,baseline=self.baseline_spatial[s,c],constraint=new[s,c]) for s,c in sorted(new)])
            write(output/'spatial_comparison.json',comparison)
            row['spatial_validation']={a:comparison[a]['validation'] for a in ('baseline','constraint')}
            return row

        self.validate=checked
        self.state=dict(status='preparing_full_matrix',fold=0,seed=0,updates=125,
            learning_rate_horizon_updates=200,arms=list(ALL_ARMS),
            controls=[archives.verify_zip(archives.CONTEXT['cache']/(baseline_name+'.zip'))],results=[],
            recipe=RECIPE,band_formulation='original semantic mask*score*presence',
            gradient_ratio_targets=dict(bands=.01,edge=.1,sdf=.1,depth=.1),
            p95_cap=.5,ramp_updates=20,reused_arms=['dice','dice_bands'],
            scheduler='Adaptive1–3 workers; all arms evaluated independently',
            continuation='Complete all16 combinations; negative Dice gains do not stop other arms',
            geometry=dict(tensor_spacing_mm=[2.,1.,1.],bands='two six-connected voxel steps',
                sdf='Physical EDT clipped8mm',depth='Physical spacing; unverified source-axis orientation'),
            warning='Single-fold single-seed exploratory matrix; no generalization claim. Updates matched, not FLOPs.')
        source=textwrap.dedent(inspect.getsource(parallel.Campaign.launch))
        old="'scripts.colab_hfh.medsam_constraints_worker'"
        if source.count(old)!=1:raise ValueError('Launcher changed')
        source=source.replace(old,"'scripts.colab_hfh.medsam_matrix125_worker'")
        needle='for relative in ('
        if source.count(needle)!=1:raise ValueError('Provenance setup changed')
        source=source.replace(needle,needle+
            "'scripts/colab_hfh/medsam_matrix125_worker.py', 'scripts/colab_hfh/medsam_matrix125_campaign.py', "
            "'scripts/colab_hfh/medsam_125_campaign.py', 'scripts/colab_hfh/medsam_spatial_audit.py', ")
        ns=dict(parallel.__dict__,RECIPE=RECIPE,ALL_ARMS=ALL_ARMS)
        exec(compile(source,__file__,'exec'),ns)
        self.launch=ns['launch'].__get__(self,type(self))
        exec(compile(matrix_phase_source(),__file__,'exec'),ns)
        self.phase_matrix=ns['phase_matrix'].__get__(self,type(self))

    def prepare(self,recipe,arm,seed,fold):
        if (seed,fold)!=(0,0) or arm not in ALL_ARMS:raise ValueError('Matrix scope changed')
        name=f"u125_{recipe['name']}__{arm}__f{fold}_s{seed}"
        output=self.work/'runs'/name;output.mkdir(parents=True,exist_ok=True)
        reference=read(self.root/'medsam_constraints_strict_v1/protocols'/f'strict_native__dice__f{fold}_s{seed}.json')
        control=self.controls[seed,fold]
        protocol=short_protocol(reference,recipe,control)
        # BAND-only stays byte-identical to the completed protocol for reuse.
        if arm!='dice_bands':
            protocol['gradient_ratio_targets']={k:(.01 if k=='bands' else .1) for k in TERMS if k in arm.split('_')}
            if 'sdf' in arm or 'depth' in arm:
                protocol['constraint_geometry']=self.state['geometry']
        path=self.work/'protocols'/(name+'.json')
        if path.exists() and read(path)!=protocol:raise ValueError('Existing matrix protocol differs')
        write(path,protocol)
        return dict(name=name,output=output,remote=self.durable/'runs'/name,control=control,
                    protocol=protocol,path=path,recipe=recipe,arm=arm,seed=seed,fold=fold)

    def existing(self,job):
        row=super().existing(job)
        if row is not None and job['arm']=='dice_bands':
            row['spatial_validation']=dict(baseline=aggregate(self.baseline_spatial)['validation'],
                                           constraint=aggregate(self.band_spatial)['validation'])
            row['reused']=True
        return row

    def run(self):
        try:
            self.state['status']='running_full_matrix'
            rows=self.phase_matrix()
            if len(rows)!=15 or {r['arm'] for r in rows}!=set(ALL_ARMS)-{'dice'}:
                raise ValueError('Full matrix is incomplete')
            ranked=sorted(rows,key=lambda r:r['validation_dice'],reverse=True)
            self.state.update(status='complete_16_arms',ranking=ranked,
                improved_arms=[r['arm'] for r in ranked if r['delta_dice_pp']>0])
            write(self.work/'matrix_results.json',self.state)
            copy_verified(self.work/'matrix_results.json',self.durable/'matrix_results.json')
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
            self.publish(parallel.gpu_sample(),[],force=True)


if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    Matrix125(Path('/content/hippo_hfh')).run()
