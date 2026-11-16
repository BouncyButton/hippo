"""Fixed125 paired BAND study; reuse pilot, then finish nine pairs if it improves."""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import time
import zipfile

from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh.medsam_constraints_parallel import Campaign, gpu_sample
from scripts.colab_hfh.medsam_constraints_policy import STRICT_CONTROL, STRICT_RECIPES
from scripts.colab_hfh.medsam_recovery import copy_verified
from scripts.colab_hfh.runtime import read, write, digest

PAIRS = tuple((seed, fold) for fold in range(3) for seed in range(3))
BAND = STRICT_RECIPES[0]


def short_protocol(reference, recipe, control=None):
    if reference['updates'] != 200 or len(reference['frozen_learning_rates']) != 200:
        raise ValueError('Expected original frozen 200-update reference')
    p = copy.deepcopy(reference)
    p.pop('arm', None)
    p.update(updates=125, learning_rate_horizon_updates=200,
        frozen_sample_schedule=p['frozen_sample_schedule'][:250],
        constraint_recipe=recipe, gradient_ratio_targets={'bands':recipe['target']},
        constraint_warmup_updates=recipe['ramp'],
        comparison='Fixed125 paired exploratory study; original200 LR/sample prefix; calibration costs separate')
    if control is not None:
        p['baseline_archive_sha256'] = control['archive_sha256']
        q = control['protocol']
        if (q['updates'] != 125 or q['seed'] != p['seed'] or q['fold'] != p['fold']
                or q['frozen_sample_schedule'] != p['frozen_sample_schedule']
                or q['frozen_learning_rates'] != p['frozen_learning_rates']):
            raise ValueError('Matched125 control identity or training trajectory differs')
    return p


def validator125():
    """Keep all audited pairing checks, changing only the explicit update budget."""
    source = inspect.getsource(serial.paired_result)
    replacements = (
        ("done['updates'] != 200", "done['updates'] != 125"),
        ("len(history) != 200", "len(history) != 125"),
        ("[r['lr'] for r in history] != protocol['frozen_learning_rates']",
         "[r['lr'] for r in history] != protocol['frozen_learning_rates'][:125]"),
        ('updates=200, training_dice=', 'updates=125, training_dice='),
        ('200 matched optimizer updates;', '125 matched optimizer updates;'),
        ('range(200)', 'range(125)'),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Audited paired validator changed: '+old)
        source=source.replace(old,new)
    namespace=dict(serial.__dict__)
    exec(compile(source,__file__,'exec'),namespace)
    def validate(output, protocol, control, arm, recipe):
        if protocol['updates'] != 125 or control['protocol']['updates'] != 125 or control['completion']['updates'] != 125:
            raise ValueError('Both sides of the comparison must use125 updates')
        return namespace['paired_result'](output, protocol, control, arm, recipe)
    return validate


def local_control(output):
    return {name:read(output/(name+'.json')) for name in (
        'protocol','training_history','slab_schedule','training_summary','completion',
        'evaluation','training_inventory','environment')}


class Fixed125(Campaign):
    def __init__(self, root, hours=4):
        # Reuse the controller locks, original reference verification and
        # resource safety checks. Neither original campaign is resumed.
        super().__init__(root,hours,3)
        self.historical=self.controls
        self.controls={}
        self.work=root/'medsam_fixed125_v1'
        self.durable=Path('/content/hfh_storage/results_medsam3_fixed125_v1')
        self.work.mkdir(exist_ok=True)
        archives.configure(self.bundle,self.work,self.durable)
        self.validate=validator125()
        self.state=dict(status='preparing_pilot',controls=[],results=[],updates=125,
            learning_rate_horizon_updates=200,first_pair_gate='Positive validation gain and BAND Dice >=50%',
            continuation='After first-pair success, complete all nine pairs and report every regression',
            recipe=BAND,warning='Validation-selected stopping budget and recipe; exploratory only')

    def prepare(self,recipe,arm,seed,fold):
        name=f"u125_{recipe['name']}__{arm}__f{fold}_s{seed}"
        output=self.work/'runs'/name
        output.mkdir(parents=True,exist_ok=True)
        reference=read(self.root/'medsam_constraints_strict_v1/protocols'/f'strict_native__dice__f{fold}_s{seed}.json')
        control=self.controls.get((seed,fold))
        protocol=short_protocol(reference,recipe,control if arm!='dice' else None)
        if arm=='dice' and (seed,fold)==(0,0):
            pilot=self.root/'medsam_budget_pilot_f0s0_v1/runs/native_stop_u125_f0_s0'
            done=read(pilot/'completion.json')
            if done['updates']!=125 or done['checkpoint_sha256']!=digest(pilot/'checkpoint_latest.pt'):
                raise ValueError('Completed125 pilot must be preserved and reused')
            protocol=read(pilot/'protocol.json'); protocol.pop('arm',None)
            if not (output/'completion.json').exists():
                for path in pilot.iterdir():
                    if path.is_file() and (path.suffix in ('.json','.log','.csv') or path.name=='checkpoint_latest.pt'):
                        shutil.copyfile(path,output/path.name)
        if arm!='dice' and control is None:
            raise ValueError('Cannot run BAND without its matched125 control')
        path=self.work/'protocols'/(name+'.json')
        if path.exists() and read(path)!=protocol:
            raise ValueError('Existing fixed125 protocol changed')
        write(path,protocol)
        return dict(name=name,output=output,remote=self.durable/'runs'/name,control=control,
                    protocol=protocol,path=path,recipe=recipe,arm=arm,seed=seed,fold=fold)

    def commit(self,job):
        output=job['output']
        control=job['control']
        if job['arm']=='dice':
            control=local_control(output)
            control['archive_sha256']=None
        row=self.validate(output,job['protocol'],control,job['arm'],job['recipe'])
        if digest(output/'checkpoint_latest.pt')!=read(output/'completion.json')['checkpoint_sha256']:
            raise ValueError('Completed checkpoint differs')
        write(output/'paired_result.json',row);write(output/'result.json',row)
        files=[p for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.log','.csv')
               and p.name!='archive_manifest.json']
        write(output/'archive_manifest.json',{p.name:digest(p) for p in files})
        path=archives.CONTEXT['cache']/(job['name']+'.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as zipped:
            for p in files+[output/'archive_manifest.json']:zipped.write(p,'results/'+p.name)
            zipped.write(__file__,'source/medsam_125_campaign.py')
        if archives.verify_zip(path)!=row:raise ValueError('Local archive verification failed')
        copy_verified(path,archives.CONTEXT['remote']/path.name)
        archives.CONTEXT['inventory'][path.name]=dict(Name=path.name,Hashes={'md5':hashlib.md5(path.read_bytes()).hexdigest()})
        print('FIXED125_PAIRED_FINAL',json.dumps(row),flush=True)
        return row

    def load_control(self,seed,fold):
        name=f"u125_{STRICT_CONTROL['name']}__dice__f{fold}_s{seed}.zip"
        self.controls[seed,fold]=serial.archived_control(archives.CONTEXT['cache']/name)

    def run(self):
        try:
            self.state['status']='preparing_pilot_control'
            self.phase(STRICT_CONTROL,'dice',[(0,0)],'controls')
            self.load_control(0,0)
            self.state['status']='running_band_pilot'
            first=self.phase(BAND,'dice_bands',[(0,0)],'results')
            if len(first)!=1 or first[0]['delta_dice_pp']<=0 or first[0]['validation_dice']<.5:
                self.state['status']='pilot_no_improvement';return
            remaining=[pair for pair in PAIRS if pair!=(0,0)]
            self.state['status']='running_remaining_controls'
            self.phase(STRICT_CONTROL,'dice',remaining,'controls')
            for pair in remaining:self.load_control(*pair)
            self.state['status']='running_remaining_bands'
            self.phase(BAND,'dice_bands',remaining,'results')
            rows=self.state['results']
            if len(rows)!=9 or {(r['seed'],r['fold']) for r in rows}!=set(PAIRS):
                raise ValueError('Incomplete fixed125 nine-pair campaign')
            self.state.update(status='complete_9_pairs',summary=dict(
                mean_delta_pp=statistics.mean(r['delta_dice_pp'] for r in rows),
                minimum_delta_pp=min(r['delta_dice_pp'] for r in rows),
                positive_pairs=sum(r['delta_dice_pp']>0 for r in rows),
                all_band_dice_above_50=all(r['validation_dice']>=.5 for r in rows)))
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
            self.publish(gpu_sample(),[],force=True)


if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    Fixed125(Path('/content/hippo_hfh')).run()
