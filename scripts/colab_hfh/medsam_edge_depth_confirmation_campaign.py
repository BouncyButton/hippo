"""Confirm a selected, unchanged gain EDGE+DEPTH recipe on seeds 1 and 2."""
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.colab_hfh.medsam_gain_replication_campaign import GainReplication, portable_gpu_sample
from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh.medsam_recovery import copy_verified
from scripts.colab_hfh.runtime import read, write, digest

PAIRS=tuple((seed,fold) for seed in (1,2) for fold in (0,1,2))


def check_plan(plan):
    if plan.get('selection_status')!='selected_for_confirmation':
        raise ValueError('Wait for seed-0 evidence and an explicit candidate selection')
    if plan.get('candidate')!='original_gain_edge_depth' or plan.get('pairs')!=[list(p) for p in PAIRS]:
        raise ValueError('Confirmation scope changed')
    if plan.get('reuse_all_existing_results') is not True or plan.get('maximum_active_workers')!=1:
        raise ValueError('Existing results and sequential execution must be preserved')
    if set(plan.get('seed0_archives',{}))!={'0','1','2'}:
        raise ValueError('Three verified seed-0 archives required')


class EdgeDepthConfirmation(GainReplication):
    def __init__(self,root):
        super().__init__(root)
        self.original_launch=self.launch
        self.launch=self.launch_missing
        self.plan=read(self.work/'edge_depth_confirmation_plan.json')
        check_plan(self.plan)
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
        if gpu!='NVIDIA L4':raise ValueError('Use the requested L4')
        for fold in range(3):
            path=self.pilot_paths['dice_edge_depth'] if fold==0 else archives.CONTEXT['cache']/f'gain125__dice_edge_depth__f{fold}_s0.zip'
            result=archives.verify_zip(path)
            if digest(path)!=self.plan['seed0_archives'][str(fold)] or (result['seed'],result['fold'],result['arm'])!=(0,fold,'dice_edge_depth'):
                raise ValueError('Selected seed-0 evidence changed')
        self.state=read(self.work/'campaign_status.json')
        if self.state.get('workers'):raise ValueError('Previous worker must finish or pause cleanly')
        # Completed confirmation archives are reattached by phase(), not retrained
        # or duplicated in the live summary after a controller restart.
        for bucket in ('controls','results'):
            self.state[bucket]=[r for r in self.state[bucket] if (r['seed'],r['fold']) not in PAIRS]
        self.state.update(status='starting_edge_depth_confirmation',error=None,workers=[],
            arms=['dice','dice_edge_depth'],requested_workers=1,queued_folds=[],
            confirmation_pairs=[list(p) for p in PAIRS],new_runs_at_most=12,
            continuation_scope='Only missing original gain native/EDGE+DEPTH pairs for seeds1/2; reuse every completed archive; no BAND or recipe search',
            automatic_expansion=False)

    def prepare(self,recipe,arm,seed,fold):
        if arm not in ('dice','dice_edge_depth') or (seed,fold) not in PAIRS:
            raise ValueError('Outside the frozen confirmation scope')
        return super().prepare(recipe,arm,seed,fold)

    def launch_missing(self,job):
        # A completed archive is an unconditional reuse gate, never a reason
        # to launch a replacement experiment.
        if job['name']+'.zip' in archives.CONTEXT['inventory']:
            raise ValueError('Refusing to rerun a completed condition')
        return self.original_launch(job)

    def run(self):
        try:
            for i,(seed,fold) in enumerate(PAIRS):
                self.state.update(status='running_confirmation_control',current_pair=dict(seed=seed,fold=fold),
                    queued_job=dict(arm='dice_edge_depth',seed=seed,fold=fold))
                self.phase({'name':'confirmation_native'},'dice',[(seed,fold)],'controls',False)
                self.load_control(seed,fold)
                next_pair=PAIRS[i+1] if i+1<len(PAIRS) else None
                self.state.update(status='running_confirmation_constraints',
                    queued_job=dict(arm='dice',seed=next_pair[0],fold=next_pair[1]) if next_pair else None)
                self.phase({'name':'confirmation_edge_depth'},'dice_edge_depth',[(seed,fold)],'results',False)
            rows=[r for r in self.state['results'] if r['arm']=='dice_edge_depth']
            if len(rows)!=9 or {(r['seed'],r['fold']) for r in rows}!={(s,f) for s in range(3) for f in range(3)}:
                raise ValueError('Incomplete nine-pair confirmation')
            self.state.update(status='complete_edge_depth_9pairs',queued_job=None,
                next_action='Analyze every paired effect and uncertainty; computation complete does not imply recipe success')
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
            (self.work/'STOP').touch()
            copy_verified(Path(__file__),self.durable/'edge_depth_confirmation_driver.py')
            self.publish(portable_gpu_sample(),[],True)


if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    EdgeDepthConfirmation(Path('/content/hippo_hfh')).run()
