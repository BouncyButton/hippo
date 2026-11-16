"""Sequential fixed125 mask-only BAND pairs; stop at first non-improvement."""
import inspect
import json
import os
from pathlib import Path
import signal
import statistics
import time

from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_parallel as parallel
from scripts.colab_hfh.medsam_125_campaign import Fixed125, PAIRS
from scripts.colab_hfh.medsam_constraints_policy import STRICT_CONTROL
from scripts.colab_hfh.medsam_constraints_mask_only import FORMULATION
from scripts.colab_hfh.medsam_recovery import copy_verified
from scripts.colab_hfh.runtime import read, write, digest

RECIPE = dict(name='mask_only_t001_r20_v1', target=.01, ramp=20,
              numerics='strict_empty_grid_v1', formulation=FORMULATION,
              positive_query='native_one_to_one_match',
              empty_query='detached_pixelwise_semantic_winner',
              inference='unchanged_mask_times_score_times_presence')


def improves(row):
    import math
    return (math.isfinite(row['delta_dice_pp']) and row['delta_dice_pp'] > 0
            and math.isfinite(row['validation_dice']) and .5 <= row['validation_dice'] <= 1)


def spatial_rows(control, root):
    rows = control['evaluation']['records']
    if all('spatial' in r for r in rows):
        return {(r['split'],r['case_name']):r['spatial'] for r in rows}
    expected = '3d0e332535d1ea1346e2f546431e82f3c106acfad342a866f2f9cd7fcc4784a1'
    if control['completion']['checkpoint_sha256'] != expected:
        raise ValueError('Missing spatial reference for a non-pilot control')
    audit = read(root/'medsam_spatial125_v1/analysis.json')
    if audit['status'] != 'complete' or not audit['all_prediction_hashes_verified']:
        raise ValueError('Pilot spatial audit is incomplete')
    result = {(r['split'],r['case']):r['baseline'] for r in audit['records']}
    for r in rows:
        s = result[r['split'],r['case_name']]
        if s['fp'] != r['foreground_fp'] or s['fn'] != r['foreground_fn'] or abs(s['dice']-r['macro_hard_dice'])>1e-12:
            raise ValueError('Pilot spatial metrics differ from saved baseline')
    return result


def aggregate(rows):
    return {split:{
        **{k:sum(v[k] for (s,c),v in rows.items() if s==split)
           for k in ('fp','fn','tp','fp_on_empty_slices','detached_component_fp')},
        'regions':{k:sum(v['errors_by_region'][k] for (s,c),v in rows.items() if s==split)
                   for k in ('inner','core','outer','distant_background')}}
        for split in ('training_support','validation')}


class MaskOnly125(Fixed125):
    def __init__(self, root):
        super().__init__(root, hours=8)
        self.maximum = 1
        self.work = root/'medsam_mask_only125_v1'
        self.durable = Path('/content/hfh_storage/results_medsam3_mask_only125_v1')
        self.work.mkdir(exist_ok=True)
        archives.configure(self.bundle,self.work,self.durable)
        validate = self.validate

        def checked(output, protocol, control, arm, recipe):
            row = validate(output, protocol, control, arm, recipe)
            if arm != 'dice':
                old = spatial_rows(control,self.root)
                new = {(r['split'],r['case_name']):r['spatial'] for r in read(output/'evaluation.json')['records']}
                if set(old)!=set(new):raise ValueError('Spatial comparison case mismatch')
                comparison = dict(baseline=aggregate(old), mask_only=aggregate(new),
                    checkpoint_sha256=read(output/'completion.json')['checkpoint_sha256'],
                    baseline_checkpoint_sha256=control['completion']['checkpoint_sha256'],
                    cases=[dict(split=s,case=c,baseline=old[s,c],mask_only=new[s,c]) for s,c in sorted(new)])
                write(output/'spatial_comparison.json',comparison)
                row['spatial_validation'] = {a:comparison[a]['validation'] for a in ('baseline','mask_only')}
            return row

        self.validate = checked
        self.state.update(status='preparing', recipe=RECIPE,
            continuation='Sequential pairs; stop immediately after first nonpositive validation gain or Dice below50%',
            scheduler='One worker; no later pair starts before previous pair passes')
        # Keep existing process/recovery semantics; change only worker entrypoint.
        source = inspect.getsource(parallel.Campaign.launch)
        import textwrap
        source = textwrap.dedent(source)
        old = "'scripts.colab_hfh.medsam_constraints_worker'"
        if source.count(old)!=1:raise ValueError('Parallel launcher changed')
        source = source.replace(old,"'scripts.colab_hfh.medsam_mask_only_worker'")
        needle = 'for relative in ('
        if source.count(needle)!=1:raise ValueError('Launcher provenance changed')
        source = source.replace(needle, needle +
            "'scripts/colab_hfh/medsam_mask_only_worker.py', "
            "'scripts/colab_hfh/medsam_mask_only_campaign.py', "
            "'scripts/colab_hfh/medsam_125_campaign.py', "
            "'scripts/colab_hfh/medsam_spatial_audit.py', ")
        ns = dict(parallel.__dict__)
        exec(compile(source,__file__,'exec'),ns)
        self.launch = ns['launch'].__get__(self,type(self))

    def run(self):
        try:
            for seed,fold in PAIRS:
                self.state.update(status='running_matched_control',pair=dict(seed=seed,fold=fold))
                self.phase(STRICT_CONTROL,'dice',[(seed,fold)],'controls')
                self.load_control(seed,fold)
                self.state['status']='running_mask_only_band'
                rows=self.phase(RECIPE,'dice_bands',[(seed,fold)],'results')
                if len(rows)!=1:raise ValueError('Expected exactly one completed pair')
                if not improves(rows[0]):
                    self.state.update(status='stopped_no_improvement',failed_pair=rows[0]);return
            rows=self.state['results']
            self.state.update(status='complete_9_pairs',summary=dict(
                mean_delta_pp=statistics.mean(r['delta_dice_pp'] for r in rows),
                minimum_delta_pp=min(r['delta_dice_pp'] for r in rows),positive_pairs=len(rows)))
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
    MaskOnly125(Path('/content/hippo_hfh')).run()
