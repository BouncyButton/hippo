"""Interpret Adam probes only after exact replay of archived UNETR references."""
import csv
import json
from pathlib import Path
from statistics import median

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import unetr_optimizer_probe_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def metric_replay(actual,reference):
    def rows(folder):
        with (folder/'metrics.csv').open() as file:return list(csv.DictReader(file))
    left,right=rows(actual),rows(reference)
    assert len(left)==len(right)
    maximum=0.
    for a,b in zip(left,right):
        assert set(a)==set(b)
        for key in a:
            if a[key]==b[key]:continue
            error=abs(float(a[key])-float(b[key]));maximum=max(maximum,error)
    assert maximum==0.,('Metric replay differs',actual.name,maximum)
    return dict(checkpoints=len(left),maximum_numeric_error=maximum)


def summarize(probes):
    rows=[r for r in probes if r['active_arm']!='dice' and r['scale']>=1.]
    assert rows
    def values(getter):
        observed=[value for r in rows if (value:=getter(r)) is not None]
        return dict(observations=len(observed),minimum=min(observed) if observed else None,
                    median=median(observed) if observed else None,maximum=max(observed) if observed else None)
    groups={g:median(r['parameter_groups'][g]['auxiliary_induced_over_base_update'] for r in rows)
            for g in rows[0]['parameter_groups']}
    return dict(full_strength_probe_steps=[r['step'] for r in rows],
        logit_auxiliary_over_base=values(lambda r:r['logit_gradients']['auxiliary_over_base']),
        parameter_auxiliary_over_base=values(lambda r:r['parameter_gradients']['auxiliary_over_base']),
        same_state_adam_perturbation_over_base=values(lambda r:r['adam']['auxiliary_induced_over_base_update']),
        parameter_gradient_cosine=values(lambda r:r['parameter_gradients']['cosine']),
        CA4_auxiliary_gradient_cosine=values(lambda r:r['parameter_gradients']['CA4_auxiliary_cosine']),
        groups_median_adam_perturbation=groups,
        base_loss_predicted_ascent_steps=[r['step'] for r in rows if r['first_order_loss_changes']['base']['actual_update']>0],
        auxiliary_reduces_base_descent_steps=[r['step'] for r in rows if r['first_order_loss_changes']['base']['auxiliary_induced_change']>0],
        auxiliary_reduces_CA4_descent_steps=[r['step'] for r in rows if r['first_order_loss_changes']['CA4_base_component']['auxiliary_induced_change']>0],
        caution='Current-training-batch first-order directions and same-state virtual updates, not measured counterfactual validation performance.')


def main():
    report=study.ROOT/'reports'/study.NAME;frozen=report/'frozen';plan=read(frozen/'plan.json')
    rows=[]
    for task in plan['tasks']:
        folder=report/'runs'/task['id']
        if not (folder/'cleanup.json').exists():continue
        reference=frozen/'reference_runs'/task['reference']
        current,prior=evaluation(folder),evaluation(reference)
        cfg,rc=read(folder/'config.json'),read(reference/'config.json')
        for key in ('seed','fold','train_cases','validation_cases','schedule','initial_lr','lr_horizon_updates',
                    'dataset_sha256','objective_source_sha256','augmentation','torch','monai'):
            assert cfg[key]==rc[key],(task['id'],key)
        assert cfg['script_sha256']==digest(frozen/study.CORE)
        assert rc['script_sha256']==digest(frozen/'reference_source/capped_core.py')
        calibration,original_calibration=read(folder/'calibration.json'),read(reference/'calibration.json')
        for key in ('anchor_model_sha256','weights','statistics','anchor_successful_updates','cases'):
            assert calibration[key]==original_calibration[key],(task['id'],'calibration',key)
        model_hash=read(folder/'spatial_100.json')['model_sha256']
        assert model_hash==read(reference/'spatial_100.json')['model_sha256'], 'Diagnostic changed final model'
        replay=metric_replay(folder,reference)
        assert read(folder/'validation_latest.json')==read(reference/'validation_latest.json')
        diagnostics=read(folder/'unetr_optimizer_diagnostics.json');probes=diagnostics['rows']
        assert diagnostics['optimizer_updates_added']==0
        assert [r['step'] for r in probes]==diagnostics['probe_steps']
        maximum_error=max(r['adam']['virtual_vs_actual_relative_error'] for r in probes)
        assert maximum_error<5e-4,('Virtual Adam does not accurately reproduce actual steps',maximum_error)
        rows.append(dict(**task,current=current,reference_evaluation=prior,exact_replay=True,
            replay_metrics=replay,final_model_sha256=model_hash,calibration_weights=calibration['weights'],
            maximum_virtual_adam_relative_error=maximum_error,summary=summarize(probes),
            diagnostic_profiled_flops=diagnostics['total_diagnostic_profiled_flops'],
            diagnostic_seconds=diagnostics['total_diagnostic_seconds']))
    live=read(report/'live.json').get('status.json',{})
    complete=live.get('status')=='complete' and len(rows)==2
    result=dict(status='complete' if complete else 'partial',outcome='diagnostic_replay_verified' if complete else 'running',
        winner=None,rows=rows,summary=[dict(seed=r['seed'],fold=r['fold'],**r['summary']) for r in rows],
        interpretation_limits=['No optimizer or constraint intervention was tested here.',
            'Only exact reference replays are accepted as observational evidence.',
            'Same-state virtual base-only updates retain constrained history and are not an unconstrained training run.',
            'Diagnostics add separately reported compute; do not count these as new cap-matched performance candidates.'],
        analysis_source_sha256=digest(Path(__file__)))
    write(report/'optimizer_probe_results.json',result)
    print(json.dumps(dict(status=result['status'],summary=result['summary'])))
    if complete:plot(report,rows)


def plot(report,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,row in zip(axes,rows):
        probes=read(report/'runs'/row['id']/'unetr_optimizer_diagnostics.json')['rows']
        active=[p for p in probes if p['active_arm']!='dice']
        for field,key,label in [('logit_gradients','auxiliary_over_base','Logit gradient'),
                                ('parameter_gradients','auxiliary_over_base','Parameter gradient'),
                                ('adam','auxiliary_induced_over_base_update','Same-state Adam perturbation')]:
            ax.plot([p['step'] for p in active],[100*p[field][key] for p in active],marker='o',label=label)
        ax.axhline(5,color='#999999',linestyle='--',label='Calibration target (logits)')
        ax.set_title(f"seed{row['seed']} / fold{row['fold']} — exact replay")
        ax.set_xlabel('Training update');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].set_ylabel('Relative magnitude (%)');axes[1].legend(fontsize=8)
    fig.suptitle('UNETR: gradient ratios versus Adam update perturbation');fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(report/f'optimizer_probe_results.{suffix}',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
