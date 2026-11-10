"""Verify and summarize every cell of the fixed-definition UNETR pilot."""
import csv
import json
from pathlib import Path
from statistics import median

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.unetr_constraint_pilot import ROOT, NAME, PARENT, BASELINE, CONDITIONS, CAP


def evaluation(folder):
    manifest = read(folder/'archive_manifest.json')
    verify_archive(folder, manifest)
    assert read(folder/'cleanup.json')['status'] == 'complete'
    config, completion = read(folder/'config.json'), read(folder/'completion.json')
    assert config['budget_flops'] == CAP and completion['discarded_attempts'] == 0
    assert completion['updates'] == config['max_steps'] == 409
    assert completion['consumed_training_flops'] <= CAP
    data = read(folder/'validation_latest.json')
    matrix = [[sum(c['confusion'][i][j] for c in data['cases']) for j in range(4)] for i in range(4)]
    return dict(validation_percent=100*data['summary']['macro_hard_dice'],
                training_percent=100*read(folder/'training_final.json')['summary']['macro_hard_dice'],
                union_percent=100*data['summary']['whole_hippocampus_hard_dice'],
                per_subfield_percent={k:100*v for k,v in data['summary'].items()
                                      if k in ('CA1_3_hard_dice','subiculum_hard_dice','CA4_DG_hard_dice')},
                fn=sum(matrix[i][0] for i in range(1,4)), fp=sum(matrix[0][1:]),
                swaps=sum(matrix[i][j] for i in range(1,4) for j in range(1,4) if i!=j),
                updates=completion['updates'], optimization_flops=completion['consumed_training_flops'],
                archive_manifest_sha256=digest(folder/'archive_manifest.json'))


def main():
    report = ROOT/'reports'/NAME
    baseline_folder = ROOT/'reports'/PARENT/'runs'/BASELINE
    baseline = evaluation(baseline_folder)
    baseline_config = read(baseline_folder/'config.json')
    rows = []
    for condition, (target,prefix) in CONDITIONS.items():
        folder = report/'runs'/f'MNI_unetr_{condition}_f0s19'
        if not (folder/'cleanup.json').exists():
            continue
        current = evaluation(folder)
        config = read(folder/'config.json')
        for key in ('seed','fold','train_cases','validation_cases','initial_lr','lr_horizon_updates',
                    'dataset_sha256','script_sha256','objective_source_sha256','augmentation'):
            assert config[key] == baseline_config[key], key
        calibration = read(folder/'calibration.json')
        assert calibration['target_ratio'] == target and calibration['anchor_successful_updates'] == prefix
        assert len(calibration['cases']) == 20 and calibration['statistics']['bands']['weighted_p95_ratio'] <= .7500001
        quarters = []
        for quarter in (25,50,75,100):
            spatial = read(folder/f'spatial_{quarter:03d}.json')
            training = [c for c in spatial['cases'] if c['split']=='training']
            assert len(training) == 20
            ratio = median(c['gradient_probe']['original_bands']['weighted_norm_ratio'] for c in training)
            ramp = min(1.,max(0.,(spatial['charged_updates']-prefix)/33))
            quarters.append(dict(quarter=quarter,updates=spatial['charged_updates'],
                full_coefficient_median_logit_ratio=ratio,
                ramp_at_last_training_update=ramp,
                ramp_scaled_median_logit_ratio=ratio*ramp,
                scope='Unaugmented eval-mode training crops; not parameter gradients or Adam updates.'))
        rows.append(dict(condition=condition,target=target,prefix_updates=prefix,**current,
            gain_vs_baseline_pp=current['validation_percent']-baseline['validation_percent'],
            error_delta_vs_baseline={k:current[k]-baseline[k] for k in ('fn','fp','swaps')},
            calibration_weight=calibration['weights']['bands'],calibration_statistics=calibration['statistics']['bands'],
            calibration_anchor_sha256=calibration['anchor_model_sha256'],gradient_probes=quarters))
    factorial = None
    if len(rows)==4:
        by_name={r['condition']:r for r in rows}
        for onset in ('original','delayed'):
            assert by_name[onset+'_t05']['calibration_anchor_sha256'] == by_name[onset+'_t20']['calibration_anchor_sha256']
        y={r['condition']:r['validation_percent'] for r in rows}
        factorial=dict(strength_effect_original_onset_pp=y['original_t20']-y['original_t05'],
                       strength_effect_delayed_onset_pp=y['delayed_t20']-y['delayed_t05'],
                       delay_effect_target05_pp=y['delayed_t05']-y['original_t05'],
                       delay_effect_target20_pp=y['delayed_t20']-y['original_t20'],
                       interaction_pp=(y['delayed_t20']-y['delayed_t05'])-(y['original_t20']-y['original_t05']))
    result=dict(status='complete' if len(rows)==4 else 'partial',baseline=baseline,rows=rows,
        factorial_effects=factorial,
        scope='Seed19 fold0,n10,original union-band definition,409 updates,53.305T optimization cap. Calibration separately charged. Final-budget results, not best validation.',
        limitations=['Single development fold/seed; no generalization claim from selecting the best cell.',
                     'Delayed onset also delays calibration; cannot separate those effects.',
                     'Gradient probes measure logits in eval mode, not actual optimizer updates.'])
    write(report/'pilot_results.json',result)
    print(json.dumps(dict(status=result['status'],baseline=baseline['validation_percent'],
        results=[{k:r[k] for k in ('condition','validation_percent','training_percent','gain_vs_baseline_pp','calibration_weight')} for r in rows],
        factorial=factorial)))
    if len(rows)==4:
        plot(report, baseline_folder, rows)


def plot(report, baseline_folder, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    names = {'Baseline':'Baseline','original_t05':'Original onset, 5%',
             'original_t20':'Original onset, 20%','delayed_t05':'Delayed onset, 5%',
             'delayed_t20':'Delayed onset, 20%'}
    colors = {name:f'C{i}' for i,name in enumerate(names)}
    folders=[('Baseline',baseline_folder)]+[(r['condition'],report/'runs'/f"MNI_unetr_{r['condition']}_f0s19") for r in rows]
    for label,folder in folders:
        with (folder/'metrics.csv').open() as file:
            curve=list(csv.DictReader(file))
        axes[0].plot([int(r['charged_attempts']) for r in curve],[100*float(r['macro_hard_dice']) for r in curve],label=names[label],color=colors[label])
    for row in rows:
        axes[1].plot([r['updates'] for r in row['gradient_probes']],
                     [100*r['ramp_scaled_median_logit_ratio'] for r in row['gradient_probes']],marker='o',label=names[row['condition']],color=colors[row['condition']])
    axes[0].set(xlabel='Training updates',ylabel='Validation macro Dice (%)',title='Matched seed 19 / fold 0 at 53.305T')
    axes[1].set(xlabel='Training updates',ylabel='Median weighted band / base gradient (%)',title='Training-only detached-logit probes')
    for ax in axes:
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(report/f'pilot_results.{suffix}',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
