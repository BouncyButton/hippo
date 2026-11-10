"""Compare reduced-LR controls with the identical plain LR0.0006 recipe."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unetr_lower_lr_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    report = study.ROOT / 'reports' / study.NAME
    prior = study.ROOT / 'reports/mni_unetr_plain_cosine53t_20261109'
    result_bytes = (report / 'lower_lr_results.json').read_bytes()
    result = json.loads(result_bytes)
    rows, missing = [], []
    for item in result['archived_runs']:
        if item['role'] != 'baseline':
            continue
        seed, fold = item['seed'], item['fold']
        old = prior / 'runs' / f'MNI_unetr_plain_cosine_dice_f{fold}s{seed}'
        if not (old / 'cleanup.json').exists():
            missing.append(dict(recipe=item['recipe'], seed=seed, fold=fold))
            continue
        current = report / 'runs' / item['id']
        cfg, old_cfg = read(current / 'config.json'), read(old / 'config.json')
        for key in ('seed', 'fold', 'objective', 'train_cases', 'validation_cases',
                    'lr_horizon_updates', 'schedule', 'dataset_sha256', 'script_sha256',
                    'objective_source_sha256', 'augmentation', 'torch', 'monai'):
            assert cfg[key] == old_cfg[key], (item['id'], key)
        assert cfg['initial_lr'] == item['peak_lr'] and old_cfg['initial_lr'] == .0006
        new_protocol = report / 'frozen/protocols' / item['protocol'] / 'protocol.json'
        old_protocol = prior / 'frozen/protocols/plain_cosine_dice/protocol.json'
        assert cfg['protocol_sha256'] == digest(new_protocol)
        assert old_cfg['protocol_sha256'] == digest(old_protocol)
        p, q = read(new_protocol), read(old_protocol)
        for key in ('lr_policy', 'lr_horizon_updates', 'lr_schedule_details',
                    'adamw_weight_decay', 'dropout_rate', 'augmentation',
                    'loss_arguments', 'loss_weights', 'loss_source_sha256'):
            assert p[key] == q[key], (item['id'], key)
        model, old_model = read(current / 'model_recipe.json'), read(old / 'model_recipe.json')
        for key in ('model', 'declared', 'kwargs', 'actual_dropout_modules', 'parameters'):
            assert model[key] == old_model[key], key
        new_score, old_score = evaluation(current), evaluation(old)
        rows.append(dict(
            recipe=item['recipe'], seed=seed, fold=fold, peak_lr=item['peak_lr'],
            baseline_dice=new_score['validation_percent'],
            previous_baseline_dice=old_score['validation_percent'],
            baseline_subfields=new_score['per_subfield_percent'],
            previous_baseline_subfields=old_score['per_subfield_percent'],
            validation_change_pp=new_score['validation_percent']-old_score['validation_percent'],
            training_change_pp=new_score['training_percent']-old_score['training_percent'],
            subfield_change_pp={key: value-old_score['per_subfield_percent'][key]
                               for key, value in new_score['per_subfield_percent'].items()},
            current_archive_sha256=new_score['archive_manifest_sha256'],
            previous_archive_sha256=old_score['archive_manifest_sha256'],
        ))
    summary = []
    for recipe in sorted({row['recipe'] for row in rows}):
        group = [row for row in rows if row['recipe'] == recipe]
        summary.append(dict(recipe=recipe, pairs=len(group),
                            mean_validation_change_pp=mean(row['validation_change_pp'] for row in group),
                            improved_pairs=sum(row['validation_change_pp'] > 0 for row in group)))
    write(report / 'peak_lr_baseline_comparison.json', dict(
        rows=rows, summary=summary, missing_previous_cells=missing,
        input_result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        analysis_source_sha256=digest(Path(__file__)),
        scope='Matched same-fold/seed/data, plain DiceCE1 AdamW cosine controls; only peak LR changes.',
        caveat='Available development comparisons with overlapping folds. Missing controls are not imputed; this measures baseline effects, not constraint effects.',
    ))
    print(json.dumps(summary))
    if args.plot:
        plot(report, rows)


def plot(report, rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    for seed, fold in study.PAIRS:
        group = [row for row in rows if (row['seed'], row['fold']) == (seed, fold)]
        if not group:
            continue
        values = {row['peak_lr']: (row['baseline_dice'],
                  row['baseline_subfields']['CA4_DG_hard_dice']) for row in group}
        previous = {(row['previous_baseline_dice'],
                     row['previous_baseline_subfields']['CA4_DG_hard_dice']) for row in group}
        assert len(previous) == 1
        values[.0006] = previous.pop()
        rates = sorted(values)
        for index, ax in enumerate(axes):
            ax.plot(rates, [values[rate][index] for rate in rates], marker='o',
                    label=f'Seed {seed}, fold {fold}', linewidth=1.8, markersize=5)
    for ax, title in zip(axes, ['Validation macro Dice', 'Validation CA4/DG Dice']):
        ax.set_title(title)
        ax.set_xlabel('Peak learning rate')
        ax.set_ylabel('Dice (%)')
        ax.set_xticks([.0003, .00045, .0006], ['0.00030', '0.00045', '0.00060'])
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)), frameon=False)
    fig.suptitle('UNETR baseline controls across peak learning rates\n'
                 'Same plain Dice + CE1 recipe, 409 updates, 53.305T optimization-FLOPs cap',
                 fontsize=12)
    legend_rows = (len(labels) + 3) // 4
    fig.tight_layout(rect=(0, .05 + .05 * legend_rows, 1, .88))
    for suffix in ('png', 'pdf'):
        fig.savefig(report / f'peak_lr_baselines.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    comparison = report / 'peak_lr_baseline_comparison.json'
    write(report / 'peak_lr_baselines_plot.json', dict(
        rows=rows, comparison_sha256=digest(comparison),
        input_result_sha256=read(comparison)['input_result_sha256'],
        analysis_source_sha256=digest(Path(__file__)),
        files={suffix: digest(report / f'peak_lr_baselines.{suffix}')
               for suffix in ('png', 'pdf')},
        scope='Baseline controls only. Lines connect observed rates; missing cells are not imputed.',
    ))


if __name__ == '__main__':
    main()
