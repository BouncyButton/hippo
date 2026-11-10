"""Audit completed repair runs and the fixed nine-cell55% acceptance rule."""
import argparse
import csv
from pathlib import Path
from statistics import mean
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.nnunet_repair import ROOT, PILOT, CONFIRM, CAP, RECIPES


def verify_run(report, task):
    d = report / 'runs' / task['id']
    verify_archive(d, read(d / 'archive_manifest.json'))
    assert read(d / 'cleanup.json')['status'] == 'complete'
    config, done, ledger = [read(d / (name + '.json')) for name in ('config', 'completion', 'budget_ledger')]
    recipe = RECIPES[task['recipe']]
    assert done['status'] == 'complete' and done['discarded_attempts'] == 0
    assert config['max_steps'] == done['updates'] == ledger['charged_attempts'] == 150
    assert done['consumed_total_flops'] == config['budget_flops'] == CAP
    assert config['initial_lr'] == recipe['lr'] and config['nnunet_ce_weight'] == recipe['ce']
    assert config['lr_horizon_updates'] == 200 and config['nnunet_sgd_momentum'] == .95
    assert config['augmentation'] == 'none' and config['fold'] == task['fold'] and config['seed'] == task['seed']
    assert digest(d / 'config.json') == done['config_sha256'] == ledger['config_sha256']
    assert digest(d / 'budget_ledger.json') == done['budget_ledger_sha256']
    frozen = report / 'frozen'
    assert config['script_sha256'] == digest(frozen / 'source/scripts/cross_dataset_campaign/capped_core.py')
    assert config['dataset_sha256'] == digest(frozen / 'data/MNI.pkl')
    protocol = frozen / 'protocols' / task['protocol']
    assert config['protocol_sha256'] == digest(protocol / 'protocol.json')
    split = read(protocol / 'splits_10patients.json')[task['fold']]
    assert config['train_cases'] == split['train'] and config['validation_cases'] == split['val']
    prior = read(protocol / 'parent_protocol.json')
    assert config['schedule'] == prior['models']['nnunet']['objectives']['dice']
    assert read(d / 'evaluation_diagnostics.json')['status'] == 'complete'
    for filename, count in [('training_final.json', 20), ('validation_latest.json', 10)]:
        evaluation = read(d / filename)
        assert len(evaluation['cases']) == count
        assert all('confusion' in c and 'foreground_fp' in c and 'foreground_fn' in c for c in evaluation['cases'])
    with (d / 'metrics.csv').open() as stream:
        curve = list(csv.DictReader(stream))
    final = curve[-1]
    assert int(final['step']) == 150
    val = float(final['macro_hard_dice'])
    assert val == read(d / 'validation_latest.json')['summary']['macro_hard_dice']
    return dict(id=task['id'], recipe=task['recipe'], fold=task['fold'], seed=task['seed'],
        validation_pct=100 * val, training_pct=100 * float(final['training_macro_hard_dice']),
        class_validation_pct={c: 100 * float(final[c + '_hard_dice']) for c in ('CA1_3', 'subiculum', 'CA4_DG')},
        first_logged_ge55_update=next((int(r['step']) for r in curve if float(r['macro_hard_dice']) >= .55), None),
        passes55=val >= .55, updates=150, optimization_flops=CAP,
        released_gib=read(d / 'cleanup.json')['released_file_bytes'] / 2**30,
        archive=str(d), archive_manifest_sha256=digest(d / 'archive_manifest.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    pilot = ROOT / 'reports' / PILOT
    pilots = [verify_run(pilot, task) for task in read(pilot / 'frozen/plan.json')['tasks']]
    selected = next((r['recipe'] for r in pilots if r['recipe'] == 'lr002' and r['passes55']), None)
    if selected is None:
        selected = next((r['recipe'] for r in pilots if r['passes55']), None)
    result = dict(status='complete', pilots=pilots, selected_recipe=selected,
        selection_rule='Prefer LR-only if it clears55%; otherwise choose reduced CE if it clears55%.',
        cap=CAP, acceptance_threshold_pct=55., analysis_source_sha256=digest(Path(__file__)))
    write(pilot / 'pilot_results.json', result)
    if args.confirm:
        assert selected
        report = ROOT / 'reports' / CONFIRM
        rows = [r for r in pilots if r['recipe'] == selected]
        rows += [verify_run(report, t) for t in read(report / 'frozen/plan.json')['tasks']]
        assert len(rows) == 9 and {(r['fold'], r['seed']) for r in rows} == {(f, s) for f in range(3) for s in (0, 5, 19)}
        assert {r['recipe'] for r in rows} == {selected}
        rows.sort(key=lambda r: (r['seed'], r['fold']))
        result.update(rows=rows, shared_recipe=RECIPES[selected],
            all_nine_pass55=all(r['passes55'] for r in rows),
            mean_validation_pct=mean(r['validation_pct'] for r in rows),
            minimum_validation_pct=min(r['validation_pct'] for r in rows),
            by_seed=[dict(seed=s, folds_pct=[r['validation_pct'] for r in rows if r['seed'] == s],
                mean_pct=mean(r['validation_pct'] for r in rows if r['seed'] == s)) for s in (0, 5, 19)],
            limitation='Same development-validation folds used for recipe tuning. These are final-budget results across fixed seeds, not an independent test estimate.')
        # Count every new experiment exactly once, including the unsuccessful pilot.
        result['total_released_gib'] = sum(r['released_gib'] for r in {r['id']: r for r in pilots + rows}.values())
        write(report / 'confirmation_results.json', result)
        if result['all_nine_pass55']:
            protocol_path = report / 'frozen/protocols' / selected / 'protocol.json'
            write(report / 'winning_recipe.json', dict(model='nnunet', status='verified_all_nine_ge55',
                initial_lr=RECIPES[selected]['lr'], lr_policy='hold_half', lr_horizon_updates=200,
                optimizer='SGD', momentum=.95, nesterov=True, weight_decay=3e-5,
                loss='Original native foreground Dice+CE with deep supervision',
                dice_weight=1., ce_weight=RECIPES[selected]['ce'], augmentation='none',
                gradient_clip_norm=12., training_subjects=10, training_crops=20,
                seeds=[0, 5, 19], folds=[0, 1, 2], updates=150, optimization_flops_cap=CAP,
                protocol=str(protocol_path), protocol_sha256=digest(protocol_path),
                core_source_sha256=digest(report / 'frozen/source/scripts/cross_dataset_campaign/capped_core.py'),
                validation_results=str(report / 'confirmation_results.json'),
                minimum_validation_pct=result['minimum_validation_pct'], mean_validation_pct=result['mean_validation_pct'],
                adoption='Use this LR0.02 recipe for future nnUNet runs at the53.305T cap. Historical frozen campaigns remain unchanged and paused.'))
        with (report / 'all_seeds_folds.csv').open('w', newline='') as stream:
            keys = ('recipe', 'seed', 'fold', 'validation_pct', 'training_pct', 'passes55', 'updates', 'optimization_flops')
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: r[k] for k in keys} for r in rows)
        plot_results(report, pilot, result)
    print(__import__('json').dumps(result, indent=2))


def plot_results(report, pilot, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(12, 8), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .65])
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    original = ROOT / 'reports/mni_common0519_shared53t_3fold_20261108/runs/MNI_nnunet_dice_f0s19'
    sources = [(original, 'Original: LR 0.01, CE 1.0', '#777777'),
        (pilot / 'runs/MNI_nnunet_lr002_f0s19', 'LR 0.02, CE 1.0', '#1768ac'),
        (pilot / 'runs/MNI_nnunet_ce01_f0s19', 'LR 0.01, CE 0.1', '#bf6530')]
    for source, label, color in sources:
        with (source / 'metrics.csv').open() as stream:
            curve = list(csv.DictReader(stream))
        x = [int(r['step']) for r in curve]
        for ax, key in zip(axes, ('macro_hard_dice', 'CA4_DG_hard_dice')):
            ax.plot(x, [100 * float(r[key]) for r in curve], label=label, color=color, linewidth=2)
    axes[0].axhline(55, color='#333333', linestyle=':', label='55% target')
    for ax, title in zip(axes, ('Validation macro Dice · seed 19, fold 0', 'CA4/DG Dice · seed 19, fold 0')):
        ax.set(title=title, xlabel='Optimizer update', ylabel='Dice (%)', ylim=(-2, 90))
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8, loc='upper left')
    ax = fig.add_subplot(grid[1, :])
    values = np.array([[next(r['validation_pct'] for r in result['rows'] if r['seed'] == s and r['fold'] == f)
                        for f in range(3)] for s in (0, 5, 19)])
    ax.imshow(values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for y in range(3):
        for x in range(3):
            ax.text(x, y, f'{values[y, x]:.2f}%', ha='center', va='center', color='white' if values[y, x] > 55 else '#111111', fontsize=14)
    ax.set(xticks=range(3), xticklabels=['Fold 0', 'Fold 1', 'Fold 2'],
           yticks=range(3), yticklabels=['Seed 0', 'Seed 5', 'Seed 19'],
           title='Final-budget validation Dice · one fixed LR 0.02 recipe')
    fig.suptitle('nnUNet repair at the unchanged 53.305 TFLOPs cap / 150 updates', fontsize=15)
    fig.savefig(report / 'repair_results.png', dpi=160)
    fig.savefig(report / 'repair_results.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
