"""Verify reproduction and summarize observed loss/SGD directions."""
import csv
from pathlib import Path
from statistics import median
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.nnunet_seed19_instrumented import ROOT, NAME, IDENTIFIER, CAP
from scripts.mni_constraints.optimizer_diagnostics import PROBE_STEPS


def summarize(rows):
    probes = [r for r in rows if r['parameter_probe']]
    return dict(updates=len(rows), clipped_updates=sum(r['clipping_factor'] < 1 for r in rows),
        preclip_norm_range=[min(r['preclip_gradient_norm'] for r in rows), max(r['preclip_gradient_norm'] for r in rows)],
        median_update_vs_negative_current_gradient_cosine=median(r['optimizer']['update_vs_negative_current_gradient_cosine'] for r in rows),
        updates_opposing_current_gradient=sum(r['optimizer']['update_vs_negative_current_gradient_cosine'] < 0 for r in rows),
        median_ce_over_dice_parameter_gradient_norm=median(r['parameter_gradients']['ce_over_dice_norm'] for r in probes),
        median_dice_ce_parameter_gradient_cosine=median(r['parameter_gradients']['dice_ce_cosine'] for r in probes),
        probes=len(probes),
        first_order_direction_counts={name: dict(
            actual_predicts_increase=sum(r['first_order_loss_changes'][name]['actual_sgd_update'] > 0 for r in probes),
            no_momentum_predicts_increase=sum(r['first_order_loss_changes'][name]['hypothetical_no_momentum_update'] > 0 for r in probes))
            for name in ('dice', 'ce', 'ca4_dice', 'ca4_ce')})


def main():
    report = ROOT / 'reports' / NAME
    d, reference = report / 'runs' / IDENTIFIER, report / 'frozen/reference_run'
    for folder in (d, reference):
        verify_archive(folder, read(folder / 'archive_manifest.json'))
        assert read(folder / 'cleanup.json')['status'] == 'complete'
    cfg, prior = read(d / 'config.json'), read(reference / 'config.json')
    changes = sorted(k for k in cfg.keys() | prior.keys() if cfg.get(k) != prior.get(k))
    assert set(changes) <= {'script_sha256', 'protocol_sha256'}, changes
    done, ledger = read(d / 'completion.json'), read(d / 'budget_ledger.json')
    assert done['updates'] == ledger['charged_attempts'] == 150
    assert done['discarded_attempts'] == 0 and done['consumed_total_flops'] == CAP
    assert digest(d / 'config.json') == done['config_sha256'] == ledger['config_sha256']
    assert digest(d / 'budget_ledger.json') == done['budget_ledger_sha256']
    diagnostic = read(d / 'optimizer_diagnostics.json')
    rows = diagnostic['rows']
    assert [r['step'] for r in rows] == list(range(1, 151))
    assert [r['step'] for r in rows if r['parameter_probe']] == list(PROBE_STEPS)
    assert diagnostic['optimizer_updates_added'] == 0
    for file, count in [('training_final.json', 20), ('validation_latest.json', 10)]:
        ev = read(d / file)
        assert len(ev['cases']) == count
        assert all('confusion' in c and 'foreground_fn' in c and 'foreground_fp' in c for c in ev['cases'])
    with (d / 'metrics.csv').open() as stream:
        curve = list(csv.DictReader(stream))
    with (reference / 'metrics.csv').open() as stream:
        oldcurve = list(csv.DictReader(stream))
    assert [r['step'] for r in curve] == [r['step'] for r in oldcurve]
    max_curve_difference = max(abs(float(a[k]) - float(b[k])) for a, b in zip(curve, oldcurve) for k in a)
    spatial_hashes = {suffix: read(d / f'spatial_{suffix}.json')['model_sha256'] ==
                      read(reference / f'spatial_{suffix}.json')['model_sha256']
                      for suffix in ('025', '050', '075', '100')}
    probe_table = []
    for r in rows:
        if not r['parameter_probe']:
            continue
        g, direction = r['parameter_gradients'], r['first_order_loss_changes']
        probe_table.append(dict(step=r['step'], dice_loss=r['dice_loss'], ce_loss=r['ce_loss'],
            ce_over_dice_gradient_norm=g['ce_over_dice_norm'], dice_ce_cosine=g['dice_ce_cosine'],
            ca4_dice_vs_ce_cosine=g['ca4_dice_ce_cosine'],
            update_vs_negative_gradient_cosine=r['optimizer']['update_vs_negative_current_gradient_cosine'],
            first_order_total_actual=direction['dice']['actual_sgd_update'] + direction['ce']['actual_sgd_update'],
            first_order_total_no_momentum=direction['dice']['hypothetical_no_momentum_update'] + direction['ce']['hypothetical_no_momentum_update'],
            **{f'{name}_{mode}': value for name, values in direction.items() for mode, value in values.items()}))
    with (report / 'gradient_probes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(probe_table[0]))
        writer.writeheader()
        writer.writerows(probe_table)
    result = dict(status='complete', archives_verified=True, config_changes=changes,
        exact_metric_reproduction=max_curve_difference == 0, max_absolute_metric_difference=max_curve_difference,
        matching_model_hashes=spatial_hashes,
        validation_dice_pct=100 * read(d / 'validation_latest.json')['summary']['macro_hard_dice'],
        training_dice_pct=100 * read(d / 'training_final.json')['summary']['macro_hard_dice'],
        optimization_flops=CAP, extra_diagnostic_profiled_flops=diagnostic['total_diagnostic_profiled_flops'],
        diagnostic_profiled_flops_fraction_of_optimization=diagnostic['total_diagnostic_profiled_flops'] / CAP,
        extra_parameter_backward_calls=sum(r['extra_parameter_gradient_calls'] for r in rows),
        extra_logit_backward_calls=sum(r['extra_logit_gradient_calls'] for r in rows),
        diagnostic_seconds=diagnostic['total_diagnostic_seconds'],
        training_invocation_seconds=done['seconds_this_invocation'],
        phases={name: summarize([r for r in rows if low <= r['step'] <= high])
                for name, low, high in [('all', 1, 150), ('early', 1, 32), ('middle', 33, 100), ('late', 101, 150)]},
        released_gib=read(d / 'cleanup.json')['released_file_bytes'] / 2**30,
        limitations='One observed trajectory. Component gradients and same-batch first-order estimates do not establish outcomes of reweighting or a different optimizer. Diagnostic FLOPs cover supported operators, not all logging/reductions.',
        analysis_source_sha256=digest(Path(__file__)))
    write(report / 'diagnostic_summary.json', result)
    plot(report, rows, curve, oldcurve)
    print(__import__('json').dumps(result, indent=2))


def plot(report, rows, curve, oldcurve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    x = [int(r['step']) for r in curve]
    ax = axes[0, 0]
    ax.plot(x, [100 * float(r['macro_hard_dice']) for r in curve], label='Validation macro Dice', color='#1768ac')
    ax.plot(x, [100 * float(r['CA4_DG_hard_dice']) for r in curve], label='Validation CA4/DG', color='#c04035')
    ax.scatter(x, [100 * float(r['macro_hard_dice']) for r in oldcurve], s=30, facecolors='none', edgecolors='#1768ac', label='Original run')
    ax.set(title='Reproduction of the original training run', ylabel='Dice (%)')
    ax.legend(fontsize=8)
    probes = [r for r in rows if r['parameter_probe']]
    steps = [r['step'] for r in probes]
    ax = axes[0, 1]
    for component, label in [('dice', 'Dice'), ('ce', 'Cross-entropy')]:
        ax.semilogy(steps, [r['parameter_gradients']['norms'][component] for r in probes], 'o-', label=label)
    ax.set(title='Parameter gradient magnitude', ylabel='L2 norm')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(steps, [r['parameter_gradients']['dice_ce_cosine'] for r in probes], 'o-', label='Dice vs CE gradients')
    ax.plot(steps, [r['parameter_gradients']['ca4_dice_ce_cosine'] for r in probes], 'o-', label='CA4/DG Dice vs total CE')
    ax.axhline(0, color='gray', lw=.8)
    ax.set(title='Do the loss terms agree?', ylabel='Cosine (negative = conflict)', ylim=(-1.05, 1.05))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot([r['step'] for r in rows], [r['optimizer']['update_vs_negative_current_gradient_cosine'] for r in rows], color='#1768ac')
    ax.axhline(0, color='gray', lw=.8)
    ax.set(title='Actual SGD update vs current descent direction', ylabel='Cosine (negative = opposition)', ylim=(-1.05, 1.05))
    for ax in axes.flat:
        ax.set_xlabel('Optimizer update')
        ax.grid(alpha=.15)
    fig.suptitle('nnUNet · seed 19 · fold 0 · unchanged 150-update recipe', fontsize=14)
    fig.savefig(report / 'optimizer_loss_diagnostics.png', dpi=160)
    fig.savefig(report / 'optimizer_loss_diagnostics.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
