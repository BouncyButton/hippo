"""Verify and summarize the paired43T/53T common-seed baseline matrix."""
import argparse
from pathlib import Path
from statistics import mean
from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive


def finish(report):
    plan = read(report / 'frozen/plan.json')
    rows = []
    released = 0
    for task in plan['tasks']:
        d = report / 'runs' / task['id']
        old = report / 'frozen/controls' / task['id']
        for folder in (d, old):
            verify_archive(folder, read(folder / 'archive_manifest.json'))
            assert read(folder / 'cleanup.json')['status'] == 'complete'
        cfg, prior = read(d / 'config.json'), read(old / 'config.json')
        changes = sorted(k for k in set(cfg) | set(prior) if cfg.get(k) != prior.get(k))
        allowed = {'schedule', 'max_steps', 'budget_flops', 'protocol_sha256', 'selection'}
        if task['model'] != 'nnunet':
            allowed.add('lr_horizon_updates')
        assert set(changes) <= allowed, (task['id'], changes)
        done, ledger = read(d / 'completion.json'), read(d / 'budget_ledger.json')
        assert done['status'] == 'complete' and done['discarded_attempts'] == 0
        assert done['updates'] == cfg['max_steps'] == ledger['charged_attempts']
        assert done['consumed_total_flops'] <= cfg['budget_flops'] == plan['budget_flops']
        assert cfg['budget_flops'] - done['consumed_total_flops'] < cfg['schedule']['flops_per_update']
        assert digest(d / 'config.json') == done['config_sha256'] == ledger['config_sha256']
        assert digest(d / 'budget_ledger.json') == done['budget_ledger_sha256']
        assert read(d / 'evaluation_diagnostics.json')['status'] == 'complete'
        for filename, count, key in [('training_final.json', 20, 'train_cases'), ('validation_latest.json', 10, 'validation_cases')]:
            ev = read(d / filename)
            assert len(ev['cases']) == count and {c['case_name'] for c in ev['cases']} == set(cfg[key])
            assert all('confusion' in c and 'foreground_fp' in c and 'foreground_fn' in c for c in ev['cases'])
        val = read(d / 'validation_latest.json')['summary']['macro_hard_dice']
        oldval = read(old / 'validation_latest.json')['summary']['macro_hard_dice']
        released += read(d / 'cleanup.json')['released_file_bytes']
        rows.append(dict(id=task['id'], model=task['model'], seed=task['seed'], fold=task['fold'],
            validation_pct=100 * val, previous_validation_pct=100 * oldval, gain_pp=100 * (val - oldval),
            training_pct=100 * read(d / 'training_final.json')['summary']['macro_hard_dice'],
            updates=done['updates'], consumed_flops=done['consumed_total_flops'], config_changes=changes,
            archive=str(d.resolve()), archive_manifest_sha256=digest(d / 'archive_manifest.json')))
    summary = []
    by_seed = []
    for model in plan['models']:
        group = [r for r in rows if r['model'] == model]
        summary.append(dict(model=model, mean_validation_pct=mean(r['validation_pct'] for r in group),
            previous_mean_validation_pct=mean(r['previous_validation_pct'] for r in group),
            mean_gain_pp=mean(r['gain_pp'] for r in group), improved_cells=sum(r['gain_pp'] > 0 for r in group),
            cells_ge50=sum(r['validation_pct'] >= 50 for r in group), minimum_validation_pct=min(r['validation_pct'] for r in group)))
        for seed in plan['seeds']:
            cells = sorted((r for r in group if r['seed'] == seed), key=lambda r: r['fold'])
            assert len(cells) == 3
            by_seed.append(dict(model=model, seed=seed, folds_pct=[r['validation_pct'] for r in cells],
                mean_validation_pct=mean(r['validation_pct'] for r in cells), mean_gain_pp=mean(r['gain_pp'] for r in cells),
                passes_every_fold50=all(r['validation_pct'] >= 50 for r in cells)))
    result = dict(status='complete', all_archives_verified=True, rows=rows, summary=summary, by_seed=by_seed,
                  budget_flops=plan['budget_flops'], previous_budget_flops=plan['previous_budget_flops'],
                  new_runs_released_gib=released / 2**30, caveat=plan['selection_caveat'],
                  analysis_source_sha256=digest(Path(__file__)))
    write(report / 'cap_comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('stage', 'local-stage', 'destination'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    collect.serve(args.stage, args.local_stage, args.destination)
    if read(args.destination / 'live.json').get('status.json', {}).get('status') == 'complete':
        finish(args.destination)


if __name__ == '__main__':
    main()
