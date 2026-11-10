"""Paired final-budget comparisons against the verified MNI baselines."""
import csv
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, stdev

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.campaign import ROOT, NAME, CAP


def main():
    report = ROOT/'reports'/NAME
    references = read(report/'frozen/baseline_reuse.json')
    baselines = {}
    for row in references:
        from pathlib import Path
        path = Path(row['archive'])
        assert digest(path/'archive_manifest.json') == row['archive_manifest_sha256']
        verify_archive(path, read(path/'archive_manifest.json'))
        score = read(path/'validation_latest.json')['summary']['macro_hard_dice']
        assert score == row['final_dice']
        baselines[row['model'], row['fold'], row['seed']] = (score, read(path/'config.json'))
    records, cleanup = [], []
    for path in sorted((report/'runs').iterdir()):
        if not (path/'archive_acknowledgment.json').exists():
            continue
        manifest = read(path/'archive_manifest.json')
        acknowledgment = read(path/'archive_acknowledgment.json')
        assert acknowledgment['status'] == 'verified'
        assert acknowledgment['manifest_sha256'] == digest(path/'archive_manifest.json')
        verify_archive(path, manifest)
        done = read(path/'completion.json'); config = read(path/'config.json')
        assert done['status'] == 'complete' and done['discarded_attempts'] == 0
        assert done['consumed_training_flops'] <= CAP == config['budget_flops']
        identity, fold, seed = config['model_identity'], config['fold'], config['seed']
        baseline, baseline_config = baselines[identity, fold, seed]
        for key in ('train_cases', 'validation_cases', 'dataset_sha256', 'augmentation', 'budget_flops'):
            assert config[key] == baseline_config[key], (path.name, key)
        score = read(path/'validation_latest.json')['summary']['macro_hard_dice']
        records.append(dict(model=identity, arm=config['objective'], fold=fold, seed=seed,
            final_dice=score, baseline_dice=baseline, delta_pp=100*(score-baseline),
            updates=done['updates'], run_id=path.name))
        if (path/'cleanup.json').exists():
            receipt = read(path/'cleanup.json')
            assert receipt['status'] == 'complete'
            cleanup.append(receipt['released_file_bytes'])
    groups = defaultdict(list)
    for row in records:
        groups[row['model'], row['arm']].append(row)
    comparisons = []
    for (identity, arm), rows in sorted(groups.items()):
        pairs = {(r['fold'], r['seed']) for r in rows}
        assert len(pairs) == len(rows)
        full = pairs == {(f,s) for f in range(3) for s in range(3)}
        by_fold = [dict(fold=f, n=len(group), delta_pp=mean(r['delta_pp'] for r in group))
                   for f in range(3) if (group := [r for r in rows if r['fold']==f])]
        by_seed = [dict(seed=s, n=len(group), delta_pp=mean(r['delta_pp'] for r in group))
                   for s in range(3) if (group := [r for r in rows if r['seed']==s])]
        comparisons.append(dict(model=identity, arm=arm, n=len(rows), complete_3fold_3seed=full,
            baseline_mean=mean(r['baseline_dice'] for r in rows),
            constrained_mean=mean(r['final_dice'] for r in rows),
            mean_delta_pp=mean(r['delta_pp'] for r in rows),
            sd_delta_pp=stdev(r['delta_pp'] for r in rows) if len(rows)>1 else None,
            min_delta_pp=min(r['delta_pp'] for r in rows), max_delta_pp=max(r['delta_pp'] for r in rows),
            wins=sum(r['delta_pp']>0 for r in rows), losses=sum(r['delta_pp']<0 for r in rows),
            by_fold=by_fold, by_seed=by_seed))
    result = dict(as_of_utc=datetime.now(timezone.utc).isoformat(), archived_runs=len(records),
        planned_new_runs=468, verified_cleanup_runs=len(cleanup), released_bytes=sum(cleanup),
        metric='Final-budget mean per-volume foreground-subfield macro hard Dice; paired same model/fold/seed baseline. Deltas in percentage points.',
        caveat='Development validation, not an untouched test set. Repeated seeds within folds are not independent subjects; no significance claim from nine cells.',
        comparisons=comparisons)
    write(report/'paired_comparison.json',result)
    with (report/'paired_run_deltas.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(__import__('json').dumps(result,indent=2))


if __name__ == '__main__':main()
