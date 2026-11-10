"""Report all fixed-recipe MNI cells, including the four archived pilot cells."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive


def run_row(run, model, fold, seed, cap, reused=False):
    verify_archive(run, read(run/'archive_manifest.json'))
    config = read(run/'config.json')
    done = read(run/'completion.json')
    assert config['budget_flops'] == cap
    assert done['consumed_training_flops'] <= cap
    training = read(run/'training_final.json')
    validation = read(run/'validation_latest.json')
    score = training['summary']['macro_hard_dice']
    validation_score=validation['summary']['macro_hard_dice']
    with (run/'metrics.csv').open() as stream:
        curve=list(csv.DictReader(stream))
    assert score == done['final_training_hard_dice']
    return dict(model=model, fold=fold, seed=seed, reused_pilot=reused,
                training_dice=score,
                validation_dice=validation_score,
                validation_above_floor=validation_score>.5,
                training_validation_gap_pp=100*(score-validation_score),
                peak_to_final_validation_drop_pp=100*(max(float(r['macro_hard_dice']) for r in curve)-validation_score),
                minimum_training_subfield_dice=min(
                    c['dice'] for case in training['cases']
                    for name, c in case['per_class'].items() if name != 'background'),
                updates=done['updates'], optimization_flops=done['consumed_training_flops'],
                archive=str(run), archive_manifest_sha256=digest(run/'archive_manifest.json'),
                cleaned=(run/'cleanup.json').exists() and read(run/'cleanup.json')['status']=='complete')


def summarize(stage, report):
    plan = read(stage/'plan.json')
    rows = []
    for item in read(stage/'baseline_reuse.json'):
        run = Path(item['archive'])
        assert digest(run/'archive_manifest.json') == item['archive_manifest_sha256']
        rows.append(run_row(run, item['model'], 0, 0, plan['budget_flops'], reused=True))
    for task in plan['tasks']:
        run = report/'runs'/task['id']
        if not (run/'archive_manifest.json').exists():
            continue
        rows.append(run_row(run, task['model'], task['fold'], task['seed'], plan['budget_flops']))
    assert len({(r['model'],r['fold'],r['seed']) for r in rows}) == len(rows)
    aggregate = {}
    for model in plan['models']:
        cells = [r for r in rows if r['model'] == model]
        training = [r['training_dice'] for r in cells]
        validation = [r['validation_dice'] for r in cells]
        aggregate[model] = dict(
            completed=len(cells), expected=9, training_mean=mean(training),
            training_range=[min(training), max(training)],
            validation_mean=mean(validation),
            validation_sd=stdev(validation) if len(validation)>1 else None,
            cells_above_validation_floor=sum(r['validation_above_floor'] for r in cells),
            training_validation_gap_mean_pp=100*(mean(training)-mean(validation)),
            peak_to_final_validation_drop_mean_pp=mean(r['peak_to_final_validation_drop_pp'] for r in cells),
            all_cleaned=all(r['cleaned'] for r in cells))
    result = dict(status='complete' if len(rows)==36 else 'partial', cap=plan['budget_flops'],
                  completed=len(rows), expected=36, validation_floor=.5, training_policy='flexible_headroom_no_rigid_dice_band',models=aggregate, runs=rows,
                  caveat='Fixed recipes selected on fold 0 seed 0 using training headroom, a validation floor and gap diagnostics. Validation is development data. '
                         'Folds can overlap; SD is descriptive, not an independent-sample confidence interval.')
    write(report/'training_headroom.json', result)
    lines = ['# MNI shared-cap headroom confirmation', '',
             f"Completed: {len(rows)}/36. Shared optimization FLOP cap: {plan['budget_flops']:,}.", '',
             '| Model | Cells | Training mean | Training range | Validation >50% | Validation mean | Train/val gap |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for model, a in aggregate.items():
        lo, hi = a['training_range']
        lines.append(f"| {model} | {a['completed']}/9 | {100*a['training_mean']:.2f}% | "
                     f"{100*lo:.2f}–{100*hi:.2f}% | {a['cells_above_validation_floor']}/{a['completed']} | "
                     f"{100*a['validation_mean']:.2f}% | {a['training_validation_gap_mean_pp']:.2f} pp |")
    lines += ['', result['caveat'], '']
    (report/'training_headroom.md').write_text('\n'.join(lines))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='mni_shared_cap3_headroom_3fold3seed_20261107')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    result=summarize(root/'experiments'/args.name, root/'reports'/args.name)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__':
    main()
