"""Audit and summarize the completed 504-cell MNI winning-recipe matrix."""
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive

NAME = 'mni_winning_recipes_matrix2030_53t_0519_20261109'
LABELS = {'basic_unet16': 'UNet', 'nnunet': 'nnUNet',
          'unetr_vit_b16': 'UNETR', 'swinunetr': 'Swin UNETR'}
MODELS = ['basic_unet16', 'nnunet', 'unetr_vit_b16', 'swinunetr']


def main():
    root = Path(__file__).resolve().parents[2]
    report = root/'reports'/NAME
    frozen = report/'frozen'
    plan = read(frozen/'plan.json')
    reused = read(frozen/'reused_runs.json')
    tasks = [(t, report/'runs'/t['id'], False) for t in plan['tasks']]
    tasks += [(r['task'], frozen/'reused_runs'/r['task']['id'], True) for r in reused]
    rows = []
    for task, folder, reuse in tasks:
        am = read(folder/'archive_manifest.json')
        verify_archive(folder, am)
        done, config = read(folder/'completion.json'), read(folder/'config.json')
        assert done['status'] == 'complete' and done['discarded_attempts'] == 0
        assert config['budget_flops'] == plan['budget_flops']
        assert done['consumed_training_flops'] <= plan['budget_flops']
        assert config['fold'] == task['fold'] and config['seed'] == task['seed']
        assert config['objective'] == task['arm']
        if not reuse:
            ack = read(folder/'archive_acknowledgment.json')
            assert ack['status'] == 'verified' and ack['manifest_sha256'] == digest(folder/'archive_manifest.json')
            assert read(folder/'cleanup.json')['status'] == 'complete'
        result = read(folder/'validation_latest.json')
        summary = result['summary']
        assert len(result['cases']) == len(config['validation_cases'])
        rows.append(dict(**task, model_label=LABELS[task['model']], reused=reuse,
            final_dice=summary['macro_hard_dice'],
            ca1_3_dice=summary['CA1_3_hard_dice'],
            subiculum_dice=summary['subiculum_hard_dice'], ca4_dg_dice=summary['CA4_DG_hard_dice'],
            whole_hippocampus_dice=summary['whole_hippocampus_hard_dice'],
            training_dice=done['final_training_hard_dice'],
            best_checkpoint_dice=done['best_validation_hard_dice'],
            updates=done['updates'], optimization_flops=done['consumed_training_flops'],
            calibration_flops=done['calibration_flops'], validation_cases=len(result['cases']),
            archive=str(folder), validation_sha256=digest(folder/'validation_latest.json')))
    arms = plan['conditions']
    assert len(arms) == 14 and len(rows) == 504
    expected = set(itertools.product(MODELS, arms, [0,5,19], [0,1,2]))
    assert len({(r['model'],r['arm'],r['seed'],r['fold']) for r in rows}) == 504
    assert {(r['model'],r['arm'],r['seed'],r['fold']) for r in rows} == expected
    baselines = {(r['model'],r['seed'],r['fold']):r for r in rows if r['arm']=='dice'}
    for r in rows:
        baseline = baselines[r['model'],r['seed'],r['fold']]
        r['baseline_dice'] = baseline['final_dice']
        r['gain_pp'] = 100*(r['final_dice']-r['baseline_dice'])
    rows.sort(key=lambda r:(MODELS.index(r['model']),arms.index(r['arm']),r['seed'],r['fold']))
    aggregate = []
    for model, arm in itertools.product(MODELS, arms):
        cells = [r for r in rows if r['model']==model and r['arm']==arm]
        scores = [r['final_dice'] for r in cells]
        gains = [r['gain_pp'] for r in cells]
        aggregate.append(dict(model=model,model_label=LABELS[model],arm=arm,
            mean=mean(scores), sd=stdev(scores), n=len(cells),
            seed_means={str(s):mean(r['final_dice'] for r in cells if r['seed']==s) for s in [0,5,19]},
            fold_means={str(f):mean(r['final_dice'] for r in cells if r['fold']==f) for f in [0,1,2]},
            baseline_mean=mean(r['baseline_dice'] for r in cells),gain_pp=mean(gains),
            wins=sum(g>0 for g in gains),ties=sum(g==0 for g in gains),minimum_gain_pp=min(gains),
            maximum_gain_pp=max(gains),minimum_dice=min(scores),maximum_dice=max(scores)))
    winners = []
    for model in MODELS:
        cells = [r for r in aggregate if r['model']==model]
        ordered = sorted(cells,key=lambda r:r['mean'],reverse=True)
        for rank, row in enumerate(ordered,1):row['rank']=rank
        consistent = [r for r in ordered if r['arm']!='dice' and r['wins']==9]
        winners.append(dict(model=model,model_label=LABELS[model],best_mean=ordered[0],
            best_all_nine=consistent[0] if consistent else None,
            baseline=next(r for r in cells if r['arm']=='dice')))
    result = dict(status='complete',total=504,new_runs=407,reused_runs=97,
        metric='Final-budget validation macro hard Dice; unweighted mean of the three foreground subfields',
        averaging='Each seed column averages folds 0,1,2. Overall averages all nine seed/fold cells equally.',
        cap=plan['budget_flops'],budget_scope='optimization_only; calibration FLOPs tracked separately',
        caveat='Validation data were used for recipe selection. Folds can overlap; descriptive SD is not an independent-sample confidence interval.',
        models=MODELS,labels=LABELS,arms=arms,rows=rows,aggregate=aggregate,winners=winners)
    write(report/'final_results.json',result)
    for model in MODELS:
        print('\n'+LABELS[model])
        print('| Constraints | Seed 0 | Seed 5 | Seed 19 | Mean | Gain pp | Wins |')
        print('|---|---:|---:|---:|---:|---:|---:|')
        for a in aggregate:
            if a['model']!=model:continue
            title='Baseline' if a['arm']=='dice' else a['arm'].removeprefix('dice_').replace('_',' + ')
            seed=' | '.join(f"{100*a['seed_means'][str(s)]:.2f}" for s in [0,5,19])
            print(f"| {title} | {seed} | {100*a['mean']:.2f} | {a['gain_pp']:+.2f} | {a['wins']}/9 |")
    print('\nWINNERS\n'+json.dumps(winners,indent=2))


if __name__=='__main__':main()
