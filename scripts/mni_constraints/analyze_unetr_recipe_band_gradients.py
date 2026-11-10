"""Compare the saved original-band logit probes across base recipes."""
import json
from pathlib import Path
from statistics import median

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unetr_simple_recipe_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def trajectory(folder):
    evaluation(folder)
    rows=[]
    for quarter in (25,50,75,100):
        path=folder/f'spatial_{quarter:03d}.json'
        snapshot=read(path)
        assert snapshot['optimizer_updates']==0
        training=[c for c in snapshot['cases'] if c['split']=='training']
        assert len(training)==20
        values=[c['gradient_probe']['original_bands']['weighted_norm_ratio'] for c in training]
        assert all(v is not None and v>=0 for v in values)
        ramp=min(1.,max(0.,(snapshot['charged_updates']-102)/33))
        rows.append(dict(quarter=quarter,updates=snapshot['charged_updates'],
            median_band_over_base=median(values),maximum_band_over_base=max(values),
            ramp=ramp,median_ramp_scaled_band_over_base=ramp*median(values),
            source_sha256=digest(path)))
    return rows


def main():
    report=study.ROOT/'reports'/study.NAME
    result=read(report/'simple_recipe_results.json')
    old={(r['arm'],r['seed'],r['fold']):r for r in read(report/'frozen/old_constraints.json')}
    references={}
    rows=[]
    for run in result['rows']:
        key=run['arm'],run['seed'],run['fold']
        reference=old.get(key)
        if reference is None:
            continue
        if key not in references:
            references[key]=trajectory(report/'frozen/old_constraints'/reference['name'])
        rows.append(dict(candidate=run['candidate'],arm=run['arm'],seed=run['seed'],fold=run['fold'],
            gain_pp=run['gain_pp'],old_recipe=references[key],new_recipe=trajectory(report/'runs'/run['id'])))
    output=dict(rows=rows,
        scope='Only the original-band component: detached-logit gradient norms on20 unaugmented eval-mode training crops. Saved probes, zero additional optimizer updates. Ratios compare the weighted band gradient with the base loss gradient; full-strength and ramp-scaled values are separate.',
        limitations='Not combined auxiliary gradients, parameter gradients, actual Adam updates, or proof of why validation changed. Hypothetical weighted/adaptive band probes present in the raw files are excluded; they were not trained.',
        input_result_sha256=digest(report/'simple_recipe_results.json'),analysis_source_sha256=digest(Path(__file__)))
    write(report/'band_gradient_recipe_comparison.json',output)
    print(json.dumps([dict(candidate=r['candidate'],seed=r['seed'],fold=r['fold'],
        old_median_percent=[100*q['median_band_over_base'] for q in r['old_recipe']],
        new_median_percent=[100*q['median_band_over_base'] for q in r['new_recipe']]) for r in rows]))


if __name__=='__main__':
    main()
