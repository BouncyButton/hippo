"""Summarize rates within held-out partitions, then weight partitions and datasets equally."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent


def rates(cases):
    s=cases.sum(numeric_only=True)
    denominators=['mismatches','union_errors','centers','faces','low','high','class_errors']
    assert all(s[k]>0 for k in denominators)
    return pd.Series({
        'crop_evaluations':len(cases),'mean_crop_foreground_dice_pct':100*cases.dice.mean(),
        'boundary_fg_bg_share_all_errors_pct':100*s.band_union_errors/s.mismatches,
        'boundary_share_fg_bg_errors_pct':100*s.band_union_errors/s.union_errors,
        'same_side_disagreement_pct':100*s.broken/s.faces,
        'crops_with_disagreement_pct':100*(cases.broken>0).mean(),
        'isolated_pct_of_eligible_boundary_errors':100*s.isolated/s.centers,
        'majority_correct_neighbors_pct_of_eligible_boundary_errors':100*s.majority_correct/s.centers,
        'crops_with_isolated_error_pct':100*(cases.isolated>0).mean(),
        'lower_population_boundary_error_rate_pct':100*s.lowerr/s.low,
        'higher_population_boundary_error_rate_pct':100*s.higherr/s.high,
        'lower_population_share_boundary_errors_pct':100*s.lowerr/(s.lowerr+s.higherr),
        'lower_population_share_boundary_voxels_pct':100*s.low/(s.low+s.high),
        'class_errors_beyond_2_pct':100*s.far/s.class_errors,
        'class_errors_beyond_4_pct':100*s.far4/s.class_errors,
        'class_errors_beyond_2_per_crop':s.far/len(cases),
    })


def summarize():
    files=list(OUT.glob('*_f*_s*_u*.csv'))
    cases=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    assert not cases.duplicated(['dataset','fold','seed','step','case']).any()
    groups=cases.groupby(['dataset','fold','seed','step'])
    runs=pd.DataFrame([dict(dataset=d,fold=f,seed=s,step=t,**rates(c)) for (d,f,s,t),c in groups])
    runs.to_csv(OUT/'per_run.csv',index=False)
    if len(runs)!=24:
        print('Coverage:',len(runs),'of 24 evaluations')
        return
    for ds, n in [('MSD',52),('MNI',10),('ADNI',54),('HFH',10)]:
        data=cases[cases.dataset==ds]
        assert len(data)==3*2*n
        for _,g in data.groupby('fold'):
            assert g.groupby('step').case.apply(set).iloc[0]==g.groupby('step').case.apply(set).iloc[1]
    metrics=runs.drop(columns=['fold','seed','crop_evaluations']).groupby(['dataset','step']).mean()
    equal=metrics.groupby('step').mean()
    for t in equal.index:
        metrics.loc[('Equal-dataset mean',t),:]=equal.loc[t]
    metrics.to_csv(OUT/'dataset_means.csv')
    print(metrics.round(2).to_string())


if __name__=='__main__':
    summarize()
