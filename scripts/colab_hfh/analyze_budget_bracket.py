"""Locate native learning transitions in source-bound historical curves."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile


def main():
    root=Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111'
    base=json.loads((root/'bands_v1_payload.json').read_text())
    output=root/'lower_budget50_analysis';output.mkdir(exist_ok=True)
    curves={};sources={}
    for key,reference in base['references'].items():
        if not key.startswith('nnunet_f'):continue
        path=root/'recipe_analysis'/(key+'.zip')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=reference['archive_sha256']:raise ValueError('Historical archive differs')
        with zipfile.ZipFile(path) as archive:
            manifest=json.loads(archive.read('results/archive_manifest.json'))
            data=archive.read('results/metrics.csv')
            if hashlib.sha256(data).hexdigest()!=manifest['metrics.csv']:
                raise ValueError('Historical curve differs from archived manifest')
            curves[key]={int(r['step']):100*float(r['macro_hard_dice'])
                         for r in csv.DictReader(io.StringIO(data.decode()))}
            sources[key]=digest
    if len(curves)!=9:raise ValueError('Expected nine native curves')
    common=sorted(set.intersection(*(set(c) for c in curves.values())))
    rows=[]
    for step in common:
        values=[dict(pair=key,dice_pct=curve[step]) for key,curve in sorted(curves.items())]
        scores=[r['dice_pct'] for r in values]
        rows.append(dict(updates=step,pairs=9,mean_pct=statistics.mean(scores),
            worst_pct=min(scores),best_pct=max(scores),below50=sum(s<50 for s in scores),values=values))
    first=next(r for r in rows if r['below50']==0)
    report=dict(interpretation='Source-bound historical native curves. Exploratory budget bracketing only; '
        'no constrained result, independent endpoint replication, or best-checkpoint selection implied.',
        rows=rows,source_archives=sources,first_logged_all_nine_ge50_updates=first['updates'],
        user_selected_followup=dict(budget_fraction=[3,5],expected_native_updates=54,
            decision='User selected54updates after reviewing the native learning bracket. '
                     'Keep all pairs including the historically weak control, loss strengths and original native LR trajectory; '
                     'scale constraint timing by the declared rule.'))
    (output/'native_budget_bracket_preview.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.6))
    for key,curve in curves.items():
        ax.plot(common,[curve[s] for s in common],color='#8594a6',alpha=.45,lw=1)
    ax.plot(common,[r['mean_pct'] for r in rows],color='#164b73',lw=2.4,label='Mean of all nine native pairs')
    ax.axhline(50,color='#777',ls=':',label='50% Dice')
    for step,label,color in ((45,'50% cap: 45 updates','#bd3b32'),(54,'Selected 60% cap: 54 updates','#337f55'),(67,'75% cap: 67 updates','#a7751d')):
        ax.axvline(step,color=color,ls='--',lw=1.2,label=label)
    ax.set(xlim=(30,91),ylim=(-2,85),xlabel='Optimizer updates; original LR horizon remains 90',
           ylabel='Validation foreground Dice (%)',title='nnU-Net: two native pairs learn foreground after update 45')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=8,loc='lower right')
    fig.text(.5,.01,'Historical curves across fixed 3 folds × 3 seeds; these do not predict constrained performance.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(output/'native_budget_bracket.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(curves=9,first_logged_all_nine_ge50=first['updates'],
                         user_selected_budget_fraction=[3,5],output=str(output))))


if __name__=='__main__':main()
