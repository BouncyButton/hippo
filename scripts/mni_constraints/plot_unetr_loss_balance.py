"""Plot archived same-pair constraint gains across three base CE coefficients."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    reports=ROOT/'reports'
    dicefile=reports/'mni_unetr_plain_diceonly53t_20261109/plain_diceonly_results.json'
    quarterfile=reports/'mni_unetr_plain_ce02553t_20261109/plain_ce025_results.json'
    oldfile=reports/'mni_unetr_plain_diceonly53t_20261109/base_loss_interactions.json'
    read=lambda p:json.loads(p.read_text())
    zero,quarter,old=map(read,(dicefile,quarterfile,oldfile))
    arms=['bands','full','depth','sdf_depth'];labels=['Bands','Full combination','Depth','SDF + depth']
    cells=np.full((4,3),np.nan);baselines={};evidence=[]
    zero_rows={r['candidate'].removeprefix('plain_diceonly_'):r for r in zero['rows'] if (r['seed'],r['fold'])==(0,1)}
    for row in old['rows']:
        if (row['seed'],row['fold'])!=(0,1):continue
        arm=row['candidate'].removeprefix('plain_diceonly_');i=arms.index(arm)
        cells[i,0]=row['previous_constraint_gain_pp']
        baselines[0]=zero_rows[arm]['prior_plain_baseline']['metrics']['validation_percent']
        evidence.append(dict(ce_weight=1.,arm=arm,gain_pp=float(cells[i,0]),archive=row['previous_constraint_archive_sha256']))
    for column,coefficient,result,prefix in [(1,.25,quarter,'plain_ce025_'),(2,0.,zero,'plain_diceonly_')]:
        for row in result['rows']:
            if (row['seed'],row['fold'])!=(0,1):continue
            arm=row['candidate'].removeprefix(prefix);i=arms.index(arm)
            cells[i,column]=row['gain_pp'];baselines[column]=row['baseline']['validation_percent']
            evidence.append(dict(ce_weight=coefficient,arm=arm,gain_pp=row['gain_pp'],archive=row['current']['archive_manifest_sha256']))
    fig,ax=plt.subplots(figsize=(10,5.2))
    palette=plt.get_cmap('RdBu').copy();palette.set_bad('#eeeeee')
    heat=ax.imshow(np.ma.masked_invalid(cells),cmap=palette,vmin=-.75,vmax=.75,aspect='auto')
    ax.set_xticks(range(3),[f'CE weight {c:g}\nBaseline {baselines[i]:.2f}%' for i,c in enumerate([1.,.25,0.])])
    ax.set_yticks(range(4),labels)
    for i in range(4):
        for j in range(3):
            v=cells[i,j];ax.text(j,i,'Pending' if np.isnan(v) else f'{v:+.3f}',ha='center',va='center',color='white' if abs(v)>.45 else 'black')
    ax.set_title('Original constraints under different base-loss balances\nSeed 0 / fold 1 only; final-budget gain over the SAME-loss baseline')
    fig.colorbar(heat,ax=ax,label='Dice gain (percentage points)')
    fig.text(.5,.025,'Same plain UNETR, learning-rate schedule and cap. Each base uses fresh training-only calibration.\nDice-only was rejected: it also caused severe false positives on seeds 5 and 19.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1))
    destination=reports/'mni_unetr_plain_ce02553t_20261109'
    for suffix in ('png','pdf'):fig.savefig(destination/f'loss_balance_first_pair.{suffix}',dpi=160)
    plt.close(fig)
    out=dict(scope='Descriptive development comparison on one pair; not cross-seed confirmation or a selected checkpoint.',rows=evidence,baseline_percent=baselines,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (dicefile,quarterfile,oldfile)},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (destination/'loss_balance_first_pair.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'Plotted {len(evidence)} archived comparisons; missing cells remain pending.')


if __name__=='__main__':main()
