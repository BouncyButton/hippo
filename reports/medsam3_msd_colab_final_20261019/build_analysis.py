"""Recompute the completed matched study and render exportable results."""
from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
source = json.loads((ROOT / 'comparison.json').read_text())
evidence = json.loads((ROOT / 'verification.json').read_text())
calibrations = json.loads((ROOT / 'calibration.json').read_text())
data = pd.DataFrame(source['runs'])
assert len(data) == data.id.nunique() == 108
assert set(data.updates) == {200} and set(data.augmentation) == {'none'}
assert set(data.status) == {'complete'}
assert evidence['session_result']['status'] == 'complete'
assert evidence['session_result']['error'] is None
assert evidence['last_cleanup']['status'] == 'complete'
assert set(data.fold) == {0, 1, 3} and set(data.seed) == {17, 83, 191}
baseline = data[data.arm == 'baseline'].set_index(['fold', 'seed']).validation_dice
for row in data.itertuples():
    assert abs(row.gain_pp - 100*(row.validation_dice - baseline.loc[row.fold, row.seed])) < 1e-9
assert data.groupby(['arm', 'fold']).size().eq(3).all()
summary = data.groupby('arm').agg(
    runs=('id', 'count'), validation_dice_pct=('validation_dice', 'mean'),
    training_dice_pct=('training_dice', 'mean'), gain_pp=('gain_pp', 'mean'),
    minimum_gain_pp=('gain_pp', 'min'), maximum_gain_pp=('gain_pp', 'max'),
    improved_pairs=('gain_pp', lambda x: int((x > 0).sum())),
    foreground_fn=('foreground_fn', 'mean'), foreground_fp=('foreground_fp', 'mean'),
    ap_swaps=('ap_swaps', 'mean'))
summary[['validation_dice_pct', 'training_dice_pct']] *= 100
summary['train_validation_gap_pp'] = summary.training_dice_pct-summary.validation_dice_pct
summary = summary.sort_values('validation_dice_pct', ascending=False)
for row in source['aggregate']:
    if row['scope'] == 'fresh_matched':
        assert abs(summary.loc[row['arm'], 'validation_dice_pct'] - row['dice']) < 1e-9
        assert abs(summary.loc[row['arm'], 'gain_pp'] - row['gain_pp']) < 1e-9
fold_dice = data.pivot_table(index='arm', columns='fold', values='validation_dice')*100
fold_gain = data.pivot_table(index='arm', columns='fold', values='gain_pp')
paired = data.pivot(index=['fold','seed'], columns='arm', values='gain_pp')
labels = lambda x: x.replace('bands','Bands').replace('edge','Edge').replace('sdf','SDF').replace('depth','Depth').replace('_',' + ').replace('baseline','Baseline')
order = list(summary.index)
colors = {0:'#2272b5', 1:'#d87c1d', 3:'#209387'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,2,figsize=(14,7.8),gridspec_kw={'width_ratios':[1.35,1]})
ax=axes[0]
for y,arm in enumerate(order):
    vals=paired[arm]
    ax.hlines(y,vals.min(),vals.max(),color='#cbd5e1',lw=2,zorder=1)
    for f in [0,1,3]:
        v=vals.loc[f].values
        ax.scatter(v,y+np.array([-.16,0,.16]),s=23,color=colors[f],alpha=.85,label=f'Fold {f}' if y==0 else None,zorder=2)
    ax.scatter(summary.loc[arm,'gain_pp'],y,marker='D',s=35,color='#16283d',edgecolors='white',linewidths=.7,zorder=3)
ax.axvline(0,color='#64748b',lw=1)
ax.set_yticks(range(len(order)),[labels(a) for a in order]);ax.invert_yaxis()
ax.set_xlabel('Paired validation Dice gain (percentage points)')
ax.set_xlim(-2.4,9);ax.grid(axis='x',alpha=.15)
ax.set_title('Gains against each fresh matched baseline',loc='left',fontweight='bold',pad=18)
ax.legend(frameon=False,loc='lower right',title='Dots: individual seeds; diamond: mean',fontsize=9,title_fontsize=9)
ax=axes[1]
v=fold_dice.loc[['baseline','bands_edge_sdf_depth']]
x=np.arange(3)
ax.bar(x-.19,v.loc['baseline'],.36,color='#a9b7c7',label='Baseline')
ax.bar(x+.19,v.loc['bands_edge_sdf_depth'],.36,color='#185d79',label='Bands + Edge + SDF + Depth')
for i,f in enumerate([0,1,3]):
    for shift,arm in [(-.19,'baseline'),(.19,'bands_edge_sdf_depth')]:
        value=v.loc[arm,f];ax.text(i+shift,value+.3,f'{value:.2f}',ha='center',fontsize=10)
    gain=fold_gain.loc['bands_edge_sdf_depth',f]
    ax.text(i,87,f'+{gain:.2f} pp',ha='center',fontweight='bold',color='#185d79')
ax.set_ylim(0,92);ax.set_xticks(x,['Fold 0','Fold 1','Fold 3']);ax.set_ylabel('Validation Dice (%)')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.set_title('Best mean condition improves every fold',loc='left',fontweight='bold',pad=18)
ax.legend(frameon=False,loc='lower center',fontsize=9)
fig.suptitle('MedSAM3 fine-tuning · MSD · 108 completed runs',x=.02,ha='left',fontsize=19,fontweight='bold')
fig.text(.02,.925,'3 selected folds × 3 seeds × 12 fresh conditions · 200 updates/run · no augmentation',fontsize=11,color='#475569')
fig.text(.02,.015,'Means average seeds within folds, then weight folds equally. Descriptive results; seeds are not nine independent data splits.\nHistorical bands and bands+edge runs are excluded from matched comparisons. No FLOP cap was applied.',fontsize=9,color='#475569')
fig.tight_layout(rect=[0,.07,1,.89]);fig.savefig(ROOT/'results_overview.png',dpi=180);fig.savefig(ROOT/'results_overview.pdf');plt.close(fig)

calrows=[]
for fold,c in calibrations.items():
    for arm,v in c['arms'].items():
        assert v['target']==.1 and v['p95_cap']==.5
        for constraint,d in v['diagnostics'].items():
            assert np.isfinite(d['weight']) and d['weight']>0 and d['p95']<=.5+1e-10
            calrows.append(dict(fold=int(fold),arm=arm,constraint=constraint,weight=d['weight'],median=d['median'],p95=d['p95'],valid_slabs=d['valid_slabs']))
cal=pd.DataFrame(calrows)
cal.to_csv(ROOT/'calibrated_weights.csv',index=False)
summary.to_csv(ROOT/'condition_summary.csv');data.to_csv(ROOT/'all_108_runs.csv',index=False)
fold_dice.to_csv(ROOT/'fold_validation_dice.csv');fold_gain.to_csv(ROOT/'fold_paired_gains.csv')
historical=pd.DataFrame(source['historical_reference']).groupby('arm').agg(validation_dice_pct=('validation_dice','mean'))*100
historical['within_cluster_gain_pp']=historical.validation_dice_pct-historical.loc['baseline','validation_dice_pct']
hist_base=pd.DataFrame(source['historical_reference']).query("arm == 'baseline'").set_index(['fold','seed']).validation_dice
winner=data.query("arm == 'bands_edge_sdf_depth'").sort_values(['fold','seed']).copy()
winner['baseline_dice_pct']=[100*baseline.loc[r.fold,r.seed] for r in winner.itertuples()]
winner['validation_dice_pct']=winner.validation_dice*100
winner['training_dice_pct']=winner.training_dice*100
selection=summary[['validation_dice_pct','gain_pp','training_dice_pct','improved_pairs','foreground_fn','foreground_fp','ap_swaps']].rename_axis('condition').reset_index()
selection['condition']=selection.condition.map(labels)
fold_display=fold_dice.loc[order].copy();fold_display.index=fold_display.index.map(labels)
table=lambda frame:frame.to_html(index=False if isinstance(frame.index,pd.RangeIndex) else True,float_format=lambda n:f'{n:.2f}',border=0,classes='results')
quota=evidence['storage_quota_last_recorded']['data']
quota_text=f"Last recorded quota check (19 October, 17:47 CEST): {int(quota['usage'])/1024**4:.3f} TiB used out of {int(quota['limit'])/1024**4:.0f} TiB; {int(quota['usageInDriveTrash'])/1024**4:.3f} TiB was in Trash. This is an account-wide snapshot, not the campaign size. No Trash was emptied."
body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>MedSAM3 MSD final results</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#17283e;max-width:1280px;margin:40px auto;padding:0 26px}}h1{{font-size:32px}}h2{{margin-top:38px}}p{{max-width:1050px}}.meta{{color:#526477}}.results{{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}}th,td{{padding:9px 10px;border-bottom:1px solid #dce3ea;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead{{background:#edf3f7}}img{{width:100%;height:auto}}.note{{background:#eef5f8;padding:16px 20px;border-left:4px solid #2272b5}}a{{color:#126491}}</style>
<h1>MedSAM3 MSD: final matched results</h1><p class="meta">Completed 19 October 2026, 18:47 CEST · filippofocaccia2@gmail.com</p>
<p class="note"><strong>108/108 complete.</strong> Bands + Edge + SDF + Depth achieved <strong>81.73% validation Dice</strong>, compared with <strong>76.34%</strong> for the fresh baseline: <strong>+5.38 percentage points</strong>. It improved all 9 matched fold/seed pairs.</p>
<p>Final-update checkpoints after 200 updates; folds 0, 1, 3; seeds 17, 83, 191; no augmentation. Dice is the per-volume mean of anterior and posterior Dice, averaged over validation volumes in each run, seeds within each fold, then folds equally. Training Dice uses the saved training-support evaluation.</p>
<img src="results_overview.png" alt="Paired gains for all conditions and per-fold baseline versus full combination">
<h2>All fresh conditions</h2><p>Every row contains 9 runs. Gain is paired to the same fold and seed. FN, FP and anterior/posterior swaps are mean voxel counts per validation volume, averaged equally across runs; they are not rates.</p>{table(selection)}
<h2>Consistency across folds</h2>{table(fold_display)}
<h2>Best mean condition: all nine matched pairs</h2>{table(winner[['fold','seed','baseline_dice_pct','validation_dice_pct','gain_pp','training_dice_pct']].reset_index(drop=True))}
<h2>Interpretation</h2><ul><li>Full combination gains range from +3.33 to +8.03 points. Its fold-mean gains are +4.74, +5.71 and +5.71 points.</li><li>The full combination leads Edge + SDF + Depth by only 0.44 points and wins 7/9 pairs. This smaller ranking difference is less secure than the improvement over baseline.</li><li>Full-combination FN falls from 869.18 to 555.99 voxels/volume (−36.0%); FP rises from 310.63 to 436.97 (+40.7%); A/P swaps fall from 123.13 to 97.59 (−20.7%). This is a sensitivity/false-positive trade-off.</li><li>Training Dice rises from 81.00% to 86.88%, alongside validation improvement. The train–validation gap is 4.66 points for baseline and 5.16 for the full combination.</li><li>SDF alone is weak and variable (+0.62 points, 6/9 positive pairs; negative fold-mean gains in folds 0 and 1). Depth alone improves 8/9 pairs. Every fresh multi-constraint condition improves all 9 pairs.</li><li>Selected folds and repeated seeds do not establish statistical significance or nine independent test sets. The condition ranking is exploratory, and was selected using these validation results.</li></ul>
<h2>Original cluster study: separate historical reference</h2>{table(historical)}<p>Bands and Bands + Edge were deliberately not rerun. These 27 historical results use their own cluster baseline and calibration. They were not substituted into the fresh Colab controls; cross-environment differences cannot be attributed solely to constraints.</p>
<h2>Calibration and budget checks</h2><p>All 108 rows confirm no augmentation and exactly 200 updates (21,600 optimizer updates in this fresh study). This study is update-matched, not FLOP-capped or compute-matched to the four-model sweeps. Calibration is accounted for separately. All fold calibration records retain the 10% sequential target and 50% p95 cap; measured medians can be below 10% when the p95 cap binds. Coefficients were calibrated in the Colab environment and are not identical to historic cluster coefficients.</p>
<p>The last run records AdamW, learning rate 5e−5 to 5e−6, 20-update warm-up, accumulation 2, gradient clipping at 1, native BF16 autocast, 18,497,664 trainable parameters and final-update selection. Frozen-parameter hashes match before/after; adapter hashes changed.</p>
<p><a href="calibrated_weights.csv">All fold/condition calibration coefficients and diagnostics</a></p>
<h2>Completion, storage and evidence</h2><p>The final session receipt is complete with no error and 108 archived experiments. The report generator includes runs only after archive-file hashes and complete cleanup receipts pass. The last run's cleanup receipt confirms removal of its checkpoint and bulky recovery archive after verification. This analysis independently recomputed every paired gain and aggregate from the 108 stored completion rows; it did not redownload and rehash all archive files.</p><p>{quota_text}</p>
<p><a href="https://drive.google.com/drive/folders/1TZwrPM1EJf-uFkAIOYjzfjH0iUC-CbbE">Drive results</a> · <a href="all_108_runs.csv">All 108 runs CSV</a> · <a href="condition_summary.csv">Condition summary CSV</a> · <a href="results_overview.pdf">Exportable figure PDF</a> · <a href="comparison.json">Original comparison JSON</a> · <a href="verification.json">Completion evidence</a></p></html>'''
(ROOT/'results.html').write_text(body)
(ROOT/'analysis_checks.json').write_text(json.dumps(dict(rows=108,unique_tasks=108,matched_pairs_per_condition=9,conditions=12,updates_per_run=200,augmentation='none',all_aggregate_and_gain_recomputations_passed=True,all_calibration_p95_caps_passed=True,finished='2026-10-19T16:47:47Z'),indent=2)+'\n')
print('Verified 108 rows and all paired gains; created HTML, CSV, PNG and PDF results.')
