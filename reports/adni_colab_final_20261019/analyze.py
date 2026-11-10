"""Recompute the completed ADNI comparison without running any experiment."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
CAP = 31_976_450_667_090
MODELS = ['nnunet','swinunetr','unetr_vit_b16','basic_unet16']
LABELS = dict(zip(MODELS,['nnUNet','SwinUNETR','UNETR','3D U-Net']))

def read(p): return json.loads(p.read_text())
def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def label(arm): return 'Baseline' if arm=='dice' else arm.removeprefix('dice_').replace('_',' + ').replace('sdf','SDF').title().replace('Sdf','SDF')
def table_csv(name, records):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

data=read(ROOT/'source/comparison.json');rows=data['runs']
old=read(ROOT.parent/'adni_colab_missing_only_20261018/package/historical_snapshot.json')['runs']
scope=read(ROOT.parent/'adni_colab_missing_only_20261018/package/continuation.json')
arms=[x['arm'] for x in data['aggregate'] if x['model']==MODELS[0]]
expected={(m,a,f,s) for m in MODELS for a in arms for f in range(3) for s in (0,1,4)}
actual={(r['model'],r['arm'],r['fold'],r['seed']) for r in rows}
assert len(rows)==len(actual)==504 and actual==expected
assert len({r['id'] for r in rows})==504 and data['completed']==data['total']==504
index={r['id']:r for r in rows}
for historical in old:
    assert all(index[historical['id']][k]==v for k,v in historical.items())
assert sum(r['archive_validation']=='verified_in_continuation' for r in rows)==175
assert all(math.isfinite(r['dice']) and r['training_flops']<=CAP and r['discarded_attempts']==0 for r in rows)
assert all(r['training_flops']+r['unused_flops']==CAP for r in rows)
assert all(not r['ap_swaps_applicable'] for r in rows)
inventory=read(ROOT/'source/run_inventory.json')
inventory_ids={x['title'] for group in inventory.values() for x in group}
assert inventory_ids=={t['id'] for t in scope['seed01_tasks']+scope['seed4_tasks']}
baseline={(r['model'],r['fold'],r['seed']):r for r in rows if r['arm']=='dice'}
for r in rows:
    b=baseline[r['model'],r['fold'],r['seed']]
    assert abs(r['gain_pp']-(r['dice']-b['dice']))<1e-10

aggregate=[]; folds=[]
for m in MODELS:
    for a in arms:
        group=[r for r in rows if (r['model'],r['arm'])==(m,a)]
        means=[st.mean(r['dice'] for r in group if r['fold']==f) for f in range(3)]
        gains=[st.mean(r['gain_pp'] for r in group if r['fold']==f) for f in range(3)]
        b=[baseline[m,r['fold'],r['seed']] for r in group]
        agg=dict(model=m,condition=label(a),arm=a,n=9,validation_dice=st.mean(means),
            sd_across_fold_means=st.stdev(means),gain_pp=st.mean(gains),
            wins_of_9=sum(r['gain_pp']>0 for r in group),positive_folds_of_3=sum(x>0 for x in gains),
            training_dice=st.mean(r['training_dice'] for r in group),
            train_validation_gap_pp=st.mean(r['training_dice']-r['dice'] for r in group),
            validation_fn_mean=st.mean(r['fn'] for r in group),validation_fp_mean=st.mean(r['fp'] for r in group),
            fn_change_pct=100*(sum(r['fn'] for r in group)/sum(r['fn'] for r in b)-1),
            fp_change_pct=100*(sum(r['fp'] for r in group)/sum(r['fp'] for r in b)-1),
            training_fn_mean=st.mean(r['training_fn'] for r in group),training_fp_mean=st.mean(r['training_fp'] for r in group),
            min_updates=min(r['updates'] for r in group),max_updates=max(r['updates'] for r in group),
            optimization_tflops_mean=st.mean(r['training_flops']/1e12 for r in group),
            calibration_tflops_mean=st.mean(r['calibration_flops']/1e12 for r in group),
            budget_used_pct=100*st.mean(r['training_flops']/CAP for r in group))
        saved=next(x for x in data['aggregate'] if (x['model'],x['arm'])==(m,a))
        assert abs(saved['mean_dice']-agg['validation_dice'])<1e-10
        assert abs(saved['sd_across_folds']-agg['sd_across_fold_means'])<1e-10
        aggregate.append(agg)
        for f in range(3):folds.append(dict(model=m,arm=a,fold=f,validation_dice=means[f],gain_pp=gains[f]))

table_csv('condition_summary.csv',aggregate);table_csv('fold_summary.csv',folds)
flat=[{**r,'calibrated_weights':json.dumps(r['calibrated_weights'],sort_keys=True)} for r in rows]
table_csv('all_504_runs.csv',flat)
weights=[dict(id=r['id'],model=r['model'],arm=r['arm'],fold=r['fold'],seed=r['seed'],constraint=k,weight=v)
         for r in rows for k,v in r['calibrated_weights'].items()]
table_csv('calibrated_weights.csv',weights)
best={m:max((a for a in aggregate if a['model']==m),key=lambda a:a['validation_dice']) for m in MODELS}
bands=[a for a in aggregate if a['arm'] in ('dice','dice_bands','dice_bands_edge')]
audit=dict(completed=504,unique_expected_cells=504,all_groups_n=9,folds=[0,1,2],seeds=[0,1,4],
    historical_rows_preserved_exactly=329,continuation_verified_by_saved_report=175,
    continuation_run_folders_observed=175,all_aggregate_statistics_recomputed=True,
    raw_archive_payloads_rehashed_in_this_analysis=False,augmentation=scope['augmentation'],
    optimization_cap=CAP,all_optimization_flops_within_cap=True,
    min_optimization_budget_used_pct=min(r['training_flops']/CAP*100 for r in rows),
    max_optimization_budget_used_pct=max(r['training_flops']/CAP*100 for r in rows),
    total_optimization_pflops=sum(r['training_flops'] for r in rows)/1e15,
    total_calibration_pflops=sum(r['calibration_flops'] for r in rows)/1e15,
    discarded_attempts_in_completed_summaries=0,
    interpretation='Exploratory post-selection seeds; 3 fold-level summaries, not 9 independent cohorts.',
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').iterdir() if p.is_file()})
save(ROOT/'validation.json',audit);save(ROOT/'analysis.json',dict(aggregate=aggregate,best=best,bands=bands,audit=audit))

print('COMPLETE',len(rows),'rows; unchanged historical rows:',len(old))
print('Condition\t'+'\t'.join(LABELS.values()))
for a in arms:
    parts=[label(a)]+[f"{next(x for x in aggregate if x['model']==m and x['arm']==a)['validation_dice']:.2f}" for m in MODELS]
    print('\t'.join(parts))
print('BEST',json.dumps(best,indent=2))
print('BANDS',json.dumps(bands,indent=2))
print('AUDIT',json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},indent=2))

head=''.join('<th>'+LABELS[m]+'</th>' for m in MODELS)
body=''
for a in arms:
    body+='<tr><th>'+html.escape(label(a))+'</th>'
    for m in MODELS:
        item=next(x for x in aggregate if x['model']==m and x['arm']==a)
        color='#e5f4ec' if item['gain_pp']>0 else '#f9e9e6' if item['gain_pp']<0 else '#f2f3f5'
        body+=f"<td style='background:{color}'><strong>{item['validation_dice']:.2f}</strong><br><span>{item['gain_pp']:+.2f} pp · {item['wins_of_9']}/9 wins</span></td>"
    body+='</tr>'
details=''
for m in MODELS:
    b=best[m]
    details+=f"<p><b>{LABELS[m]}</b>: best observed condition, {html.escape(b['condition'])}, {b['validation_dice']:.2f}% Dice ({b['gain_pp']:+.2f} pp). Improved in {b['wins_of_9']}/9 paired runs and {b['positive_folds_of_3']}/3 fold means. Training Dice {b['training_dice']:.2f}%; FN {b['fn_change_pct']:+.1f}%, FP {b['fp_change_pct']:+.1f}% versus baseline.</p>"
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ADNI completed sweep</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#152b3d;background:#f6f8fb;margin:0}main{max-width:1150px;margin:36px auto;padding:30px;background:white;border-radius:18px}h1{font-size:36px;margin:0}small,.note{color:#526372}table{border-collapse:separate;border-spacing:4px;width:100%}th{text-align:left}td,th{padding:10px 13px}td span{font-size:12px;color:#425466}td strong{font-size:19px}.table{overflow:auto}.badge{display:inline-block;background:#dceee5;padding:5px 13px;border-radius:30px}a{color:#125ec5}select{padding:8px;font-size:16px}pre{white-space:pre-wrap}footer{font-size:13px}</style><main>
<span class="badge">504 / 504 completed</span><h1>ADNI: constraint comparison</h1><p>3 folds × 3 seeds · Four models · 14 conditions · No augmentation</p>
<p class="note">Final validation Dice (%). Means average seeds within each fold, then weight the three folds equally. Changes use the baseline with the same model, fold and seed. Each cell has nine runs. “Wins” are descriptive paired comparisons, not independent statistical replications.</p>
<div class="table"><table><thead><tr><th>Added constraint</th>'''+head+'</tr></thead><tbody>'+body+'''</tbody></table></div>
<h2>Observed improvements</h2>'''+details+'''<h2>Inspect training, errors and compute</h2><label>Model <select id="model">'''+''.join('<option value="'+m+'">'+LABELS[m]+'</option>' for m in MODELS)+'''</select></label><div class="table"><table id="details"></table></div>
<h2>Scope and checks</h2><p>All 504 unique expected cells are present; every condition/model has nine results. All aggregate Dice values and fold standard deviations were independently recomputed. All reported optimization costs fit the 31.97645 TFLOP per-run cap; calibration is separately recorded. No discarded updates are reported in the completed rows.</p>
<p>175 continuation results were verified by the saved notebook reporting pass, and all 175 result folders are present. The 329 historical rows match the pinned earlier snapshot exactly. This analysis did not rescan every raw archive or independently rehash every checkpoint. The saved completion log records per-run evaluation, archive verification and cleanup.</p>
<p><b>Interpretation:</b> seeds are 0, 1 and 4; seed 4 replaced seed 2 after results were inspected. The study and the best-condition ranking are exploratory. nnUNet uses native Dice+CE, SGD and 60% constraint targets; the other three use Dice, AdamW and 10%. Cross-model differences therefore combine architecture, baseline-loss and training-recipe differences. ADNI has one foreground class; anterior/posterior swaps do not apply.</p>
<p><a href="condition_summary.csv">56-condition summary CSV</a> · <a href="all_504_runs.csv">All 504 runs</a> · <a href="fold_summary.csv">Fold means</a> · <a href="calibrated_weights.csv">Calibration weights</a> · <a href="validation.json">Analysis checks</a></p>
<footer>Source: <a href="https://drive.google.com/file/d/1uCeHThzQM6Vt6tngOZHFXReoA9gc4Ybs/view">saved Drive comparison</a>, 19 October 2026. No training or Drive writes were performed for this analysis.</footer>
<script>const rows='''+json.dumps(aggregate)+''';function render(){const m=document.querySelector('#model').value;document.querySelector('#details').innerHTML='<tr><th>Condition</th><th>Train Dice</th><th>Val Dice</th><th>FN change</th><th>FP change</th><th>Updates</th><th>Cap used</th></tr>'+rows.filter(r=>r.model===m).map(r=>`<tr><th>${r.condition}</th><td>${r.training_dice.toFixed(2)}</td><td>${r.validation_dice.toFixed(2)}</td><td>${r.fn_change_pct.toFixed(1)}%</td><td>${r.fp_change_pct.toFixed(1)}%</td><td>${r.min_updates}–${r.max_updates}</td><td>${r.budget_used_pct.toFixed(2)}%</td></tr>`).join('')}document.querySelector('#model').onchange=render;render();</script></main></html>'''
(ROOT/'ADNI_results.html').write_text(page)
