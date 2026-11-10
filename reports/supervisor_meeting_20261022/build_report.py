from pathlib import Path
import csv, json, html, hashlib
from statistics import mean
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def read(p): return json.loads((ROOT/p).read_text())
def csvread(p): return list(csv.DictReader((ROOT/p).open()))
def label(a): return 'Baseline' if a in ('dice','baseline') else a.removeprefix('dice_').replace('_',' + ').title().replace('Sdf','SDF')
def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def f(v): return f'{float(v):.2f}'
names={'nnunet':'nnUNet','swinunetr':'SwinUNETR','unetr_vit_b16':'UNETR','basic_unet16':'3D U-Net'}
models=list(names)
msd=csvread('reports/msd_3fold_3seed_final_20261020/all_56_condition_summaries.csv')
adni=csvread('reports/adni_colab_final_20261019/condition_summary.csv')
med=csvread('reports/medsam3_msd_colab_final_20261019/condition_summary.csv')
admed=read('reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/comparison.json')
geo=read('reports/geofwi3d_reweighted_full_suite_20261021/final_summary.json')
live=read('reports/supervisor_meeting_20261022/live_snapshot.json')
sam=read('reports/sam3_geofwi_f013_s012_126_20261019/split_cluster_20261022/handoff/source_campaign_state.json')['completed']
sam.update(live['sam3_f013_s012_cluster_tail_20261022']['results'])
sb={(r['fold'],r['seed']):r for r in sam.values() if r['arm']=='dice'}
arms=list(dict.fromkeys(r['arm'] for r in sam.values()))
common=set.intersection(*[{(r['fold'],r['seed']) for r in sam.values() if r['arm']==a} for a in arms])
sam_summary=[]
for a in arms:
    rows=[r for r in sam.values() if r['arm']==a]
    gains=[100*(r['validation_dice']-sb[r['fold'],r['seed']]['validation_dice']) for r in rows]
    shared=[r for r in rows if (r['fold'],r['seed']) in common]
    sam_summary.append(dict(arm=a,n=len(rows),dice=mean(r['validation_dice']*100 for r in rows),baseline=mean(sb[r['fold'],r['seed']]['validation_dice']*100 for r in rows),gain=mean(gains),wins=sum(g>0 for g in gains),common_dice=mean(r['validation_dice']*100 for r in shared),common_gain=mean(100*(r['validation_dice']-sb[r['fold'],r['seed']]['validation_dice']) for r in shared)))
assert len(sam)==85 and len(common)==5 and admed['completed']==126
when=datetime.fromtimestamp(live['checked_at'],ZoneInfo('Europe/Rome')).strftime('%d October %Y, %H:%M CEST')
parts=[f'<header><p class="eyebrow">SUPERVISOR MEETING · {when}</p><h1>Segmentation results across medical and seismic data</h1><p class="lead">Matched geometric constraints improve the completed medical studies and all four nonmedical models. The useful combination and strength depend on the model. SAM3 remains preliminary.</p></header>']
parts.append('<nav><a href="#medical">Medical models</a><a href="#foundation">MedSAM3</a><a href="#geo">Nonmedical models</a><a href="#sam">SAM3</a><a href="#methods">Budgets & methods</a></nav>')
parts.append(table(['Study','Completion','Folds / seeds','Budget per run'],[
    ['MSD · 4 models','504 / 504','0,1,2 / 0,2,20','31.97645 TFLOPs optimization; calibration extra'],
    ['ADNI · 4 models','504 / 504','0,1,2 / 0,1,4','31.97645 TFLOPs optimization; calibration extra'],
    ['MedSAM3 · MSD','108 / 108 fresh + 27 historical','0,1,3 / 17,83,191','200 updates; no FLOP cap'],
    ['MedSAM3 · ADNI','126 / 126','0,1,3 / 0,1,2','200 updates; no FLOP cap'],
    ['GeoFWI3D · 4 models','468 constraints + 36 baselines complete','0,1,2 / 0,1,2','182.17813 TFLOPs including calibration'],
    ['SAM3 · GeoFWI3D','85 / 126; still running','0,1,3 / 0,1,2 planned','25 updates; no equal-FLOP claim']]))
parts.append('<p class="note">All Dice values below are percentages. Gains are percentage points against a matched baseline. Primary results use final-budget or final-update checkpoints. Best combinations are selected on validation, not confirmed on an independent test set.</p>')
parts.append('<section id="medical"><h2>MSD and ADNI: completed 3-fold × 3-seed studies</h2>')
headline=[]
for title,data in [('MSD',msd),('ADNI',adni)]:
    for m in models:
        rows=[r for r in data if r['model']==m];best=max(rows,key=lambda r:float(r['validation_dice']));base=next(r for r in rows if r.get('arm',r['condition'])=='dice')
        a=best.get('arm',best['condition'])
        headline.append([title,names[m],f(base['validation_dice']),f(best['validation_dice']),f(best['gain_pp']),label(a),'9/9'])
parts.append(table(['Dataset','Model','Baseline','Best mean','Gain pp','Best observed constraints','Wins'],headline))
parts.append('<p><b>Consistency:</b> Bands alone improves all 36 matched model/fold/seed pairs in each dataset (72/72 across MSD and ADNI). Each model’s selected best also improves all nine pairs. SDF alone is weak; adding every constraint is not always optimal.</p>')
for title,data in [('MSD',msd),('ADNI',adni)]:
    order=list(dict.fromkeys(r.get('arm',r['condition']) for r in data))
    rows=[]
    for a in order:
        rr=[label(a)]
        for m in models:
            r=next(r for r in data if r['model']==m and r.get('arm',r['condition'])==a)
            rr.append(f"{f(r['validation_dice'])} ({float(r['gain_pp']):+.2f})")
        rows.append(rr)
    parts.append(f'<details><summary>{title}: all 14 conditions, Dice (gain pp)</summary>'+table(['Constraints']+[names[m] for m in models],rows)+'</details>')
parts.append('<p><b>Error tradeoffs:</b> MSD nnUNet’s best reduces foreground FN by 33.5% and A/P swaps by 10.8%, with FP +4.5%. MSD Swin, UNETR and 3D U-Net primarily reduce FP (59.9%, 23.8%, 42.2%) while increasing FN. On ADNI, nnUNet reduces both FN (2.7%) and FP (7.0%); the other models reduce FP by 27–34% while increasing FN. These are voxel-count changes, not error rates. ADNI has no A/P labels.</p></section>')
parts.append('<section id="foundation"><h2>MedSAM3 fine-tuning: both datasets complete</h2>')
parts.append(table(['Dataset','Baseline','Best mean','Gain pp','Best observed constraints','Matched wins'],[
    ['MSD, fresh Colab',f(next(r for r in med if r['arm']=='baseline')['validation_dice_pct']),f(med[0]['validation_dice_pct']),f(med[0]['gain_pp']),'Bands + Edge + SDF + Depth','9/9'],
    ['ADNI','77.32','80.85','3.53','Edge + SDF + Depth','9/9']]))
parts.append('<p>ADNI’s stale 109-row summary was regenerated from all 126 archived results, checking archive hashes, protocol, initialization, sampling, frozen parameters and cleanup. Live cluster status also confirms completion.</p>')
parts.append('<details><summary>MedSAM3 MSD: all fresh conditions</summary>'+table(['Constraints','N','Validation Dice','Training Dice','Gain pp','Wins'],[[label(r['arm']),r['runs'],f(r['validation_dice_pct']),f(r['training_dice_pct']),f(r['gain_pp']),r['improved_pairs']] for r in med])+'</details>')
ar=[]
for a in admed['aggregates']:
    rows=[r for r in admed['runs'] if r['arm']==a['arm']]
    ar.append([label(a['arm']),len(rows),f(a['mean_validation_dice_percent']),f(mean(r['training_dice_percent'] for r in rows)),f(a['mean_gain_percentage_points']),sum(r['gain_over_baseline_percentage_points']>0 for r in rows)])
parts.append('<details><summary>MedSAM3 ADNI: all 14 conditions</summary>'+table(['Constraints','N','Validation Dice','Training Dice','Gain pp','Wins'],ar)+'</details>')
parts.append('<p><b>Historical MSD bands evidence:</b> original cluster baseline 74.93%, bands 78.76% (+3.83 pp), bands+edge 80.12% (+5.19 pp), each over nine runs. These 27 runs retain their own baseline; do not compare them against the fresh Colab baseline of 76.34%.</p><p><b>What changed:</b> medical pretrained LoRA initialization, frozen backbone and trainable adapters, original native detection/presence/mask loss plus calibrated geometry losses. The verified 200-update recipe uses AdamW, LR 5×10⁻⁵ → 5×10⁻⁶, 20-update warm-up, accumulation 2, BF16 and no augmentation. ADNI uses only two support crops from two subjects, with 54 validation crops / 27 subjects per fold. Folds 0 and 3 share the two support samples.</p><p>On MSD, the full combination reduces missed foreground by 36.0% and A/P swaps by 20.7%, but increases FP by 40.7%. Its lead over Edge+SDF+Depth is only 0.44 pp. On ADNI, the all-four mean is 80.74%, close to the 80.85% best.</p></section>')
parts.append('<section id="geo"><h2>Nonmedical GeoFWI3D: salt segmentation</h2><p>Four models × three folds × three seeds. All 468 constraint results and 36 matched baselines are complete. Final results combine 381 new constraint runs and 87 compatible reused ones.</p>')
gr=[]
for m in models:
    d=geo['models'][m];b=d['arms'][0]
    gr.append([names[m],f(d['baseline_mean']),f(b['mean']),f(b['delta']),label(b['arm']),str(b['wins'])+'/9'])
parts.append(table(['Model','Baseline','Best mean','Gain pp','Best observed constraints','Wins'],gr))
grow=[]
for a in ['dice']+[a['arm'] for a in geo['models']['nnunet']['arms']]:
    row=[label(a)]
    for m in models:
        d=geo['models'][m]
        if a=='dice':row.append(f(d['baseline_mean']))
        else:
            r=next(x for x in d['arms'] if x['arm']==a);row.append(f"{f(r['mean'])} ({r['delta']:+.2f})")
    grow.append(row)
parts.append('<details><summary>GeoFWI3D: all 14 conditions, Dice (gain pp)</summary>'+table(['Constraints']+[names[m] for m in models],grow)+'</details>')
parts.append('<p><b>How these results were reached:</b> native 96³ volumes, 19 training volumes (saved 5% subset), 96 validation volumes, fixed seeds 0/1/2, full-budget stopping. UNETR needed an explored recipe with peak LR 6×10⁻⁴ and XY D4 augmentation. Other models use no augmentation. The frozen baseline recipe was reused for each matched constraint run.</p><p>Constraint-strength exploration led to UNETR 4× (gradient target 0.4, p95 ceiling 2.0) and nnUNet 6× (target 3.6, ceiling 4.5). Swin and 3D U-Net retain target 0.1 / ceiling 0.5. These are gradient-ratio targets, not loss coefficients or compute multipliers. The 182.178-TFLOP cap is identical across models and strengths. nnUNet’s fold-0 Bands+SDF+Depth improved from 72.53% at original strength to 78.72% at 6×; the completed broader sweep then found all four best at 78.22% across folds.</p><p>463/468 constraint runs exceeded 50% Dice. Five UNETR SDF-containing runs on fold 0 / seed 0 did not. All nine runs of each reported winner exceeded 50%. No replacement seeds were used in this final suite; recipe/strength exploration still makes its conclusions exploratory.</p></section>')
parts.append('<section id="sam"><h2>SAM3 on GeoFWI3D: partial, 85/126</h2><p>Live snapshot: '+when+'. Five support volumes and 96 validation volumes per fold, 25 updates; folds 0/1/3 and seeds 0/1/2 planned. Job 713003 was running two workers. 41 cells remain; one had a checkpoint/replay recovery issue being handled in the training chat.</p><p class="note">Conditions have unequal coverage (5–7 completed pairs). Do not rank their raw means directly. The table shows matched baselines and gains, plus a comparison restricted to the same five fully completed pairs: fold 0 seeds 0/1/2 and fold 1 seeds 0/1.</p>')
parts.append(table(['Constraints','N / 9','Matched baseline','Dice','Paired gain','Wins','Same 5 pairs Dice','Same 5 gain'],[[label(r['arm']),str(r['n'])+'/9',f(r['baseline']),f(r['dice']),f(r['gain']),str(r['wins'])+'/'+str(r['n']),f(r['common_dice']),f(r['common_gain'])] for r in sam_summary]))
parts.append('<p>On the same five pairs, Bands+Edge+Depth currently leads at 71.86% vs 67.16% baseline (+4.70 pp); bands alone reaches 71.58% (+4.41 pp). Across its six available pairs, Bands+Edge+Depth reaches 72.92% vs matched 68.53% (+4.38 pp, 5/6 wins). Fold 3 has started poorly for several constraints, so the final ranking may change substantially.</p></section>')
parts.append('<section id="methods"><h2>Compute budgets and the explanation to give supervisors</h2>')
parts.append(table(['Study','Per-run budget','Accounting and interpretation'],[
    ['MSD / ADNI conventional','31,976,450,667,090 FLOPs = 31.97645 TFLOPs','Optimization capped; calibration and recovery overhead recorded separately.'],
    ['GeoFWI3D conventional','182,178,126,543,160 FLOPs = 182.17813 TFLOPs','Optimization + calibration capped. Nominal cap is 5.70× the medical cap; accounting differs.'],
    ['MedSAM3','200 optimizer updates','Update-matched within study, not compute-matched to conventional models.'],
    ['SAM3','25 optimizer updates','Partial update-matched experiment; no cross-model equal-FLOP claim.']]))
parts.append('<p><b>Why constraints help:</b> bands supervise the region around the target boundary; edge terms add boundary detail; SDF encodes signed distance; depth weights supervision using the geometry. Gradient calibration scales auxiliary terms relative to the native objective, with warm-up and a p95 ceiling. The observed improvements support better spatial supervision under limited labels/compute; these experiments do not isolate a universal causal mechanism.</p><p><b>Medical training:</b> nnUNet uses native foreground Dice+CE, deep supervision and SGD (60% auxiliary target, 75% p95 ceiling). SwinUNETR, UNETR and 3D U-Net use Dice and AdamW (10% target, 50% ceiling). All medical sweeps have no augmentation. FLOP-aware schedules give cheaper models more updates, while charging auxiliary computation.</p><p><b>Execution:</b> cached preprocessing, validated GPU acceleration and concurrent independent workers made the sweeps practical. Frozen protocols and initializations preserve matched comparisons; resumable ledgers track interruptions. Evaluations and compact archives are verified before checkpoint cleanup. These execution improvements should be described as throughput/reliability work, not as scientific causes of Dice improvement.</p><p><b>Limits:</b> MSD seed 20 replaced seed 1, ADNI seed 4 replaced seed 2 after inspection. Different model losses, optimizers, strengths, supports, folds and augmentation prevent clean architecture-only comparisons. Fold/seed cells are not nine independent test sets. Report wins descriptively; do not claim significance from these tables. Foundation-model pretraining and fixed-update budgets prevent a fair compute-efficiency ranking against training from scratch.</p></section>')
sources=[('MSD detailed analysis','reports/msd_3fold_3seed_final_20261020/analysis.html'),('ADNI detailed analysis','reports/adni_colab_final_20261019/ADNI_results.html'),('MedSAM3 MSD analysis','reports/medsam3_msd_colab_final_20261019/results.html'),('MedSAM3 ADNI verified complete JSON','reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/comparison.json'),('GeoFWI3D all fold/seed results','reports/geofwi3d_reweighted_full_suite_20261021/final_report.html'),('Live cluster snapshot','reports/supervisor_meeting_20261022/live_snapshot.json')]
parts.append('<footer><h2>Evidence and detailed results</h2>'+''.join(f'<p><a href="../../{p}">{n}</a></p>' for n,p in sources)+'<p>Prepared from local durable archives and a read-only cluster status check. No training jobs changed. All numbers in this report are rounded from stored values.</p></footer>')
css='body{margin:0;background:#f4f6f8;color:#152336;font:16px/1.55 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:40px 32px;background:white}h1{font-size:38px;line-height:1.15;max-width:950px}h2{margin-top:38px}.eyebrow{color:#466579;font-size:13px;letter-spacing:.1em}.lead{font-size:21px;max-width:1100px}.note{background:#edf4f8;border-left:4px solid #277397;padding:16px 22px}nav{display:flex;gap:20px;flex-wrap:wrap;margin:28px 0}a{color:#145f85}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;margin:18px 0}th{background:#eaf0f5;text-align:left}td,th{padding:10px 12px;border-bottom:1px solid #d8e1e7}tr:nth-child(even){background:#f7f9fb}.scroll{overflow-x:auto}details{border:1px solid #d8e1e7;padding:15px;margin:18px 0}summary{cursor:pointer;font-weight:650}section{padding-top:8px}footer{border-top:2px solid #d8e1e7;margin-top:40px;font-size:14px}@media print{body{background:white}main{padding:0}nav{display:none}details{break-inside:avoid}h2{break-after:avoid}table{font-size:11px}}'
(OUT/'meeting_results.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Segmentation results — 22 October 2026</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
(OUT/'sam3_partial_summary.json').write_text(json.dumps({'completed':len(sam),'common_pairs':sorted(common),'summary':sam_summary,'runs':sam},indent=2))
(OUT/'provenance.json').write_text(json.dumps({'snapshot_time':when,'sources':[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for _,p in sources]},indent=2))
print(OUT/'meeting_results.html')
