"""Audit the three archived HFH conditional-BAND runs without retraining."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/hfh_colab_exploration_20261111/nnunet_conditional_band_v1/analysis'


def main():
    from scripts.colab_hfh.residual_band_archive import validate_trace
    from scripts.colab_hfh.constraint_archive import paired_result
    original=json.loads((OUT.parents[1]/'bands_v1_payload.json').read_text())['references']
    seeds=[];cases=[];exposures=[];spatial=[]
    for seed in range(3):
        with zipfile.ZipFile(OUT/f'nnunet_f0_s{seed}.zip') as z, TemporaryDirectory() as temporary:
            read=lambda name:json.loads(z.read(name))
            manifest=read('results/archive_manifest.json')
            for name,digest in manifest.items():
                assert Path(name).name==name
                assert hashlib.sha256(z.read('results/'+name)).hexdigest()==digest
            dest=Path(temporary)
            for name in z.namelist():
                if name.startswith('results/'):
                    assert len(Path(name).parts)==2
                    (dest/Path(name).name).write_bytes(z.read(name))
            p=read('protocol/protocol.json')
            for name,digest in p['residual_band_source_sha256'].items():
                assert hashlib.sha256(z.read('source/'+name)).hexdigest()==digest
            ref=read('protocol/baseline_references.json')[f'nnunet_f0_s{seed}']
            assert ref==original[f'nnunet_f0_s{seed}']
            pair=read('results/paired_comparison.json')
            assert paired_result(dest,ref,p)==pair
            assert validate_trace(dest,p)==read('results/residual_band_trace_audit.json')
            trace=[json.loads(line) for line in z.read('results/residual_band_trace.jsonl').decode().splitlines()]
            final=read('results/residual_band_spatial_final.json')['cases']
            names={r['annotation_sha256']:r['case_name'] for r in final if r['split']=='train'}
            active=[r for r in trace if r['selected_errors']]
            schedule=p['models']['nnunet']['objectives']['dice_bands']
            first=active[0]['charged_attempt']
            grad=read('results/gradient_strength_final.json')
            summary=dict(seed=seed,fold=0,baseline_dice_pct=100*pair['baseline_dice'],
                conditional_dice_pct=100*pair['constraint_dice'],delta_pp=pair['delta_dice_pp'],
                training_delta_pp=100*(pair['constraint_training_dice']-pair['baseline_training_dice']),
                baseline_fp=pair['baseline_foreground_fp'],conditional_fp=pair['constraint_foreground_fp'],
                baseline_fn=pair['baseline_foreground_fn'],conditional_fn=pair['constraint_foreground_fn'],
                fp_change=pair['constraint_foreground_fp']-pair['baseline_foreground_fp'],
                fn_change=pair['constraint_foreground_fn']-pair['baseline_foreground_fn'],
                first_active_update=first,active_updates=len(active),eligible_updates=len(trace),
                skipped_updates=[r['charged_attempt'] for r in trace if not r['selected_errors']],
                first_active_ramp=min(1.,(first-schedule['prefix_updates'])/schedule['constraint_warmup_updates']),
                active_error_coverage=sum(r['selected_errors'] for r in active)/sum(r['fp']+r['fn'] for r in active),
                coefficient=pair['weights']['bands'],
                envelope_gradient_ratio_anchor=grad['anchor']['bands'],
                envelope_gradient_ratio_final=grad['final']['bands'],
                gradient_caveat='Envelope gradient ratios, not actual gated residual gradient ratios',
                side_decisions={s:dict(Counter(r['sides'][s]['reason'] for r in trace)) for s in ('inner','outer')},
                side_selected_error_exposures={s:sum(r['sides'][s]['selected_errors'] for r in trace) for s in ('inner','outer')})
            seeds.append(summary)
            for row in trace:
                exposures.append(dict(seed=seed,update=row['charged_attempt'],case=names[row['annotation_sha256']],
                    dice=row['training_dice'],fp=row['fp'],fn=row['fn'],ready=row['ready'],
                    inner_errors=row['sides']['inner']['selected_errors'],outer_errors=row['sides']['outer']['selected_errors'],
                    inner_width=row['sides']['inner']['selected_width_mm'],outer_width=row['sides']['outer']['selected_width_mm'],
                    selected_errors=row['selected_errors'],coverage=row['total_error_coverage']))
            baseline={r['case_name']:r for r in ref['validation']['cases']}
            for row in read('results/validation_latest.json')['cases']:
                b=baseline[row['case_name']]
                cases.append(dict(seed=seed,fold=0,case=row['case_name'],baseline_dice_pct=100*b['macro_hard_dice'],
                    conditional_dice_pct=100*row['macro_hard_dice'],delta_pp=100*(row['macro_hard_dice']-b['macro_hard_dice']),
                    baseline_fp=b['foreground_fp'],conditional_fp=row['foreground_fp'],fp_change=row['foreground_fp']-b['foreground_fp'],
                    baseline_fn=b['foreground_fn'],conditional_fn=row['foreground_fn'],fn_change=row['foreground_fn']-b['foreground_fn']))
            for row in final:spatial.append(dict(row,seed=seed))
    final_support={}
    for split in ('train','validation'):
        group=[r for r in spatial if r['split']==split]
        final_support[split]=dict(cases=len(group),fp=sum(r['fp'] for r in group),fn=sum(r['fn'] for r in group),
            fn_inside_1mm=sum(r['sides']['inner']['candidates'][0]['covered_errors'] for r in group),
            fp_inside_1mm=sum(r['sides']['outer']['candidates'][0]['covered_errors'] for r in group),
            selected_fn=sum(r['sides']['inner']['selected_errors'] for r in group),
            selected_fp=sum(r['sides']['outer']['selected_errors'] for r in group),
            total_errors=sum(r['fp']+r['fn'] for r in group),selected_errors=sum(r['selected_errors'] for r in group))
    by_case=[]
    for name in sorted({r['case'] for r in cases}):
        group=[r for r in cases if r['case']==name]
        by_case.append(dict(case=name,mean_delta_pp=statistics.mean(r['delta_pp'] for r in group),
            fp_change=sum(r['fp_change'] for r in group),fn_change=sum(r['fn_change'] for r in group)))
    report=dict(archive_checks='All three manifests, frozen sources, original baseline references, matched pair checks and training-only trace checks passed',
        seeds=seeds,mean_baseline_pct=statistics.mean(r['baseline_dice_pct'] for r in seeds),
        mean_conditional_pct=statistics.mean(r['conditional_dice_pct'] for r in seeds),
        mean_delta_pp=statistics.mean(r['delta_pp'] for r in seeds),worst_delta_pp=min(r['delta_pp'] for r in seeds),
        fp_change=sum(r['fp_change'] for r in seeds),fn_change=sum(r['fn_change'] for r in seeds),
        training_mean_delta_pp=statistics.mean(r['training_delta_pp'] for r in seeds),
        improved_validation_crop_seed_pairs=sum(r['delta_pp']>0 for r in cases),validation_crop_seed_pairs=len(cases),
        final_support=final_support,by_case=by_case,
        limitations=['Error changes are net counts, not voxel-by-voxel correction/new-error transitions: paired baseline prediction maps were retired.',
            'Final spatial support describes the conditional model, not the baseline error pool.',
            'Training exposures count repeated appearances, not unique voxels.',
            'No actual gated gradient magnitude or parameter-gradient conflict was logged; envelope ratios cannot establish effective training strength.',
            'These are exploratory fold-0 results, not a full three-fold evaluation.'],
        interpretation=['Not mostly inactive: both sides active together on 129/135 eligible updates.',
            'Late readiness shortens the effective ramp in seed0: first nonzero auxiliary at update52 already uses7/9 ramp.',
            'All three train means improve slightly but all three validation means regress; boundary fitting did not generalize.',
            'Net behavior still favors foreground contraction; loss support coverage alone did not resolve FP/FN tradeoff.'],
        status='pilot_complete_no_expansion_three_model_matrix_resumed')
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    for filename,rows in [('case_comparison.csv',cases),('activation_trace.csv',exposures),('case_mean_comparison.csv',by_case)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
    for seed,a in enumerate(ax):
        rows=[r for r in exposures if r['seed']==seed];x=np.array([r['update'] for r in rows])
        inner=np.array([r['inner_errors'] for r in rows]);outer=np.array([r['outer_errors'] for r in rows])
        a.bar(x,inner,color='#2987b8',label='Selected missed foreground (FN)')
        a.bar(x,outer,bottom=inner,color='#db8843',label='Selected false positives (FP)')
        a.axvspan(45.5,54.5,color='grey',alpha=.1)
        for r in rows:
            if not r['ready']:a.axvspan(r['update']-.5,r['update']+.5,color='#cf3d4e',alpha=.22)
        a.set_ylabel(f'Seed {seed}\nError exposures');a.spines[['top','right']].set_visible(False)
        a.set_title(f"First active update {seeds[seed]['first_active_update']}; {seeds[seed]['active_updates']}/45 active; validation change {seeds[seed]['delta_pp']:+.3f} pp",loc='left',fontsize=10)
    ax[0].legend(loc='upper right',fontsize=8);ax[-1].set_xlabel('Optimizer update (native-only prefix: 1–45)')
    fig.suptitle('HFH nnU-Net conditional BAND: training-only activation\nGrey: scheduled ramp 46–54; red: readiness gate off',fontsize=12)
    fig.savefig(OUT/'activation.png',dpi=160);plt.close(fig)
    print(json.dumps({k:report[k] for k in ('mean_baseline_pct','mean_conditional_pct','mean_delta_pp','fp_change','fn_change','training_mean_delta_pp','final_support','status')},indent=2))


if __name__=='__main__':main()
