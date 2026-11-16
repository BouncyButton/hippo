"""Audit HFH nnU-Net behavior against saved MSD, ADNI and MNI evidence."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics as stats
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/hfh_colab_exploration_20261111'
OUT = EXPORT/'nnunet_diagnosis'


def read(path):
    return json.loads(path.read_text())


def hfh_baselines(base):
    curves, timing, models, cases = [], [], [], []
    inventory = {r['case']:r for r in read(ROOT/'reports/hfh_dataset_20261111/inventory.json')['cases']}
    for model in base['rates']:
        refs = [(key,r) for key,r in base['references'].items() if key.startswith(model+'_f')]
        train=stats.mean(r['training']['summary']['macro_hard_dice'] for _,r in refs)*100
        validation=stats.mean(r['result']['validation_dice'] for _,r in refs)*100
        tp=fp=fn=0
        for key,ref in refs:
            for c in ref['validation']['cases']:
                pc=c['per_class']['hippocampus']; tp+=pc['tp'];fp+=pc['fp'];fn+=pc['fn']
                if model=='nnunet':
                    subject=c['case_name'].rsplit('_h',1)[0]
                    cases.append(dict(fold=ref['result']['fold'],seed=ref['result']['seed'],
                        case=c['case_name'],subject=subject,dice_pct=c['macro_hard_dice']*100,
                        tp=pc['tp'],fp=pc['fp'],fn=pc['fn'],volume_voxels=pc['tp']+pc['fn'],
                        original_matrix=inventory[subject]['shape'][0]))
            if model!='nnunet':continue
            path=EXPORT/'recipe_analysis'/f'{key}.zip'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['archive_sha256']
            with zipfile.ZipFile(path) as z:
                for name,digest in json.loads(z.read('results/archive_manifest.json')).items():
                    assert hashlib.sha256(z.read('results/'+name)).hexdigest()==digest
                assert json.loads(z.read('results/result.json'))==ref['result']
                curve=list(csv.DictReader(io.StringIO(z.read('results/metrics.csv').decode())))
            curve=[{k:float(v) for k,v in row.items()} for row in curve]
            f,s=ref['result']['fold'],ref['result']['seed']
            curves.append(dict(fold=f,seed=s,points=curve))
            timing.append(dict(fold=f,seed=s,
                first_logged_ge1_update=next(r['step'] for r in curve if r['macro_hard_dice']>=.01),
                first_logged_ge50_update=next(r['step'] for r in curve if r['macro_hard_dice']>=.5),
                dice_at_update22_pct=next(r['macro_hard_dice']*100 for r in curve if r['step']==22),
                best_minus_final_pp=100*(max(r['macro_hard_dice'] for r in curve)-curve[-1]['macro_hard_dice'])))
        models.append(dict(model=model,training_dice_pct=train,validation_dice_pct=validation,
            gap_pp=train-validation,precision_pct=100*tp/(tp+fp),recall_pct=100*tp/(tp+fn),
            predicted_to_true_volume=(tp+fp)/(tp+fn),false_positives=fp,false_negatives=fn,
            native_loss=base['protocols'][model]['protocol.json']['training_loss'],
            training_soft_dice_pct=100*stats.mean(r['training']['summary']['macro_soft_dice'] for _,r in refs)))
    return models,curves,timing,cases


def historical_comparison():
    rows=[]
    for dataset,file,arm_key in (
        ('MSD','reports/msd_3fold_3seed_final_20261020/all_56_condition_summaries.csv','condition'),
        ('ADNI','reports/adni_colab_final_20261019/condition_summary.csv','arm')):
        data=list(csv.DictReader((ROOT/file).open()))
        baseline=next(r for r in data if r['model']=='nnunet' and r[arm_key]=='dice')
        band=next(r for r in data if r['model']=='nnunet' and r[arm_key]=='dice_bands')
        rows.append(dict(dataset=dataset,training_dice_pct=float(baseline['training_dice']),
            validation_dice_pct=float(baseline['validation_dice']),band_gain_pp=float(band['gain_pp']),
            n=9,source=file,source_sha256=hashlib.sha256((ROOT/file).read_bytes()).hexdigest(),
            verification='Recomputed from historical audited summary; not a fresh audit of all historical raw archives'))
    mni_file=ROOT/'reports/mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109/cross_model_bands_ca4_review/analysis.json'
    mni=[r for r in read(mni_file)['rows'] if r['model']=='nnunet']
    assert len(mni)==9 and len({(r['fold'],r['seed']) for r in mni})==9
    training=[]
    for r in mni:
        for label in ('baseline','bands'):
            folder=Path(r[label+'_archive']);manifest=read(folder/'archive_manifest.json')
            assert hashlib.sha256((folder/'archive_manifest.json').read_bytes()).hexdigest()==r[label+'_archive_sha256']
            for name in ('validation_latest.json','training_final.json','config.json','metrics.csv','budget_ledger.json'):
                assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==manifest['metadata'][name]['sha256']
            score=read(folder/'validation_latest.json')['summary']['macro_hard_dice']*100
            assert abs(score-r[label])<1e-10
            if label=='baseline':training.append(read(folder/'training_final.json')['summary']['macro_hard_dice']*100)
    rows.append(dict(dataset='MNI repaired 53.3T',training_dice_pct=stats.mean(training),
        validation_dice_pct=stats.mean(r['baseline'] for r in mni),
        band_gain_pp=stats.mean(r['macro_gain'] for r in mni),
        minimum_band_gain_pp=min(r['macro_gain'] for r in mni),positive_pairs=sum(r['macro_gain']>0 for r in mni),
        n=9,source=str(mni_file.relative_to(ROOT)),verification='Nine paired baseline/BAND archive manifests and metric bytes reverified'))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    base=read(EXPORT/'bands_v1_payload.json')
    models,curves,timing,cases=hfh_baselines(base)
    cross=historical_comparison()
    selected=read(EXPORT/'bands_analysis/selected_recipes/nnunet_r12/verified_pairs.json')
    nn=next(r for r in models if r['model']=='nnunet')
    cross.append(dict(dataset='HFH selected BAND r12',training_dice_pct=nn['training_dice_pct'],
        validation_dice_pct=nn['validation_dice_pct'],band_gain_pp=stats.mean(r['delta_dice_pp'] for r in selected),
        n=9,source='reports/hfh_colab_exploration_20261111/bands_analysis/selected_recipes/nnunet_r12/verified_pairs.json',
        verification='Previously independently verified HFH pairs; new candidates still running'))
    combination=read(EXPORT/'constraints_analysis/nnunet_band_sdf_all_nine.json')
    drift={}
    for term in ('bands','sdf','joint'):
        ds=combination['gradient_diagnostics']
        drift[term]=dict(anchor_median=stats.median(d['anchor'][term]['median'] for d in ds),
            final_median=stats.median(d['final'][term]['median'] for d in ds),
            median_retained_fraction=stats.median(d['final'][term]['median']/d['anchor'][term]['median'] for d in ds))
    subjects=[]
    for subject in sorted({r['subject'] for r in cases}):
        group=[r for r in cases if r['subject']==subject]
        subjects.append(dict(subject=subject,n_observations=len(group),mean_dice_pct=stats.mean(r['dice_pct'] for r in group),
            mean_volume_voxels=stats.mean(r['volume_voxels'] for r in group),original_matrix=group[0]['original_matrix']))
    by_matrix={m:dict(subjects=sum(r['original_matrix']==m for r in subjects),
        mean_subject_dice_pct=stats.mean(r['mean_dice_pct'] for r in subjects if r['original_matrix']==m)) for m in (256,512)}
    analysis=dict(scope='HFH nnU-Net diagnosis with historical cross-dataset context; exploratory, not a causal cross-dataset ablation',
        historical_comparison=cross,hfh_model_errors=models,hfh_baseline_learning_timing=timing,
        median_first_logged_ge50_update=stats.median(r['first_logged_ge50_update'] for r in timing),
        hfH_band_sdf_all_nine={k:v for k,v in combination.items() if k not in ('results','gradient_diagnostics')},
        hfh_band_sdf_gradient_drift=drift,subject_diagnostics=subjects,original_matrix_groups=by_matrix,
        subject_volume_dice_correlation=float(np.corrcoef([r['mean_volume_voxels'] for r in subjects],
            [r['mean_dice_pct'] for r in subjects])[0,1]),
        native_recipe_comparison=dict(HFH=dict(initial_lr=.1,lr_horizon=90,updates=90,training_crops=10,classes=2),
            ADNI=dict(initial_lr=.01,lr_horizon=174,updates=90,training_crops=10,classes=2),
            MSD=dict(initial_lr=.01,lr_horizon=174,updates=90,training_crops=10,classes=3),
            MNI_repaired=dict(initial_lr=.02,lr_horizon=200,policy='hold_half',updates=150,training_crops=20,classes=4)),
        distinctions=['HFH and ADNI are binary foreground Dice; MSD and MNI use different foreground subfield macro metrics.',
            'HFH calibration ratios measure full-resolution logit gradients, not parameter updates or SGD momentum.',
            'A zero/near-zero validation hard Dice does not prove that feature learning or soft predictions are absent.',
            'MNI historical8x memorization and repaired53.3T schedule are distinct campaigns; do not pool them.',
            'Subject matrix grouping is descriptive with 15 validation subjects, not proof of scanner causality.',
            'All gains must remain paired to unchanged controls; changing native LR or loss requires new matched controls.'])
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    with (OUT/'hfh_case_errors.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=cases[0]);writer.writeheader();writer.writerows(cases)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,gr)=plt.subplots(1,2,figsize=(13,4.9),gridspec_kw={'width_ratios':[1.35,1]})
    for c in curves:
        ax.plot([r['step'] for r in c['points']],[100*r['macro_hard_dice'] for r in c['points']],
                color=['#2166ac','#e08214','#4d9221'][c['seed']],alpha=.65,lw=1.6)
    ax.axvline(22,color='#555',ls='--',lw=1.3,label='Calibration/onset in tested recipe')
    ax.axhline(50,color='#999',ls=':',lw=1)
    ax.set(xlabel='Optimizer updates (90 total)',ylabel='Validation foreground Dice (%)',
           title='HFH nnU-Net: foreground emerges late',ylim=(0,90),xlim=(0,92))
    ax.legend(loc='lower right',fontsize=9)
    x=np.arange(3);width=.34
    gr.bar(x-width/2,[drift[t]['anchor_median'] for t in ('bands','sdf','joint')],width,label='Calibration at update22',color='#4477aa')
    gr.bar(x+width/2,[drift[t]['final_median'] for t in ('bands','sdf','joint')],width,label='After update90',color='#cc6677')
    gr.set(xticks=x,xticklabels=['BAND','SDF','Combined'],ylabel='Auxiliary / native logit-gradient norm',
           title='Fixed weights lose their original balance',ylim=(0,.35))
    gr.legend(fontsize=9)
    fig.text(.5,.015,'Nine fixed seed/fold pairs. Right: median of per-run training-case medians; BAND+SDF r04. These probes do not measure optimizer-space gradients.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(OUT/'nnunet_learning_and_gradient_drift.png',dpi=180);plt.close(fig)
    print(json.dumps({k:analysis[k] for k in ('historical_comparison','median_first_logged_ge50_update','hfh_band_sdf_gradient_drift','original_matrix_groups','subject_volume_dice_correlation')},indent=2))


if __name__=='__main__':main()
