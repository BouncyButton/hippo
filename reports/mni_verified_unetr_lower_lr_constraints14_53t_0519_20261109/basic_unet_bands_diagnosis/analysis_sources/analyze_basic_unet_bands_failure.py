"""Read-only analysis of verified Basic UNet bands/baseline archives."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = 'mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109'
REPORT = ROOT / 'reports' / NAME
STAGE = ROOT / 'experiments' / NAME
METRICS = ('macro_hard_dice', 'CA1_3_hard_dice', 'subiculum_hard_dice',
           'CA4_DG_hard_dice', 'whole_hippocampus_hard_dice')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confusion(cases):
    matrix = [[sum(c['confusion'][i][j] for c in cases) for j in range(4)] for i in range(4)]
    truth = sum(matrix[3])
    return dict(matrix=matrix, ca4_recall=matrix[3][3]/truth,
                ca4_prediction_truth_ratio=sum(row[3] for row in matrix)/truth,
                ca4_to_ca1=matrix[3][1], ca4_to_subiculum=matrix[3][2],
                ca4_to_background=matrix[3][0], ca4_true=truth,
                union_fp=sum(matrix[0][1:]), union_fn=sum(row[0] for row in matrix[1:]),
                internal_swaps=sum(matrix[i][j] for i in range(1,4) for j in range(1,4) if i != j))


def archive(path):
    from scripts.cross_dataset_campaign.storage import verify_archive
    manifest=read(path/'archive_manifest.json')
    verify_archive(path, manifest)
    ack=read(path/'archive_acknowledgment.json')
    assert ack['status']=='verified' and ack['manifest_sha256']==digest(path/'archive_manifest.json')
    done=read(path/'completion.json')
    assert done['status']=='complete' and done['discarded_attempts']==0
    cfg=read(path/'config.json')
    assert done['consumed_training_flops']<=cfg['budget_flops']==53305109592450
    result=dict(archive=str(path), archive_sha256=digest(path/'archive_manifest.json'),
                config=cfg, completion=done, curves=list(csv.DictReader((path/'metrics.csv').open())))
    for key, filename in [('validation','validation_latest.json'),('training','training_final.json')]:
        data=read(path/filename)
        result[key]=dict(summary=data['summary'], errors=confusion(data['cases']),
                         case_dice={c['case_name']:c['macro_hard_dice'] for c in data['cases']})
    return result


def main():
    pairs=[]
    for seed in (0,5,19):
        for fold in range(3):
            base=REPORT/'runs'/f'MNI_basic_unet16_dice_f{fold}s{seed}'
            if not base.exists(): base=STAGE/'reused_runs'/base.name
            band=REPORT/'runs'/f'MNI_basic_unet16_dice_bands_f{fold}s{seed}'
            if not (band/'archive_acknowledgment.json').exists(): continue
            b,c=archive(base),archive(band)
            for k in ('seed','fold','train_cases','validation_cases','dataset_sha256','initial_lr','augmentation','precision','shared_source_sha256','objective_source_sha256'):
                assert b['config'][k]==c['config'][k], k
            calibration=read(band/'calibration.json')
            assert calibration['config_sha256']==digest(band/'config.json')
            assert set(x['case_name'] for x in calibration['cases'])==set(c['config']['train_cases'])
            result=dict(seed=seed,fold=fold,baseline=b,bands=c,calibration={k:calibration[k] for k in
                ('weights','statistics','target_ratio','p95_cap','anchor_successful_updates','anchor_training_attempts')})
            cosines=[x['gradient_pair_means']['base:bands']/math.sqrt(x['gradient_pair_means']['base:base']*x['gradient_pair_means']['bands:bands']) for x in calibration['cases']]
            result['calibration']['logit_cosine_median']=statistics.median(cosines)
            result['deltas']={split:{k:100*(c[split]['summary'][k]-b[split]['summary'][k]) for k in METRICS} for split in ('validation','training')}
            result['first_logged_metrics_identical']=b['curves'][0]==c['curves'][0]
            result['ca4_validation_threshold_updates']={arm:{str(threshold):next(
                (int(x['step']) for x in values['curves'] if int(x['step'])>=94
                 and 100*float(x['CA4_DG_hard_dice'])>=threshold),None)
                for threshold in (5,10,20,40)} for arm,values in [('baseline',b),('bands',c)]}
            result['validation_cases_worse']=sum(c['validation']['case_dice'][k]<value
                                                 for k,value in b['validation']['case_dice'].items())
            result['spatial']=[]
            for quarter in (25,50,75,100):
                spatial=read(band/f'spatial_{quarter:03}.json')
                for split in ('training','validation'):
                    cases=[x for x in spatial['cases'] if x['split']==split]
                    if not cases: continue
                    probes=[x['gradient_probe']['original_bands'] for x in cases if 'gradient_probe' in x]
                    ratios=[x['weighted_norm_ratio'] for x in probes if x['weighted_norm_ratio'] is not None]
                    result['spatial'].append(dict(quarter=quarter,split=split,step=spatial['charged_updates'],
                        weighted_logit_ratio_median=statistics.median(ratios) if ratios else None,
                        weighted_logit_ratio_max=max(ratios) if ratios else None,
                        swaps=sum(x['errors']['internal_swaps']['total'] for x in cases),
                        swaps_in_original_band=sum(x['errors']['internal_swaps']['original_band'] for x in cases),
                        fn=sum(x['errors']['fn']['total'] for x in cases),fp=sum(x['errors']['fp']['total'] for x in cases)))
            pairs.append(result)
    for fold in range(3):
        group=[p for p in pairs if p['fold']==fold]
        for pair in group:
            for key in ('train_cases','validation_cases'):
                assert pair['baseline']['config'][key]==group[0]['baseline']['config'][key]
    out=REPORT/'basic_unet_bands_diagnosis';out.mkdir(exist_ok=True)
    payload=dict(source_sha256=digest(Path(__file__)),pairs=pairs,
                 scope='Archived matched final-budget outcomes and diagnostic logit gradients. These do not establish parameter-gradient or Adam-update causality. No live experiment modified.')
    (out/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    import matplotlib.pyplot as plt
    selected=next(p for p in pairs if p['seed']==5 and p['fold']==1)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True)
    for ax,metric,title in zip(axes,('macro_hard_dice','CA4_DG_hard_dice','whole_hippocampus_hard_dice'),
                               ('Mean subfield Dice','CA4/DG Dice','Whole-hippocampus Dice')):
        for arm,color,label in [('baseline','#2368a2','Baseline'),('bands','#d66320','Bands')]:
            data=selected[arm]['curves']
            for prefix,style,split in [('', '-', 'validation'),('training_', '--', 'training')]:
                ax.plot([int(x['step']) for x in data],[100*float(x[prefix+metric]) for x in data],
                        color=color,ls=style,lw=2,label=f'{label}, {split}')
        ax.axvspan(76,152,color='grey',alpha=.13)
        ax.set(title=title,xlabel='Optimizer updates',ylim=(0,100));ax.grid(alpha=.2)
    axes[0].set_ylabel('Dice (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',frameon=False)
    fig.suptitle('Basic UNet, seed 5 / fold 1: boundary gains hide delayed CA4/DG learning\nGrey region: bands ramp after training-only calibration at update 76',fontsize=12)
    fig.tight_layout(rect=(0,.09,1,.9))
    fig.savefig(out/'seed5_fold1_curves.png',dpi=170)
    fig.savefig(out/'seed5_fold1_curves.pdf')
    plt.close(fig)
    for p in pairs:
        print(json.dumps(dict(seed=p['seed'],fold=p['fold'],deltas=p['deltas'],
            validation={arm:{'dice':{k:100*p[arm]['validation']['summary'][k] for k in METRICS},'errors':p[arm]['validation']['errors']} for arm in ('baseline','bands')},
            calibration=p['calibration'],late_gradients=p['spatial'][-2:])))


if __name__=='__main__':main()
