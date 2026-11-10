"""Summarize archived training-only logit probes without refitting on validation."""
from pathlib import Path
from statistics import mean, median
import json
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.shared43_study import NAME,ROOT


def main():
    report=ROOT/'reports'/NAME
    rows=[]
    for seed in range(3):
        for fold in range(3):
            identifier=f'MNI_unetr_vit_b16_dice_bands_f{fold}s{seed}'
            folder=report/('frozen/reused_runs' if fold==seed==0 else 'runs')/identifier
            verify_archive(folder,read(folder/'archive_manifest.json'))
            calibration=read(folder/'calibration.json')
            for quarter in (25,50,75,100):
                spatial=read(folder/f'spatial_{quarter:03d}.json')
                training=[r for r in spatial['cases'] if r['split']=='training']
                assert len(training)==20
                probes=[r['gradient_probe']['original_bands'] for r in training]
                row=dict(fold=fold,seed=seed,quarter=quarter,
                         coefficient=calibration['weights']['bands'],
                         anchor_weighted_median=calibration['statistics']['bands']['weighted_median_ratio'],
                         anchor_weighted_p95=calibration['statistics']['bands']['weighted_p95_ratio'],
                         training_median_weighted_logit_ratio=median(p['weighted_norm_ratio'] for p in probes),
                         fraction_band_gradient_energy_on_external_errors=mean(p['fraction_energy_on_external_errors'] for p in probes),
                         spatial_sha256=digest(folder/f'spatial_{quarter:03d}.json'),
                         archive_manifest_sha256=digest(folder/'archive_manifest.json'))
                for error in ('fn','fp','internal_swaps'):
                    total=sum(r['errors'][error]['total'] for r in training)
                    covered=sum(r['errors'][error]['original_band'] for r in training)
                    row[error]=dict(total=total,covered=covered,coverage=covered/total if total else None)
                rows.append(row)
    summaries=[]
    for quarter in (25,50,75,100):
        group=[r for r in rows if r['quarter']==quarter]
        summaries.append(dict(quarter=quarter,mean_training_median_weighted_logit_ratio=mean(r['training_median_weighted_logit_ratio'] for r in group),
            min_training_median_weighted_logit_ratio=min(r['training_median_weighted_logit_ratio'] for r in group),
            max_training_median_weighted_logit_ratio=max(r['training_median_weighted_logit_ratio'] for r in group),
            fraction_band_gradient_energy_on_external_errors=mean(r['fraction_band_gradient_energy_on_external_errors'] for r in group),
            coverage={k:sum(r[k]['covered'] for r in group)/sum(r[k]['total'] for r in group) for k in ('fn','fp','internal_swaps')}))
    result=dict(rows=rows,summary=summaries,curve_audit=read(report/'unetr_overfitting_audit_20261108.json'),
        source_sha256=digest(Path(__file__)),
        findings=['The coefficient is fixed after training-only calibration; its realized logit-gradient norm ratio can drift with training.',
                  'These are unaugmented evaluation-mode, detached-logit probes on training crops, not parameter gradients or Adam-preconditioned updates.',
                  'Union bands directly supervise foreground/background separation, not distinctions between foreground subfields; support coverage alone does not prove corrective swap gradients.',
                  'Stronger target tests must freshly calibrate on training crops and retain the75% p95 guard, fixed recipe, fixed splits and shared43T optimization cap.',
                  'Select targets provisionally on pilots; confirm any apparent gain on all nine matched fold/seed pairs and retain all targets tested.'])
    write(report/'unetr_gradient_strength_audit_20261108.json',result)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
