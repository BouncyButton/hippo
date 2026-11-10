"""Reconcile paired bands results and training fit across three datasets."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from scripts.mni_constraints.audit_bands import BASELINES, evaluation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/mni_constraint_root_cause_20261107'


def read(p): return json.loads(p.read_text())


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def verified(p):
    manifest = read(p.parent/'archive_manifest.json')
    assert sha(p) == manifest['metadata'][p.name]['sha256'], p
    return read(p)


def main():
    sources = {}
    prior = ROOT/'reports/thesis_noaug_update_20261021'
    audit = read(prior/'audit.json')
    rows = []
    for dataset, path in [('MSD', 'reports/msd_3fold_3seed_final_20261020/all_504_runs.csv'),
                          ('ADNI', 'reports/adni_colab_final_20261019/all_504_runs.csv')]:
        p = ROOT/path
        assert sha(p) == audit['sources_sha256'][path]
        sources[path] = sha(p)
        original = list(csv.DictReader(p.open()))
        for model in BASELINES:
            baseline = {(int(r['fold']),int(r['seed'])):r for r in original if r['model']==model and r['arm']=='dice'}
            bands = {(int(r['fold']),int(r['seed'])):r for r in original if r['model']==model and r['arm']=='dice_bands'}
            assert len(baseline)==len(bands)==9 and baseline.keys()==bands.keys()
            for key, b in baseline.items():
                c = bands[key]
                rows.append(dict(dataset=dataset, model=model, fold=key[0], seed=key[1],
                    baseline_train=float(b['training_dice']), bands_train=float(c['training_dice']),
                    baseline_val=float(b['dice']), bands_val=float(c['dice']),
                    baseline_external=float(b['fn'])+float(b['fp']), bands_external=float(c['fn'])+float(c['fp']),
                    baseline_internal=float(b['ap_swaps']) if dataset=='MSD' else 0,
                    bands_internal=float(c['ap_swaps']) if dataset=='MSD' else 0))
    for model, campaign in BASELINES.items():
        for fold in range(3):
            for seed in range(3):
                b = ROOT/'reports'/campaign/'runs'/f'MNI_{model}_f{fold}s{seed}'
                c = ROOT/'reports/mni_constraints14_cap8_20261107/runs'/f'MNI_{model}_dice_bands_f{fold}s{seed}'
                bt, ct = (verified(p/'training_final.json')['summary']['macro_hard_dice']*100 for p in (b,c))
                be, ce = evaluation(b), evaluation(c)
                for p in (b,c): sources[str(p.relative_to(ROOT))+'/archive_manifest.json'] = sha(p/'archive_manifest.json')
                rows.append(dict(dataset='MNI', model=model, fold=fold, seed=seed,
                    baseline_train=bt, bands_train=ct, baseline_val=100*be['macro_dice'],bands_val=100*ce['macro_dice'],
                    baseline_external=be['fn']+be['fp'],bands_external=ce['fn']+ce['fp'],
                    baseline_internal=be['swaps'],bands_internal=ce['swaps']))
    summaries = []
    for dataset in ('MSD','ADNI','MNI'):
        for model in BASELINES:
            rr = [r for r in rows if r['dataset']==dataset and r['model']==model]
            assert len(rr)==9
            gains = [r['bands_val']-r['baseline_val'] for r in rr]
            s = dict(dataset=dataset, model=model, runs=9,
                     **{key:mean(r[key] for r in rr) for key in ('baseline_train','bands_train','baseline_val','bands_val')},
                     val_gain_pp=mean(gains),train_gain_pp=mean(r['bands_train']-r['baseline_train'] for r in rr),
                     wins=sum(g>0 for g in gains),
                     fold_gains={str(f):mean(r['bands_val']-r['baseline_val'] for r in rr if r['fold']==f) for f in range(3)})
            for kind in ('external','internal'):
                a,b = (sum(r[arm+'_'+kind] for r in rr) for arm in ('baseline','bands'))
                s[kind] = dict(baseline=a,bands=b,delta=b-a,change_pct=100*(b/a-1) if a else None)
            summaries.append(s)
    OUT.mkdir(exist_ok=True)
    write = dict(status='verified', rows=rows, summaries=summaries, sources_sha256=sources,
                 interpretation_limits=['Datasets differ in labels, splits, preprocessing, losses, schedules and 1x versus8x cap: these are within-dataset paired comparisons, not an isolated dataset-effect ablation.',
                                        'Historical nnUNet used60% target; other historical models10%; MNI full9-run bands grid uses10% for every model. Current60% nnUNet is a separatefold0seed0 result.',
                                        'Three seeds within a fold are not nine independent test cohorts. All MNI folds were used for development.',
                                        'Training fit measures in-sample residual error, not a guarantee of validation generalization.'])
    (OUT/'cross_dataset_evidence.json').write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
