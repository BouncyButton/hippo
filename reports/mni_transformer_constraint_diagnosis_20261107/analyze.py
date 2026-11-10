"""Read-only analysis of archived MNI runs and historical comparison evidence."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STAGE = ROOT / 'reports/mni_shared_cap2_baseline_bands_f0s0_20261107'
SOURCES = {}


def read(path):
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def evaluate(data):
    cases = data['cases']
    matrix = [[sum(c['confusion'][i][j] for c in cases)
               for j in range(4)] for i in range(4)]
    fn = sum(matrix[i][0] for i in range(1, 4))
    fp = sum(matrix[0][1:])
    swaps = sum(matrix[i][j] for i in range(1, 4)
                for j in range(1, 4) if i != j)
    return dict(summary=data['summary'], confusion=matrix, fn=fn, fp=fp,
                swaps=swaps, internal_error_pct=100 * swaps / (fn + fp + swaps),
                fp_pct_of_external=100 * fp / (fn + fp))


def main():
    results = {}
    verified = 0
    for model in ('swinunetr', 'unetr_vit_b16', 'nnunet', 'basic_unet16'):
        pairs = {}
        for arm in ('dice', 'dice_bands'):
            run = STAGE / 'runs' / f'MNI_{model}_{arm}_f0s0'
            manifest = read(run / 'archive_manifest.json')
            for name, expected in manifest['metadata'].items():
                payload = (run / name).read_bytes()
                assert len(payload) == expected['bytes'], (run, name, 'size')
                assert hashlib.sha256(payload).hexdigest() == expected['sha256'], (run, name, 'hash')
                verified += 1
            value = dict(validation=evaluate(read(run / 'validation_latest.json')),
                         training=evaluate(read(run / 'training_final.json')),
                         config=read(run / 'config.json'),
                         recipe=read(run / 'runtime_recipe.json'))
            if arm == 'dice_bands':
                cal = read(run / 'calibration.json')
                cosines = []
                for case in cal['cases']:
                    g = case['gradient_pair_means']
                    cosines.append(g['base:bands'] / math.sqrt(g['base:base'] * g['bands:bands']))
                value['calibration'] = dict(weights=cal['weights'], statistics=cal['statistics'],
                    anchor_updates=cal['anchor_successful_updates'],
                    logit_cosine_median=statistics.median(cosines),
                    logit_cosine_range=[min(cosines), max(cosines)],
                    limitation='Full-volume logit gradients at one early anchor; not parameter/update gradients or later behavior.')
            pairs[arm] = value
        base = pairs['dice']['validation']
        bands = pairs['dice_bands']['validation']
        pairs['delta'] = dict(
            dice_pp={key: 100 * (bands['summary'][key] - base['summary'][key])
                     for key in base['summary'] if key.endswith('hard_dice')},
            errors={key: bands[key] - base[key] for key in ('fn', 'fp', 'swaps')})
        results[model] = pairs
    historical = read(ROOT / 'reports/mni_constraint_root_cause_20261107/cross_dataset_evidence.json')
    historical_summary = []
    for ds in ('MSD', 'ADNI', 'MNI'):
        for model in ('swinunetr', 'unetr_vit_b16'):
            rows = [r for r in historical['rows'] if r['dataset'] == ds and r['model'] == model]
            ext = sum(r['baseline_external'] for r in rows)
            internal = sum(r['baseline_internal'] for r in rows)
            historical_summary.append(dict(dataset=ds, model=model, runs=len(rows),
                baseline_train=statistics.mean(r['baseline_train'] for r in rows),
                baseline_val=statistics.mean(r['baseline_val'] for r in rows),
                bands_val_gain_pp=statistics.mean(r['bands_val']-r['baseline_val'] for r in rows),
                internal_error_pct=100*internal/(ext+internal)))
    audit = read(ROOT / 'reports/mni_bands_implementation_audit_20261107/audit.json')
    protocols = {}
    fields = ('initial_lrs', 'lr_horizon_updates', 'training_loss', 'learning_rate_policy',
              'augmentation', 'training_volumes_per_fold', 'validation_volumes_per_fold',
              'train_count', 'validation_count', 'calibration_target_ratio', 'num_classes',
              'patient_identity_scope', 'constraint_geometry', 'budget_flops')
    for ds in ('msd', 'adni'):
        for model in ('swinunetr', 'unetr_vit_b16'):
            path = ROOT / f'experiments/{ds}_mixed_losses_nnunet60_others10_20261014/protocols/{model}/protocol.json'
            protocol = read(path)
            protocols[f'{ds}_{model}'] = {k: protocol[k] for k in fields if k in protocol}
    evidence = dict(status='archive_verified', metadata_files_verified=verified,
        current_scope='MNI 2x historical optimization FLOP cap, fold 0 seed 0, paired fresh runs',
        current=results, historical=historical_summary,
        historical_errors=audit['historical_baseline_errors'], historical_protocols=protocols,
        sources_sha256=SOURCES,
        limitations=[
            'Current MNI pilot is one fold and seed; historical comparisons have nine runs per model.',
            'External means foreground/background confusion, not measured distance to a boundary.',
            'Voxel error shares are not additive contributions to macro Dice.',
            'No spatial predictions/checkpoints retained; prior spatial probes used different interrupted 8x runs.',
            'No new training, validation tuning, or causal ablation performed by this analysis.',
            'MNI is the evaluated new dataset here; these results do not establish causes for COBRA.'])
    (OUT / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(dict(metadata_files_verified=verified,
        deltas={m:p['delta'] for m,p in results.items()},
        historical=historical_summary), indent=2))


if __name__ == '__main__':
    main()
