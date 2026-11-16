"""Prepare the follow-on matrix without presenting hypotheses as run recipes."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / 'reports/hfh_colab_exploration_20261111'


def main():
    source = ROOT / 'scripts/adni_constraints/objectives.py'
    tree = ast.parse(source.read_text())
    arms = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in node.targets))
    base = json.loads((EXPORT / 'bands_v1_payload.json').read_text())
    models = list(base['protocols'])
    from scripts.colab_hfh.summarize_strict_bands import verify
    from scripts.colab_hfh.bands_strict_search import qualifies
    verified_bands = {}
    verified_band_recipes = {}
    selected = {
        'basic_unet16': sorted((EXPORT/'bands_analysis/selected_recipes/basic_unet16_r09').glob('*.zip')),
        'unetr_vit_b16': sorted((EXPORT/'bands_analysis/current').glob('unetr_vit_b16_*.zip')),
    }
    for model, paths in selected.items():
        results = [verify(path, base['references']) for path in paths]
        if not qualifies(results):
            raise ValueError('Previously qualified BAND recipe no longer verifies: ' + model)
        with zipfile.ZipFile(paths[0]) as archive:
            protocol = json.loads(archive.read('protocol/protocol.json'))
        schedule = protocol['models'][protocol['trainer_alias']]['objectives']['dice_bands']
        recipe_name = 'basic_unet16_r09' if model == 'basic_unet16' else 'bands_v1'
        verified_band_recipes[model] = dict(
            recipe=recipe_name, native_prefix=schedule['prefix_updates'],
            ramp=protocol['constraint_warmup_updates'], updates=schedule['max_steps'],
            target=protocol['calibration_target_ratio'], p95_cap=protocol['calibration_p95_cap'],
            inner_weight=protocol.get('band_inner_weight', .5),
            minimum_delta_pp=min(row['delta_dice_pp'] for row in results),
            mean_delta_pp=sum(row['delta_dice_pp'] for row in results)/9,
            transfer_policy='Starting hypothesis for other losses; requires fresh objective-specific calibration and cost validation.')
        for row in results:
            row['recipe'] = recipe_name
        verified_bands.update({(model, row['fold'], row['seed']): row for row in results})
    cost_references = {}
    for model in models:
        profile_path = ROOT / 'reports/adni_constraints_5pct_fold0_seed0_20261013/profiles' / (model + '.json')
        measured = json.loads(profile_path.read_text())
        protocol = base['protocols'][model]['protocol.json']
        native = protocol['models'][protocol['trainer_alias']]['objectives']['dice']
        if measured['costs']['dice']['update_flops'] != native['flops_per_update']:
            raise ValueError('Native arithmetic differs from the candidate profile')
        cost_references[model] = dict(
            status='reference_arithmetic_only_final_recipe_audit_pending',
            profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            native_update_flops=native['flops_per_update'],
            native_updates=native['max_steps'],
            canonical_costs={arm: measured['costs'][arm] for arm in arms})
    lessons = {
        'bands': {
            'supported': 'BasicUNet and UNETR have nine-pair recipes exceeding +1 pp in every pair. nnU-Net remains unresolved; complete Swin BAND and several combinations meet the revised mean target. See live matrix for current evidence.',
            'transfer': 'Retain a verified model-specific BAND recipe as the initial component; recalibrate inside each combination.',
            'avoid': 'Do not use a different recipe per validation pair, or substitute partial means for nine-pair evidence.'},
        'edge': {
            'supported': 'No HFH edge matrix has been completed in this campaign.',
            'transfer': 'Test delayed gradual activation; measure gradient alignment with BAND because both supervise boundaries.',
            'avoid': 'Do not add a full independent gradient allowance on top of BAND without checking their combined magnitude.'},
        'sdf': {
            'supported': 'No HFH SDF matrix has been completed in this campaign.',
            'transfer': 'Start with the established delayed-SDF schedule after the early constraint ramp; calibrate the actual SDF loss.',
            'avoid': 'Do not copy BAND side weights, focal exponents or one-step width into a distance-field loss.'},
        'depth': {
            'supported': 'No HFH depth matrix has been completed in this campaign.',
            'transfer': 'Validate normalized geometry weights and calibrate depth against the combined active objective, using training crops only.',
            'avoid': 'Do not describe voxel-space depth as millimetres; do not silently change geometry between control and comparison.'},
    }
    rows = []
    for model, arm, fold, seed in itertools.product(models, arms, range(3), range(3)):
        key = f'{model}_f{fold}_s{seed}'
        rows.append(dict(model=model, arm=arm, fold=fold, seed=seed,
                         baseline_archive_sha256=base['references'][key]['archive_sha256'],
                         components=[c for c in lessons if c in arm],
                         status='verified_control_reference' if arm == 'dice' else 'pending_recipe_and_cost_validation',
                         runnable=False))
        if arm == 'dice_bands' and (model, fold, seed) in verified_bands:
            rows[-1].update(status='verified_existing_result',
                            result=verified_bands[(model, fold, seed)])
    plan = dict(
        status='live_pilot_verified_screening_ready' if (EXPORT/'constraints_analysis/pilot_verified.json').exists() else 'runtime_prepared_live_pilot_pending', conditions=len(rows), models=models,
        arms=list(arms), seeds=[0, 1, 2], folds=[0, 1, 2],
        interpretation='Exploratory validation-selected comparison; independent evaluation required for generalization claims.',
        scope='The established 14-arm project matrix, not the full 16-subset factorial.',
        mean_followup_policy=dict(collect_all_nine=True,minimum_mean_delta_pp=1.,
            individual_regressions_allowed=True,freeze_qualified_arms=True,
            interpretation='Complete partial candidates; no per-pair improvement rejection.'),
        nnunet_diagnosis_followup=dict(candidate_count=14, objectives=13,
            reason='Missed foreground, late hard foreground learning, early-calibration gradient drift; MNI demonstrates schedule sensitivity and weak marginal gains after baseline repair.',
            evidence='nnunet_diagnosis/analysis.json',
            candidate_audit='constraints_analysis/nnunet_diagnosis_candidates.json',
            selection='Finish active work; prioritize fixed diagnosis-led candidates; report all nine and freeze only mean-qualified arms.',
            invariants='Append-only IDs;90updates; unchanged native recipes, references and optimization cap.',
            interpretation='Validation-informed exploratory hypotheses, not a proven causal repair.'),
        baseline_count=36, constraint_conditions=468,
        reusable_verified_band_conditions=len(verified_bands),
        verified_band_recipes=verified_band_recipes,
        remaining_constraint_conditions=468-len(verified_bands),
        budget_flops_per_run=31976450667090, budget_scope='optimization_only',
        cost_references=cost_references,
        geometry=dict(conventional_array_spacing_mm=[1, 2, 1],
                      unpadded_shape=[64, 32, 64], padded_shape=[64, 64, 64],
                      bands='Retain declared morphological voxel steps for comparability.',
                      sdf_depth='Physical millimetres: anisotropic EDT clipped/normalized at8mm; depth bins2mm and ROI8mm; source-axis projected plane, no anatomical laterality claim.',
                      convention='Discrete half-minimum-spacing EDT estimate, not exact continuous surface distance.',
                      implementation='scripts/colab_hfh/constraint_geometry.py',
                      implementation_sha256=hashlib.sha256((ROOT/'scripts/colab_hfh/constraint_geometry.py').read_bytes()).hexdigest(),
                      training_audit='constraint_geometry_training_audit.json',
                      training_audit_sha256=hashlib.sha256((EXPORT/'constraint_geometry_training_audit.json').read_bytes()).hexdigest(),
                      source='scripts/colab_hfh/prepare.py'),
        calibration='Training-only; accounted separately. Record per-term and joint gradient ratios, caps and alignment.',
        parameter_gradient_diagnostic=dict(model='nnunet',pairs=[dict(fold=0,seed=1),dict(fold=2,seed=1)],
            timing='After final training and validation evaluation; no optimization update or coefficient change.',
            sampling='First crop from each of the first two distinct training subjects; no validation input.',
            measurements='Native total, each active native deep-supervision head, and the actual weighted auxiliary objective in trainable-parameter space.',
            accounting='Separate post-training allowance:2cases*(3activeheads+2)*native_update_flops =3552203953190 FLOPs per probed run.',
            interpretation='Evaluation-mode gradients, not SGD momentum or effective optimizer updates. Existing completed runs are not rerun.'),
        online_sdf_pilot=dict(models=['swinunetr'],objectives=['dice_sdf'],targets=[.1,.3],
            reason='Measured final SDF gradient strength fell to 2-21% of its calibrated ratio in SwinUNETR pilots.',
            rule='Per-training-batch detached FP32 native/SDF logit-gradient RMS ratio, coefficient ceiling64x training anchor, unchanged native loss and delayed SDF ramp.',
            budget='Conservative detached-probe arithmetic charged inside optimization cap; audited per-update trace.',
            arithmetic_audit_sha256=hashlib.sha256((EXPORT/'online_sdf_arithmetic_audit.json').read_bytes()).hexdigest(),
            status='Separate exploratory candidates; no performance claim'),
        calibration_implementation=dict(
            method='joint_logit_gradient_all_ramp_subsets_v1',
            note='Cap the per-case worst subset before taking median/p95, covering independently ramped components and delayed SDF.',
            numeric_audit='constraint_loss_numeric_audit.json',
            source_sha256={name: hashlib.sha256((ROOT/'scripts/colab_hfh'/name).read_bytes()).hexdigest()
                           for name in ('constraint_calibration.py','constraint_geometry.py','constraint_schedule.py',
                                        'constraint_worker.py','constraint_archive.py','constraint_campaign.py','constraint_search.py','constraint_mean_search.py','constraint_drive.py','constraint_diagnostics.py','parameter_gradient_diagnostics.py','constraint_online_sdf.py','bands_variants.py')}),
        combination_policy='Recalibrate the actual combined objective. Audit its joint gradient against the native loss; do not multiply the standalone gradient allowance by the number of components. Final joint targets are not yet selected.',
        recipe_policy='One declared recipe per model/arm across all nine pairs. Never choose a separate recipe for each validation pair.',
        matrix_completion_policy='After selecting and freezing each final arm recipe, complete all nine pairs and retain regressions. Early-rejected screening trials are reported separately, not substituted for a complete final matrix.',
        qualification='Mean paired improvement >=1.0 Dice percentage point across all nine fixed pairs, all Dice >=0.50; allow small individual regressions and report all nine plus worst case.',
        active_selection_policy='Freeze every qualifying model/objective; explore unresolved nnU-Net and SwinUNETR arms only, nnU-Net first.',
        reuse='Reuse verified results only when protocol, source, precision, initialization, split and endpoint match exactly.',
        launch_gates=['Resolve current BAND consistency or explicitly record failure.',
                      'Profile/audit every active objective combination and calibration cost.',
                      'Freeze schedules and exact update counts under the cap; no invented counts copied from BAND.',
                      'Preserve native optimizer/loss/LR unless a separately labelled ablation and matched control are supplied.',
                      'Validate HFH geometry and objective gradients, including padding and SDF timing.',
                      'Verify complete archives before removing completed bulky artifacts.'],
        lessons=lessons, matrix=rows,
        objective_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert len(rows) == 504 and len({(r['model'], r['arm'], r['fold'], r['seed']) for r in rows}) == 504
    path = EXPORT / 'constraint_matrix_plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), conditions=len(rows), status=plan['status'])))


if __name__ == '__main__':
    main()
