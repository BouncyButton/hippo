"""Build the requested Markdown highlights from preserved, verified evidence."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path('/Users/filippofocaccia/Desktop/hippo/docs/experiments/tissue_profiles_20261007')
OUT = BASE / 'final_results'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    evidence = OUT / 'evidence'
    receipt = json.loads((OUT / 'collection_receipt.json').read_text())
    assert sha(OUT / 'evidence.tar.gz') == receipt['archive_sha256']
    manifest = json.loads((OUT / 'evidence_manifest.json').read_text())
    for item in manifest['files']:
        p = evidence / item['relative']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    v = json.loads((evidence / 'comparison.json').read_text())
    previous = json.loads((BASE / 'branch_review_20261008/remote_snapshot.json').read_text())
    assert v == previous['comparison'], 'Results changed since matched comparison analysis'
    rows = v['rows']
    assert len(rows) == 42 and len({(r['fraction'], r['arm'], r['objective']) for r in rows}) == 42
    shutil.copy2(BASE / 'branch_review_20261008/branch_comparisons.json', OUT / 'matched_comparisons.json')
    (OUT / 'all_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (OUT / 'all_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = {(r['fraction'], r['arm'], r['objective']): r for r in rows}
    families = [
        ('Ordinary SwinUNETR', 'native', 'native'),
        ('Direct MRI + position/depth/ROI inputs', 'informed', 'informed'),
        ('Plane branch + position channels', 'plane_position', 'plane_attention'),
        ('Unwrapped branch + position channels', 'unfold_position', 'unfold_attention'),
        ('Plane branch, MRI-only encoder', 'plane_mri_context', 'plane_mri_attention'),
        ('Unwrapped branch, MRI-only encoder', 'unfold_mri_context', 'unfold_mri_attention'),
        ('Native-view branch, MRI-only encoder', 'native_mri_context', 'native_mri_attention')]
    objectives = ['dice_baseline', 'dice_profile', 'dice_multiscale']
    text = '''# Tissue profiles and auxiliary-view controls: final results

Saved 8 October 2026. Slurm job **688313** completed successfully, exit 0, on 8 October at **01:53:06 Europe/Rome**, after **7 h 14 min 03 s**. All **36 new runs** and their preflight checks completed; six previously completed Dice controls give **42 comparison cells**. No new experiments were submitted for this report.

## Main findings

- **A separate, focused processing/fusion path helps in this experiment. Unwrapping has not been shown to be necessary.** With Dice alone, the native-view MRI branch achieved **81.9904% versus 74.1667%** at 5% data (+7.8236 percentage points), and **88.3102% versus 87.8514%** at full data (+0.4588 points). It sees the same MRI samples in their original voxel orientation, within the frozen guide-defined ROI; it is a small CNN, not a second SwinUNETR.
- **The alternative views do not clearly beat this native-view control under Dice alone.** MRI-only plane/unwrapped branches score 81.9168%/81.8895% at 5%, and 88.2627%/88.2786% at full data. Their paired case-bootstrap intervals versus the native-view branch include zero. This is absence of clear evidence of superiority, not a formal equivalence finding.
- **Explicit position/depth channels are not necessary for the large branch gain.** Directly adding anatomical channels to SwinUNETR improves Dice by 1.0800 points at 5% and 0.1121 at full data, much less than the native-view auxiliary. However, the MRI-only branches still use guide-derived ROI support and chart membership for pooling, mapping and fusion; anatomy has not been removed.
- **The profile constraints have architecture-dependent effects.** On ordinary SwinUNETR they improve 5% Dice to 75.5069% (pooled) and 75.5302% (multiscale), but only 87.8748% and 87.9159% with full data. They do not consistently improve the stronger branch models. On the native-view branch, pooled profiles reduce 5% Dice by 0.5343 points, while multiscale profiles add only 0.0625 points.
- **A higher Dice can conceal additional missing tissue.** At 5%, the native-view branch's Dice gain accompanies 48,674 fewer extra voxels, but 8,175 more missed voxels, aggregated across 52 validation cases. At full data, the same Dice-only comparison has 2,429 fewer missed voxels and 1,693 more extra voxels. These are net count differences, not exact recovered-voxel identities.

## Complete Dice matrix

All values are best-checkpoint validation anterior/posterior macro Dice, in percent. Pooled and multiscale columns mean Dice plus that profile term. At 5% the auxiliary has context fusion; at full data it additionally has attention. Cross-regime comparisons therefore also change branch architecture.

| Architecture | 5% Dice | 5% pooled | 5% multiscale | Full Dice | Full pooled | Full multiscale |
|---|---:|---:|---:|---:|---:|---:|
'''
    for label, low, full in families:
        values = [100 * index[(fraction, arm, objective)]['macro_dice'] for fraction, arm in [('5pct', low), ('full', full)] for objective in objectives]
        text += '| ' + label + ' | ' + ' | '.join(f'{x:.4f}' for x in values) + ' |\n'
    text += '''
The numerically best 5% cell is MRI-only unwrapped + multiscale, **82.0792%**, just **0.0263 points** above native-view + multiscale (**82.0528%**). At full data the best cell is MRI-only unwrapped + pooled (**88.3549%**), essentially tied numerically with native-view + multiscale (**88.3539%**); these two use different losses and are not an isolated view comparison. Selecting the best of many validation cells creates selection bias.

## Matched Dice-only evidence

Intervals below are 95% percentile intervals from 10,000 paired resamples of the 52 validation cases. They condition on the trained, validation-selected checkpoints; they do not measure training-seed uncertainty or adjust for multiple comparisons.

| Comparison | 5% delta in percentage points [95% interval] | Full delta in percentage points [95% interval] |
|---|---:|---:|
'''
    comparisons = {r['comparison']: r for r in json.loads((OUT / 'matched_comparisons.json').read_text())['comparisons']}
    for label, low, full in [
        ('Native-view MRI branch minus ordinary SwinUNETR', '5pct_none_native_aux_minus_ordinary', 'full_none_native_aux_minus_ordinary'),
        ('MRI-only plane minus native-view branch', '5pct_none_plane_mri_context_minus_native_aux', 'full_none_plane_mri_attention_minus_native_aux'),
        ('MRI-only unwrapped minus native-view branch', '5pct_none_unfold_mri_context_minus_native_aux', 'full_none_unfold_mri_attention_minus_native_aux'),
        ('Direct anatomical inputs minus ordinary SwinUNETR', '5pct_none_informed_minus_native', 'full_none_informed_minus_native')]:
        cells = []
        for key in (low, full):
            row = comparisons[key]
            a, b = row['case_bootstrap95_pp']
            cells.append(f"{row['dice_delta_pp']:+.4f} [{a:+.4f}, {b:+.4f}]")
        text += '| ' + label + ' | ' + ' | '.join(cells) + ' |\n'
    text += '''
There is a loss interaction: at 5% with pooled profiles, MRI-only unwrapped beats the corresponding native-view branch by **0.4868 points** (case-bootstrap interval **[0.2503, 0.7155]**). This partly reflects the native-view branch's deterioration under pooled profiles; it does not establish that unwrapping is generally superior. All matched comparisons are retained in `matched_comparisons.json` and the original case differences in `evidence/comparison.json`.

## What could become a learnable constraint?

The current evidence does **not** prove that the auxiliary MRI encoder is unnecessary. A previous, different checkpoint (5% unwrapped plus depth loss) retained 81.6623% Dice versus 81.8925% normally when mapped auxiliary features were zeroed at evaluation. This kept the ROI mask and learned fusion and changed inputs only after training. The corresponding full-data audit dropped from 88.3790% to 85.2813%, so the low-data result cannot be generalized to every setting. Neither audit replaces training a matched model without the auxiliary encoder.

The direct next hypothesis is a small **ROI-conditioned residual correction module**: reuse native SwinUNETR decoder features F and guide ROI mask m, learn a correction R(F,m), then classify F + R(F,m). It can use the MRI evidence already represented in F without reading the scan through another encoder. Train the whole network jointly with Dice first. This is a learned spatial prior/refinement architecture; it does not guarantee anatomical validity and is not automatically a new constraint loss.

To identify its cause, compare ordinary SwinUNETR, native-feature refinement without the ROI mask, the same refinement with the ROI mask, and the existing native-view MRI branch. Match training settings and initialization and repeat both fractions. A gain from refinement alone would support added decoder processing; a gain specifically from the mask would support spatial conditioning. These are proposals only, not submitted runs.

A training-only learned constraint is a further, separate option: train a compact anatomy encoder on **training labels only** to reconstruct clean masks from masks corrupted by additions, omissions and anterior/posterior swaps. Once this representation is learned, freeze it and penalize the difference between its representations of the model's soft probabilities and the target labels, alongside Dice. Frozen parameters still permit gradients back to the segmentation probabilities. A free penalty trained only to minimize its own value could collapse to zero, so it needs independent reconstruction/denoising supervision and freezing, or a carefully designed alternative training objective. At inference such a loss encoder can be discarded.

That learned representation could retain local arrangements that pooled profiles erase, but it can also miss small errors or overfit the ten training cases in the 5% condition. It must use the same allowed training subset; learning its prior from the full dataset would invalidate a 5% comparison. Validation labels remain evaluation targets and do not train the prior. Use actual MRI evidence and inspect FN/FP, A/P swaps and surfaces: a prior that merely shrinks predictions would repeat the existing low-data trade-off. No hard clipping to the guide ROI is proposed, because the guide itself can omit tissue.

Learned anatomy representations used as supervision have precedent in [Oktay et al., Anatomically Constrained Neural Networks](https://arxiv.org/abs/1705.08302). [Larrazabal et al., Post-DAE](https://arxiv.org/abs/2006.13791) provides a related denoising shape-prior precedent, but applies a post-processing reconstruction; it is not evidence that a training-only penalty or our proposed refinement will improve hippocampus segmentation.

## Protocol and limitations

- MSD Task04 hippocampus / Dataset101_MSD, fold 0, seed 0. Training: 10 of 208 cases (labelled 5%) or all 208; validation: 52 shared cases.
- Entire SwinUNETR and any auxiliary/fusion components train jointly from scratch. Feature size 24, padded native 64³ MRI, batch size 1, no augmentation or pretraining, CUDA bf16. Inputs are hippocampus crops, not whole brains.
- AdamW, learning rate 1e-4, weight decay 1e-5; StepLR every 20 epochs by 0.5. Maximum 75 epochs; early stopping at minimum 25 epochs, patience 8, minimum improvement 0.0005. The highest validation macro Dice checkpoint is reported.
- Dice includes background. No depth balancing, SDF, focal weighting, entropy gating, topology or component-count penalty is used in this campaign. Profile coefficients use training-only gradient calibration after five Dice epochs and ramp during epochs 6–10.
- Guide-derived frames and ROI support are frozen. Training guides were out-of-fold. MRI-only auxiliary branches still depend on the guide ROI plus 8 mm and fixed chart/context groups. These guide models add cost beyond the final segmentor.
- The native-view MRI auxiliary adds 20,984 parameters at 5% (about 0.134% of the 15,703,029-parameter baseline). Parameter count alone therefore does not establish the cause; fusion placement, convolutional processing, ROI conditioning and normalization/context are possible explanations.
- Alternative-view controls match trainable weights and parameter counts, but resampling, grid shape, effective physical convolution scale, padding and support layout differ. The comparison is between processing pipelines, not an isolated mathematical coordinate change.
- This is one fold and one seed, with fixed guides and a 75-epoch schedule. The earlier three-seed plane/unwrapped study did not replicate this new native-view control. Case bootstrap intervals are not seed replication. The 5% baseline reached the maximum 75 epochs while branch models often stopped earlier; conclusions apply to this schedule and do not establish converged-model superiority.
- All comparisons use validation data already used for checkpoint/model selection. Generalization, anatomical guarantees and clinical usefulness are not established. Aggregate error differences cannot identify exact corrected voxels. Original runtime counters differ for reused controls, so speed claims need separate benchmarking.

## Full metric appendix

FN = missed foreground; FP = extra foreground; A/P = anterior/posterior label swaps. Error counts are pooled over 52 validation cases. MSD and HD95 are mean case-level surface metrics in millimetres. Epochs are best/last. Full precision, class Dice, foreground Dice, parameter counts and runtimes are preserved in `all_results.csv` and `all_results.json`.

| Data | Arm | Loss | Dice % | FN | FP | A/P | MSD mm | HD95 mm | Epochs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for fraction in ('5pct', 'full'):
        for _, low, full in families:
            for objective in objectives:
                r = index[(fraction, low if fraction == '5pct' else full, objective)]
                text += f"| {fraction} | {r['arm']} | {objective.replace('dice_', '')} | {100*r['macro_dice']:.4f} | {r['missed']:,} | {r['extra']:,} | {r['swaps']:,} | {r['surface_distance_mean_mm']:.4f} | {r['hd95_mm']:.4f} | {r['best_epoch']}/{r['epochs']} |\n"
    text += '''
## Preservation and cleanup

All 180 recorded artifact hashes (five artifacts for each of the 36 new runs, including checkpoints and prediction arrays) were checked before cleanup. The local evidence archive retains the small per-case results, training histories, configurations, completion records, diagnostics, logs, source snapshots and deployment/protocol metadata. It excludes model weights and prediction/geometry arrays, which the user authorized deleting. Reused controls and their per-case comparison vectors are included in the comparison evidence; their original records also remain in the previously saved unfolding-study archive.

`evidence_manifest.json` lists every archived file and SHA-256. `collection_receipt.json`, `verification.json`, `preservation_receipt.json`, `cleanup_plan.json` and `cleanup_receipt.json` record verification and cleanup. The exact deletion scope is generated outputs inside `/mnt/beegfsstudents/home/3160552/tissue_profiles_fold0_seed0_20261007_01` only. Source, deployment records, the shared atlas target, raw data, unrelated campaigns and the local report/evidence are preserved. Model checkpoints deleted from this job cannot be used for further inference audits without retraining or another independent copy.

Cleanup outcome is recorded below after deletion.
'''
    (OUT / 'RESULTS.md').write_text(text)
    assert text.count('| 5pct |') == 21 and text.count('| full |') == 21
    assert len(json.loads((OUT / 'verification.json').read_text())['artifact_hashes_verified']) == 180
    shutil.copy2(Path(__file__), OUT / 'build_highlights.py')
    preserved = {'local_archive_verified': True, 'all_evidence_hashes_verified': True, 'report_verified': True,
        'archive_sha256': receipt['archive_sha256'], 'report_sha256': sha(OUT / 'RESULTS.md'),
        'cleanup_plan_sha256': sha(OUT / 'cleanup_plan.json'), 'conditions': 42,
        'report_path': str(OUT / 'RESULTS.md'), 'evidence_files': len(manifest['files'])}
    (OUT / 'preservation_receipt.json').write_text(json.dumps(preserved, indent=2) + '\n')
    print(json.dumps(preserved, indent=2))


if __name__ == '__main__':
    main()
