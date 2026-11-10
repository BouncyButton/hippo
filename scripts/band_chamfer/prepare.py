"""Freeze source, inherited data/optimizer policy, and the 64-arm matrix."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from .matrix import experiment_matrix

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/experiments/band_chamfer_20261008'
NAME = 'band_chamfer_fold0_seed0_5pct_20261008_01'
PREVIOUS = '/mnt/beegfsstudents/home/3160552/edge_ablation_fold0_seed0_5pct_20261008_01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    previous = ROOT/'docs/experiments/edge_ablation_20261008/protocol.json'
    old = json.loads(previous.read_text())
    p = copy.deepcopy(old)
    for key in ('source_sha256', 'identity_sha', 'loss_definitions', 'reduction_policy', 'calibration_policy',
                'geometry_policy', 'storage_budget'):
        p.pop(key, None)
    prefix = json.loads((ROOT/'docs/experiments/edge_ablation_20261008/recovery_verified_prefix.json').read_text())
    assert prefix['protocol_identity'] == old['identity_sha']
    p.update(name=NAME, main_runs=64, runs=experiment_matrix(), prior_protocol_sha256=sha(previous),
             predecessor_root=PREVIOUS, predecessor_job='690632', shared_warmup=prefix,
             minimum_free_gib=7.,
             estimated_additional_peak_gib=5.,
             geometry_policy='GT geometry only in losses. Original 45 arms use outer foreground; 19 added arms target the A/P transition. Verified 1 mm native RAS voxels. New geometry excludes padding; canonical bands preserve the prior implementation exactly.',
             chamfer_policy='Exact nearest-point distances on all linear p_foreground=0.5 axis-edge intersections. Coordinates retain gradients; crossing and nearest-neighbor assignments are discrete. Piecewise differentiable, not a mesh-area integral. All edges inside native FOV. Slope denominator floor 1e-6. Empty predicted/GT surfaces zero only the Chamfer term with telemetry; Dice and bands remain active. Calibration rejects a zero-gradient term.',
             calibration_policy='Reuse the verified epoch-5 Dice-only model/optimizer/scheduler/order/RNG. Each band definition gets median 0.1/p95 <=0.5 logit-gradient RMS relative to Dice. Pair/surface/AP/conditional-CE extras individually normalized, optional predeclared relative shares applied, then jointly rescaled to ONE median 0.1/p95 <=0.5 budget relative to Dice+bands. Fixed coefficients, ramp epochs 6-10. No validation tuning.',
             loss_definitions=dict(bands='Canonical baseline or explicitly declared morphology, physical shell, asymmetric support, side weighting, or soft-target replacements.',
                 pair='Coupled probability gap, independent endpoints, or relaxed gap hinge. Common support controls prevent distance/shift eligibility confounding.',
                 sdf='Outer-only side-balanced abs(p-y)*clipped absolute distance; equivalent to signed-distance probability error after side normalization. Distinct from prior classwise full-grid SDF.',
                 distance_brier='Outer-only side-balanced (p-y)^2 times clipped distance^power.',
                 chamfer='Direction-balanced point-surface distances in mm (linear), mm^2 (squared), 1 mm Huber, 0.5 mm dead zone, directional reweighting, or mean plus worst 10%.',
                 ap='q=sigmoid(z_anterior-z_posterior), interfaces only between adjacent GT foreground voxels. Actual mixed-slice labels retained. Surface/CE terms have zero direct background-logit gradient; no predicted-FG gate can hide swaps.',
                 ap_ce='Unmodified existing OriginalLabelAPConditionalCELoss for full-foreground semantic guard; original-label conditional CE within 3 mm for local control.',
                 ap_column='Chamfer independently on each X/Z column containing an actual GT Y-transition, averaging valid columns equally; missing predicted-column transitions are counted and rely on CE/Dice for recovery.',
                 ap_distribution='Posterior over all between-slice candidates spanning GT foreground: mean occupied-slice conditional log likelihood with posterior at low Y, anterior at high Y. Target is the linearly binned Y-distribution of actual annotated interface samples, retaining mixed slices. W1/CDF distance or expected absolute displacement. These are distribution-distance surrogates, not exact point Chamfer.'),
             storage_budget='64 best checkpoints; one active resumable checkpoint; one retained common warmup; compressed validation masks. Reuse dataset. Preserve all earlier results.',
             interpretation='Exploratory single fold/seed with 64 comparisons. Report every arm and paired deltas. Replication required before claiming improvement. No automatic best-only continuation.')
    files = {ROOT/name for name in old['source_sha256']}
    files.update(f for f in (ROOT/'scripts/band_chamfer').glob('*') if f.suffix in ('.py', '.sh'))
    sources = {}
    for file in sorted(files):
        if file.suffix == '.py':
            ast.parse(file.read_text(), str(file))
        sources[str(file.relative_to(ROOT))] = sha(file)
    p['source_sha256'] = sources
    audit_path = DOC/'ap_training_geometry_cluster.json'
    if audit_path.exists():
        p['ap_training_audit_sha256'] = sha(audit_path)
    p['identity_sha'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
    lines = ['# Band and Chamfer follow-up', '',
             '64 ordinary SwinUNETR runs; 5% (10 training cases), fold 0, seed 0; 52 validation cases.',
             'The same five-epoch Dice-only checkpoint initializes every arm, including two fresh controls.',
             '75 epochs maximum, unchanged optimizer, schedule, early stopping, and A/P Dice checkpoint selection.', '',
             '## Objective matrix', '', '| ID | Family | Comparison | Hypothesis |', '|---|---|---|---|']
    for entry in p['runs']:
        lines.append(f"| {entry['id']} | {entry['family']} | {entry['compare']} | {entry['idea']} |")
    lines += ['', '## Definitions and safeguards', '', p['geometry_policy'], '', p['chamfer_policy'], '',
              p['calibration_policy'], '',
              'Physical bands use Euclidean distance to the GT axis-edge midpoint samples. This is a discrete surface approximation; shell widths are in mm, not erosion iterations.', '',
              'Common3 pairs require offsets -1,0 inside and 1,2 outside at the same oriented crossing. Shift controls require -2,-1,0 inside and 1,2,3 outside. Missing long-distance support is reported, never invented.', '',
              'Chamfer variants supervise the p_anterior+p_posterior=0.5 outer surface. Final masks/metrics use the unchanged three-class argmax policy. The surface loss does not enforce topology or A/P anatomy. Grid-edge samples are not uniform surface-area samples; this bias is shared by the Chamfer comparisons.', '',
              'The empty-surface fallback is not assigned a geometric meaning: its contribution is zero and counted, while Dice/bands still train. Fixed-topology gradients cannot create a new surface by themselves.', '',
              '## A/P evidence and design', '',
              'Training-only inspection: 7/10 cases have exactly planar A/P labels; 3/10 have one mixed slice, with 39, 9, and 7 voxels disagreeing with the best plane. Affines are RAS, anterior high Y. Do not overwrite those mixed-slice labels with a fitted plane.', '',
              'The repository already implements conditional A/P CE and a strict planar cut posterior (thesis/new_constraints/ap_plane/conditional_ce.py; thesis/new_constraints/ap_cut/posterior.py). The former is reused as a control. The latter supplies the score construction, generalized here to the actual interface-distribution target because strict planar validity would exclude 3/10 training cases.', '',
              'Previous harmonic and graph-interface experiments did not pass their advancement criteria. This follow-up tests direct supervision and distance geometry, without assuming raw MRI gradients identify the anatomical landmark or adding a learned cut head.', '',
              'Chamfer alone cannot identify which side is anterior: the shape-only arm exposes that limitation; the other A/P Chamfer arms include original-label conditional CE. All A/P terms are training-only on GT foreground. No GT ROI, plane or cut is supplied to the inference model.', '',
              'Same-column matching prohibits lateral nearest-point shortcuts but cannot create a missing crossing; it logs missing columns. Cut-distribution objectives stay differentiable even without a hard A/P transition. Their targets describe the released annotation, not a verified uncal-apex landmark.', '',
              'Final raw-prediction diagnostics include A/P point-interface Chamfer and HD95, best-plane cut error (descriptive when labels are mixed), near/far 3 mm swaps, and undefined-interface counts. These do not select checkpoints or modify inference.', '',
              'Report A/P and foreground Dice, extra/missed/swapped voxels, mean surface distance, HD95, per-case differences, calibration, geometry counts, and runtime. All 64 arms remain visible.', '',
              p['interpretation'], '',
              'Sources: [Chamfer definition](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html), [regional signed-distance boundary loss](https://proceedings.mlr.press/v102/kervadec19a.html).', '',
              '## Storage', '', p['storage_budget'],
              f"Estimated additional peak: {p['estimated_additional_peak_gib']} GiB. Require {p['minimum_free_gib']} GiB free user quota before submission. No cleanup.", '']
    (DOC/'README.md').write_text('\n'.join(lines))
    return p


if __name__ == '__main__':
    p = build()
    print(json.dumps(dict(name=p['name'], runs=p['main_runs'], identity=p['identity_sha'])))
