"""Mandatory loss-only counterfactual gradient gate on training cases only."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.depth_supervision.utils import atomic_json, digest
from .prior import LearnedLoss, Corruptions, load_training, one_hot


def counterfactual_probabilities(truth, wrong, mask, confidence):
    """Move mass only between the true and wrong class; leave the third fixed.

    In particular an A/P swap must preserve foreground probability exactly.
    Symmetric three-class smoothing would also remove foreground tissue.
    """
    y, w = one_hot(truth), one_hot(wrong)
    base = y * (1 - 2e-5) + (1-y) * 1e-5
    transferred = confidence * (1-1e-5) - 1e-5
    candidate = base + mask[:, None] * transferred * (w-y)
    return base, candidate


def probe(criterion, item, context, wrong, selected, kind, confidence, key, construction='isolated'):
    device = context.device
    labels = item['label'][None].to(device).long()
    truth = labels[:, 0]
    wrong = torch.as_tensor(wrong, device=device)[None]
    mask = torch.as_tensor(selected, device=device)[None]
    base, candidate = counterfactual_probabilities(truth, wrong, mask, confidence)
    if construction == 'uncertain':
        w = one_hot(wrong)
        candidate = torch.where(mask[:, None], w*confidence+(1-w)*((1-confidence)/2), base)
    with torch.no_grad():
        baseline = float(criterion.features_loss(base, labels, context, key))
        partial = float(criterion.features_loss(.5*base+.5*candidate, labels, context, key))
    logits = candidate.log().detach().requires_grad_(True)
    loss = criterion(logits, labels, context, key)
    gradient, = torch.autograd.grad(loss, logits)
    finite = bool(torch.isfinite(gradient).all())
    true_gradient = gradient.gather(1, labels)[:, 0]
    wrong_gradient = gradient.gather(1, wrong[:, None])[:, 0]
    helpful = float(((true_gradient-wrong_gradient)[mask] < 0).float().mean())
    # A bounded real gradient step, not a masked or projected correction.
    step = .25 / gradient.abs().max().clamp_min(1e-20)
    with torch.no_grad():
        updated = (logits - step * gradient).softmax(1)
        updated_loss = float(criterion.features_loss(updated, labels, context, key))
        true_delta = (updated.gather(1, labels)-candidate.gather(1, labels))[:, 0]
        gain = float(true_delta[mask].mean())
        fg_change = ((updated[:, 1:].sum(1)-candidate[:, 1:].sum(1)) * context[:, 1]).sum()
        corrected_mass = true_delta[mask].sum().clamp_min(1e-20)
        swap_ratio = float(fg_change.abs()/corrected_mass)
        retained = (truth > 0) & ~mask
        retained_change = float((updated[:, 1:].sum(1)-candidate[:, 1:].sum(1))[retained].sum())
    loss = float(loss.detach())
    return dict(kind=kind, construction=construction, confidence=confidence, voxels=int(mask.sum()), loss=loss, baseline_loss=baseline,
        partial_repair_loss=partial, updated_loss=updated_loss, helpful_fraction=helpful,
        mean_true_probability_gain=gain, net_foreground_mass_change=float(fg_change),
        retained_tissue_mass_change=retained_change, swap_foreground_to_correction_ratio=swap_ratio,
        finite=finite, detected=loss > baseline + 1e-12,
        repair_path_decreases=baseline < partial < loss,
        gradient_step_decreases=updated_loss < loss,
        mass_direction_ok=(float(fg_change)>0 if kind=='missing' else float(fg_change)<0 if kind=='extra' else swap_ratio < .5))


def summarize(rows, settings):
    groups = {}
    constructions = sorted({r.get('construction', 'isolated') for r in rows})
    for kind, construction in [(k,c) for k in ('missing','extra','swap') for c in constructions]:
        values = [r for r in rows if r['kind'] == kind and r.get('construction','isolated') == construction]
        total = sum(r['voxels'] for r in values)
        helpful = sum(r['helpful_fraction']*r['voxels'] for r in values)/total
        group = dict(probes=len(values), voxels=total, helpful_fraction=helpful,
            worst_probe_helpful_fraction=min(r['helpful_fraction'] for r in values),
            minimum_true_probability_gain=min(r['mean_true_probability_gain'] for r in values),
            detected_fraction=np.mean([r['detected'] for r in values]),
            repair_path_fraction=np.mean([r['repair_path_decreases'] for r in values]),
            gradient_descent_fraction=np.mean([r['gradient_step_decreases'] for r in values]),
            mass_direction_fraction=np.mean([r['mass_direction_ok'] for r in values]))
        group['passed'] = bool(all(r['finite'] for r in values) and group['detected_fraction']==1
            and helpful >= settings['helpful_fraction']
            and group['worst_probe_helpful_fraction'] >= settings['minimum_probe_helpful_fraction']
            and group['minimum_true_probability_gain'] > 0
            and group['repair_path_fraction'] >= settings['repair_path_fraction']
            and group['gradient_descent_fraction'] == 1
            and group['mass_direction_fraction'] >= settings['mass_direction_fraction'])
        groups[kind if len(constructions)==1 else kind+'_'+construction] = group
    return groups


def run_gate(root, protocol, device):
    root = Path(root)
    checkpoint = root/'encoder/frozen.pt'
    completion = root/'encoder/refinement_completion.json' if protocol.get('refinement') else root/'encoder/completion.json'
    if protocol.get('mass_refinement'):
        completion = root/'encoder/mass_refinement_completion.json'
    expected = json.loads(completion.read_text())['encoder_sha256']
    assert digest(checkpoint) == expected
    criterion = LearnedLoss(checkpoint, device)
    patients, contexts = load_training(root, protocol, device)
    rng = np.random.default_rng(protocol['gate']['seed'])
    rows = []
    for item in patients.items:
        context = contexts.get(item)
        maker = Corruptions(item, context)
        labels = item['label'][None].to(device).long()
        identical = criterion.features_loss(one_hot(labels[:, 0]), labels, context, item['case_name'])
        assert float(identical) == 0
        for kind in ('missing', 'extra', 'swap'):
            for c in (1, 2):
                for radius in protocol['gate']['radii_mm']:
                    wrong, selected = maker.make(kind, c, radius, rng)
                    for confidence in protocol['gate']['confidences']:
                        for construction in protocol['gate'].get('constructions',['isolated']):
                            row = probe(criterion, item, context, wrong, selected, kind, confidence, item['case_name'], construction)
                            row.update(case_name=item['case_name'], tissue_class=c, radius_mm=radius)
                            rows.append(row)
        print(json.dumps(dict(stage='gradient_gate', case_name=item['case_name'], probes=len(rows))), flush=True)
    groups = summarize(rows, protocol['gate'])
    assert all(p.grad is None and not p.requires_grad for p in criterion.parameters())
    assert digest(checkpoint) == expected
    value = dict(passed=all(x['passed'] for x in groups.values()), protocol_sha256=protocol['identity_sha'],
        encoder_sha256=expected, train_cases=protocol['splits']['5pct']['train'], validation_cases_used=[],
        loss_only=True, includes_dice_gradient=False, thresholds=protocol['gate'], groups=groups, probes=rows,
        interpretation='New synthetic corruptions on the same ten training anatomies; not held-out anatomy or a general guarantee.')
    atomic_json(root/'gradient_gate.json', value)
    print(json.dumps(dict(stage='gradient_gate_result', passed=value['passed'], groups=groups)), flush=True)
    if not value['passed']:
        raise RuntimeError('Learned-loss gradient gate failed; SwinUNETR training is prohibited.')
    return value


def verify_gate(root, protocol):
    root = Path(root)
    value = json.loads((root/'gradient_gate.json').read_text())
    acceptance = protocol.get('exploratory_audit_acceptance')
    assert value['loss_only'] and not value['includes_dice_gradient']
    expected_protocol = acceptance['audit_protocol_sha256'] if acceptance else protocol['identity_sha']
    assert value['protocol_sha256'] == expected_protocol
    assert value['encoder_sha256'] == digest(root/'encoder/frozen.pt')
    assert value['thresholds'] == protocol['gate'] and value['train_cases'] == protocol['splits']['5pct']['train']
    assert not value['validation_cases_used']
    recomputed = summarize(value['probes'], protocol['gate'])
    assert recomputed == value['groups']
    assert value['passed'] == all(g['passed'] for g in recomputed.values())
    expected_count = 10*3*2*len(protocol['gate']['radii_mm'])*len(protocol['gate']['confidences'])*len(protocol['gate'].get('constructions',['isolated']))
    assert len(value['probes']) == expected_count
    if acceptance:
        # Explicit exploratory authorization is bound to these exact evidence
        # bytes and encoder weights. It never changes the recorded audit result.
        assert acceptance['user_authorized'] is True and acceptance['scope']=='one_5pct_swin_run'
        assert digest(root/'gradient_gate.json') == acceptance['audit_sha256']
        assert value['encoder_sha256'] == acceptance['encoder_sha256']
        assert sorted(k for k,v in recomputed.items() if not v['passed']) == acceptance['accepted_failed_groups']
    else:
        assert value['passed']
    return value
