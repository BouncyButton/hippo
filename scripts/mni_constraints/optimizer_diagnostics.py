"""Read-only loss/gradient/update probes for the frozen nnUNet SGD run."""
from contextlib import nullcontext
import time
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from scripts.cross_dataset_campaign.common import read, write

PROBE_STEPS = (1, 8, 16, 32, 48, 64, 80, 96, 104, 112, 121, 128, 136, 144, 150)


def dot(a, b):
    terms = [(x.detach().float() * y.detach().float()).sum() for x, y in zip(a, b) if x is not None and y is not None]
    return float(torch.stack(terms).sum()) if terms else 0.


def norm(a):
    return max(0., dot(a, a)) ** .5


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return dot(a, b) / denominator if denominator else None


def components(outputs, labels, wrapper, need_class=False):
    """Use the actual native loss modules and their active supervision weights."""
    native = wrapper.loss
    assert native.ignore_label is None and native.weight_ce == native.weight_dice == 1
    assert native.dc.batch_dice and not native.dc.do_bg and not native.dc.ddp
    dice, ce, ca4_dice, ca4_ce, heads = [], [], [], [], []
    for output, weight in zip(outputs, wrapper.weight_factors):
        if not weight:
            continue
        target = F.interpolate(labels.float(), size=output.shape[2:], mode='nearest').long()
        dc = native.dc(output, target)
        cross = native.ce(output, target[:, 0])
        dice.append(weight * dc)
        ce.append(weight * cross)
        heads.append(dict(size=list(output.shape[2:]), weight=float(weight), dice=float(dc.detach()), ce=float(cross.detach())))
        if need_class:
            # Reproduce MemoryEfficientSoftDiceLoss's per-class arithmetic.
            probability = native.dc.apply_nonlin(output)[:, 1:]
            onehot = torch.zeros_like(output, dtype=torch.bool).scatter_(1, target, 1)[:, 1:]
            axes = tuple(range(2, output.ndim))
            intersection = (probability * onehot).sum(axes).sum(0)
            total_pred = probability.sum(axes).sum(0)
            total_true = onehot.sum(axes).sum(0)
            per_class = -(2 * intersection + native.dc.smooth) / (total_true + total_pred + native.dc.smooth).clamp_min(1e-8)
            assert torch.allclose(per_class.mean(), dc, atol=2e-6, rtol=2e-6)
            ca4_dice.append(weight * per_class[2] / 3)
            voxel_ce = F.cross_entropy(output, target[:, 0], reduction='none')
            ca4_ce.append(weight * (voxel_ce * (target[:, 0] == 3)).sum() / target.numel())
    result = dict(dice=sum(dice), ce=sum(ce), heads=heads)
    if need_class:
        result.update(ca4_dice=sum(ca4_dice), ca4_ce=sum(ca4_ce))
    return result


class OptimizerDiagnostics:
    def __init__(self, model, optimizer, output, probe_steps=PROBE_STEPS):
        assert isinstance(optimizer, torch.optim.SGD) and len(optimizer.param_groups) == 1
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named]
        self.optimizer = optimizer
        self.path = output / 'optimizer_diagnostics.json'
        self.rows = read(self.path)['rows'] if self.path.exists() else []
        self.probe_steps = set(probe_steps)
        self.parameter_grads = {}
        self.before = None

    def observe(self, outputs, labels, criterion, loss, step, case_name, amp):
        started = time.perf_counter()
        self.parameter_grads = {}
        self.row = dict(step=step, case_name=case_name, parameter_probe=step in self.probe_steps)
        rng_before = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if amp else None
        assert all(p.grad is None for p in self.params)
        with FlopCounterMode(display=False) as counter:
            with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
                terms = components(outputs, labels, criterion, self.row['parameter_probe'])
            reconstructed = float((terms['dice'] + terms['ce']).detach())
            discrepancy = abs(reconstructed - float(loss.detach()))
            assert discrepancy < 2e-5, discrepancy
            self.row.update(loss=float(loss.detach()), dice_loss=float(terms['dice'].detach()),
                            ce_loss=float(terms['ce'].detach()), decomposition_error=discrepancy, supervision=terms['heads'])
            logit_grads = {}
            for name in ('dice', 'ce'):
                logit_grads[name] = torch.autograd.grad(terms[name], outputs[0], retain_graph=True)[0].detach().float()
            z = outputs[0].detach().float()
            probability = z.softmax(1)
            truth = labels[:, 0]
            prediction = z.argmax(1)
            self.row['training_batch'] = dict(class_counts=[int((truth == c).sum()) for c in range(4)],
                predicted_counts=[int((prediction == c).sum()) for c in range(4)],
                mean_CA4_probability_on_true_CA4=float(probability[:, 3][truth == 3].mean()),
                correct_CA4_voxels=int(((truth == 3) & (prediction == 3)).sum()))
            self.row['full_resolution_logit_gradients'] = dict(
                dice_norm=norm([logit_grads['dice']]), ce_norm=norm([logit_grads['ce']]),
                cosine=cosine([logit_grads['dice']], [logit_grads['ce']]),
                CA4_channel_by_true_class={str(c): {name: dict(
                    mean=float(g[:, 3][truth == c].mean()), sum=float(g[:, 3][truth == c].sum()))
                    for name, g in logit_grads.items()} for c in range(4)})
            if self.row['parameter_probe']:
                for name in ('dice', 'ce', 'ca4_dice', 'ca4_ce'):
                    self.parameter_grads[name] = tuple(g.detach() if g is not None else None for g in
                        torch.autograd.grad(terms[name], self.params, retain_graph=True, allow_unused=True))
                g = self.parameter_grads
                dn, cn = norm(g['dice']), norm(g['ce'])
                self.row['parameter_gradients'] = dict(norms={k: norm(v) for k, v in g.items()},
                    ce_over_dice_norm=cn / dn if dn else None,
                    dice_ce_cosine=cosine(g['dice'], g['ce']),
                    ca4_dice_ce_cosine=cosine(g['ca4_dice'], g['ce']),
                    ca4_ce_total_dice_cosine=cosine(g['ca4_ce'], g['dice']),
                    full_resolution_CA4_head_bias={name: float(grad[3]) for name, grads in g.items()
                        for (parameter_name, _), grad in zip(self.named, grads)
                        if parameter_name.endswith('seg_layers.0.bias') and grad is not None})
        assert all(p.grad is None for p in self.params), 'Diagnostic backward populated training .grad'
        assert torch.equal(torch.get_rng_state(), rng_before), 'Diagnostic altered CPU RNG'
        if amp:
            assert torch.equal(torch.cuda.get_rng_state(), cuda_rng), 'Diagnostic altered CUDA RNG'
        self.row.update(diagnostic_profiled_flops=counter.get_total_flops(),
                        diagnostic_seconds=time.perf_counter() - started,
                        extra_logit_gradient_calls=2,
                        extra_parameter_gradient_calls=4 if self.row['parameter_probe'] else 0)

    def before_step(self, preclip_norm, lr):
        self.row.update(learning_rate=lr, preclip_gradient_norm=float(preclip_norm),
                        clip_threshold=12., clipping_factor=min(1., 12. / (float(preclip_norm) + 1e-6)))
        self.before = [p.detach().clone() for p in self.params]
        self.clipped = [p.grad.detach().clone() if p.grad is not None else None for p in self.params]
        self.row['postclip_gradient_norm'] = norm(self.clipped)
        if self.parameter_grads:
            combined = [None if a is None and b is None else (a if b is None else b if a is None else a + b)
                        for a, b in zip(self.parameter_grads['dice'], self.parameter_grads['ce'])]
            # BF16 branches can round separately, so decomposition is diagnostic, not substituted into training.
            self.row['decomposed_vs_actual_gradient_cosine'] = cosine(combined, self.clipped)

    def after_step(self):
        started = time.perf_counter()
        group = self.optimizer.param_groups[0]
        delta = [p.detach() - old for p, old in zip(self.params, self.before)]
        momentum = [self.optimizer.state[p].get('momentum_buffer') for p in self.params]
        negative_gradient = [None if g is None else -g for g in self.clipped]
        plain = [None if g is None else -group['lr'] * (g + group['weight_decay'] * old)
                 for g, old in zip(self.clipped, self.before)]
        self.row['optimizer'] = dict(update_norm=norm(delta), parameter_norm_before=norm(self.before),
            momentum_buffer_norm=norm(momentum),
            update_vs_negative_current_gradient_cosine=cosine(delta, negative_gradient),
            update_vs_no_momentum_direction_cosine=cosine(delta, plain),
            update_to_parameter_norm=norm(delta) / max(norm(self.before), 1e-30),
            momentum=group['momentum'], weight_decay=group['weight_decay'], nesterov=group['nesterov'])
        if self.parameter_grads:
            self.row['first_order_loss_changes'] = {name: dict(
                actual_sgd_update=dot(gradient, delta),
                hypothetical_no_momentum_update=dot(gradient, plain),
                actual_update_vs_negative_gradient_cosine=cosine(delta, [None if g is None else -g for g in gradient]))
                for name, gradient in self.parameter_grads.items()}
        self.row['diagnostic_seconds'] += time.perf_counter() - started
        self.rows = [r for r in self.rows if r['step'] < self.row['step']] + [self.row]
        write(self.path, dict(status='recording', optimizer_updates_added=0,
            scope='Same training forwards, actual native loss modules, extra autograd.grad probes without .grad or RNG mutation; parameter updates are measured after the original SGD step.',
            compute_accounting='Optimization budget unchanged. Extra probe FLOPs use PyTorch operator counters; unsupported elementwise reductions and update logging are not fully counted. Extra gradient calls and diagnostic wall seconds are also reported.',
            direction_note='First-order directional estimates on the current training batch, not outcomes of an alternative optimizer run. Negative means predicted reduction in that loss component.',
            probe_steps=sorted(self.probe_steps), rows=self.rows,
            total_diagnostic_profiled_flops=sum(r['diagnostic_profiled_flops'] for r in self.rows),
            total_diagnostic_seconds=sum(r['diagnostic_seconds'] for r in self.rows)))
        self.before, self.clipped, self.parameter_grads = None, None, {}


def instrument_source(source):
    replacements = [
        ('    native_criterion = criterion if args.model == "nnunet" else None\n',
         '    native_criterion = criterion if args.model == "nnunet" else None\n'
         '    from scripts.mni_constraints.optimizer_diagnostics import OptimizerDiagnostics\n'
         '    diagnostics = OptimizerDiagnostics(model, optimizer, args.output)\n'),
        ('        loss.backward()\n',
         '        diagnostics.observe(logits, label, native_criterion, loss, step + 1, batch["case_name"], amp)\n'
         '        loss.backward()\n'),
        ('            torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)\n'
         '        optimizer.step()\n',
         '            preclip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)\n'
         '        diagnostics.before_step(preclip_norm, lr)\n'
         '        optimizer.step()\n'
         '        diagnostics.after_step()\n'),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source
