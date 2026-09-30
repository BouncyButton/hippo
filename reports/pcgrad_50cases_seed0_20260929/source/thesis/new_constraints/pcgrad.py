"""Symmetric PCGrad (Yu et al., NeurIPS 2020), summed weighted task gradients.

Paper Algorithm 1: each task is projected against the ORIGINAL other task
gradients in a freshly randomized order. Sequential projections need not leave
every final pair nonconflicting. No gradient normalization or norm restoration.
"""
from __future__ import annotations

import torch

TASKS = ('dice', 'bands', 'inner', 'outer', 'cross')


def gram_matrix(gradients: torch.Tensor, chunk_size=1 << 20):
    """FP64 accumulation, bounded temporary memory; return a small CPU matrix."""
    if gradients.ndim != 2 or not torch.isfinite(gradients).all():
        raise ValueError('Expected a finite [tasks, parameters] matrix.')
    gram = torch.zeros((len(gradients), len(gradients)), dtype=torch.float64, device=gradients.device)
    for start in range(0, gradients.shape[1], chunk_size):
        block = gradients[:, start:start+chunk_size].double()
        gram.add_(block @ block.T)
    return gram.cpu()


def project(gradients: torch.Tensor, generator: torch.Generator, enabled=True):
    """Return summed projected gradient and complete task-space diagnostics.

    A coefficient basis implements exactly the same sequential projection as
    full-vector subtraction, using only a small Gram matrix for dot products.
    The projection generator is independent of model/data RNG streams.
    """
    gram = gram_matrix(gradients)
    n = len(gradients)
    coefficients = torch.eye(n, dtype=torch.float64)
    orders, projections = [], []
    for i in range(n):
        order = [j for j in torch.randperm(n, generator=generator).tolist() if j != i]
        orders.append(order)
        for j in order:
            dot, norm2 = float(coefficients[i] @ gram[:, j]), float(gram[j, j])
            if enabled and dot < 0 and norm2 > 0:
                coefficients[i, j] -= dot / norm2
                projections.append([i, j])
    # Elementwise accumulation avoids TF32 matmul policy affecting projection.
    combined = (coefficients.sum(0).to(device=gradients.device, dtype=gradients.dtype)[:, None]
                * gradients).sum(0) if enabled else gradients.sum(0)
    if not torch.isfinite(combined).all():
        raise FloatingPointError('PCGrad produced a nonfinite combined gradient.')
    original = torch.ones(n, dtype=torch.float64)
    weights = coefficients.sum(0)
    before2 = float(original @ gram @ original)
    after2 = max(float(weights @ gram @ weights), 0.)
    delta = weights - original
    change2 = max(float(delta @ gram @ delta), 0.)
    diagnostics = dict(task_gram=gram.tolist(), projected_task_gram=(coefficients @ gram @ coefficients.T).tolist(),
        coefficients=coefficients.tolist(), orders=orders, projections=projections,
        projection_count=len(projections), original_norm=max(before2,0.)**.5,
        projected_norm=after2**.5, change_norm=change2**.5,
        projected_over_original_norm=(after2/before2)**.5 if before2>0 else None,
        cosine_with_original=float(weights @ gram @ original)/(after2*before2)**.5 if after2>0 and before2>0 else None)
    return combined, diagnostics


def backward_tasks(losses, parameters, scaler, generator, *, project_conflicts):
    """Collect AMP-scaled derivatives, project unscaled FP32 values, restore scale.

    The caller invokes scaler.step/update once. Any nonfinite component sets a
    nonfinite parameter gradient so GradScaler skips the entire optimizer update.
    Parameters unused by every task keep grad=None (including AdamW decay behavior).
    """
    params = tuple(parameters)
    scale = float(scaler.get_scale())
    vectors, used = [], [False] * len(params)
    for i, loss in enumerate(losses):
        grads = torch.autograd.grad(scaler.scale(loss), params, retain_graph=i < len(losses)-1, allow_unused=True)
        for j, g in enumerate(grads):
            used[j] |= g is not None
        vectors.append(torch.cat([(g.detach().float()/scale if g is not None else torch.zeros_like(p)).reshape(-1)
                                  for p,g in zip(params,grads)]))
    gradients = torch.stack(vectors)
    if not torch.isfinite(gradients).all():
        if not scaler.is_enabled():
            raise FloatingPointError('Nonfinite task gradients without AMP scaling.')
        for p, active in zip(params, used):
            p.grad = torch.full_like(p, float('nan')) if active else None
        return dict(finite=False, optimizer_step_skipped=True, amp_scale=scale)
    combined, report = project(gradients, generator, enabled=project_conflicts)
    offset = 0
    for p, active in zip(params, used):
        end = offset + p.numel()
        p.grad = combined[offset:end].reshape_as(p).to(p.dtype).mul(scale) if active else None
        offset = end
    report.update(finite=True, optimizer_step_skipped=False, amp_scale=scale)
    return report
