"""Offline candidates only: never imported by any published Colab controller.

Neither candidate has passed A100 throughput/equivalence validation. These
functions deliberately do not patch the frozen engine or write campaign state.
"""
from collections import defaultdict
from contextlib import nullcontext

import torch


def packed_cpu_snapshot(state):
    """Copy dense tensors in device/dtype groups, retaining values and aliases.

    This is synchronous. Optimizer state_dict contains live tensor references,
    so sending that dict straight to a background writer would be unsafe.
    """
    tensors = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or not value.is_contiguous():
                raise ValueError('Candidate supports only contiguous dense checkpoint tensors.')
            tensors.setdefault(id(value), value)
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)

    visit(state)
    storage_owners = {}
    for ident, tensor in tensors.items():
        if tensor.numel():
            key = (tensor.device, tensor.untyped_storage().data_ptr())
            if key in storage_owners and storage_owners[key] != ident:
                raise ValueError('Distinct tensors sharing storage require an explicit alias audit.')
            storage_owners[key] = ident
    groups = defaultdict(list)
    for ident, tensor in tensors.items():
        groups[(tensor.device, tensor.dtype)].append((ident, tensor))
    copied = {}
    for entries in groups.values():
        # Concatenation performs no arithmetic or precision conversion.
        flat = torch.cat([t.detach().reshape(-1) for _, t in entries]).cpu()
        offset = 0
        for ident, tensor in entries:
            copied[ident] = flat.narrow(0, offset, tensor.numel()).view(tensor.shape)
            offset += tensor.numel()

    def rebuild(value):
        if isinstance(value, torch.Tensor): return copied[id(value)]
        if isinstance(value, dict): return type(value)((k, rebuild(v)) for k, v in value.items())
        if isinstance(value, list): return [rebuild(v) for v in value]
        if isinstance(value, tuple): return tuple(rebuild(v) for v in value)
        return value

    return rebuild(state)


def split_semantic_outputs(output, count):
    """Return batch-one views for the original semantic_log_fields function."""
    fields = ('pred_masks', 'pred_logits', 'presence_logit_dec')
    for key in fields:
        if not isinstance(output[key], torch.Tensor) or output[key].shape[0] != count:
            raise ValueError(f'Unexpected batched output for {key}')
    return [{key: output[key][i:i+1] for key in fields} for i in range(count)]


def forward_batch(original, model, processor, images, *, collate=None, device='cuda'):
    """Evaluate a small slice batch using original targets, precision and metrics.

    Intended to be wrapped with FeatureReuse for the same image/text reuse as
    the current evaluator. Training calls and ground-truth prompts are forbidden.
    The GPU audit must compare this with the *already optimized* batch-one path.
    """
    if torch.is_grad_enabled() or any(m.training for m in model.modules()):
        raise RuntimeError('This candidate is evaluation only.')
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Evaluation parameters must be frozen.')
    if not images or len(images) > 4 or any(image.shape != images[0].shape for image in images):
        raise ValueError('Use 1–4 slices from one volume with identical native dimensions.')
    if collate is None:
        from sam3.train.data.collator import collate_fn_api
        collate = collate_fn_api
    per_slice = [[] for _ in images]
    for class_id in (1, 2):
        points = [original.datapoint(image, None, class_id, processor) for image in images]
        batch = collate(points, dict_key='input', with_seg_masks=True)['input']
        for target in batch.find_targets:
            if target.segments.numel() == 0:
                target.segments = torch.empty((0, processor.resolution, processor.resolution), dtype=torch.bool)
        batch = original.move_to_device(batch, device)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
        with context:
            outputs = model(batch)
        parts = split_semantic_outputs(outputs.output[-1][-1], len(images))
        for fields, part, image in zip(per_slice, parts, images):
            fields.append(original.semantic_log_fields(part, image.shape))
    return [original.exclusive_log_probabilities(fields) for fields in per_slice], None
