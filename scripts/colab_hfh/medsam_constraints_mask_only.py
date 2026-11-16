"""Conditional-mask BAND; native score/presence objectives remain unchanged."""
import inspect

import torch
import torch.nn.functional as F

FORMULATION = 'native_matched_mask_band_v1'


def conditional_logits(output, shape, has_foreground):
    masks = F.interpolate(output['pred_masks'][0, :, None].float(), size=shape,
                          mode='bilinear', align_corners=False)[:, 0]
    if has_foreground:
        batch, query, target = output['indices']
        if query.numel() != 1 or batch.numel() != 1 or int(batch[0]) != 0:
            raise ValueError('Binary foreground slice must have exactly one native matched query')
        return masks[query.detach().long()[0]]
    # No GT instance exists on empty slices. Keep outer-band negative training
    # on the semantic winner, but selection is explicitly detached. Presence is
    # a scalar per image and cannot change this query ordering.
    with torch.no_grad():
        score = output['pred_logits'][0].float().reshape(-1)
        winner = (F.logsigmoid(masks.detach()) + F.logsigmoid(score.detach())[:, None, None]).argmax(0)
    return masks.gather(0, winner[None])[0]


def mask_band(logits, support, anchor):
    truth = support.foreground[anchor].to(logits.device)
    inner, outer = support.inner[anchor].to(logits.device), support.outer[anchor].to(logits.device)
    if not support.inner_count or not support.outer_count:
        return logits.sum()*0
    bce = F.binary_cross_entropy_with_logits(logits, truth.float(), reduction='none')
    return .5*support.foreground.shape[0]*(
        (bce*inner).sum()/(support.inner_count+1e-6) +
        (bce*outer).sum()/(support.outer_count+1e-6))


def install(engine, protocol):
    if protocol.get('constraint_recipe', {}).get('formulation') != FORMULATION:
        return
    if protocol['arm'] != 'dice_bands' or len(protocol['classes']) != 2:
        raise ValueError('Mask-only ablation is restricted to binary native+BAND')
    reference = engine.forward_slab

    def forward_slab(model, criterion, slabs, case, anchor):
        fields = engine.semantic_log_fields
        captured = []
        support = slabs.supports[case][1]
        zs = [anchor, (anchor+1) % len(slabs.indices[case])]

        def capture(output, shape):
            if len(captured) >= 2:
                raise ValueError('Unexpected number of binary plane forwards')
            z = zs[len(captured)]
            present = bool(support.foreground[z].any())
            captured.append(conditional_logits(output, shape, present))
            return fields(output, shape)

        engine.semantic_log_fields = capture
        try:
            losses = reference(model, criterion, slabs, case, anchor)
        finally:
            engine.semantic_log_fields = fields
        if len(captured) != 2:
            raise ValueError('Expected original two-slice slab')
        losses['bands'] = mask_band(captured[0], support, anchor)
        return losses

    engine.forward_slab = forward_slab


def install_spatial_evaluation(matched):
    """Add spatial counts to the unchanged prediction/evaluation path."""
    reference = matched.REFERENCE_EVALUATE
    source = inspect.getsource(reference)
    needle = "            n=len(protocol['classes'])"
    replacement = ("            from scripts.colab_hfh.medsam_spatial_audit import spatial\n"
                   "            spatial_row, _ = spatial(truth, prediction)\n" + needle)
    if source.count(needle) != 1 or source.count('shape=list(truth.shape)))') != 1:
        raise ValueError('Original evaluator changed; spatial adaptation requires review')
    source = source.replace(needle, replacement).replace(
        'shape=list(truth.shape)))', 'shape=list(truth.shape), spatial=spatial_row))')
    namespace = dict(reference.__globals__)
    exec(compile(source, __file__, 'exec'), namespace)
    matched.REFERENCE_EVALUATE = namespace['evaluate']
