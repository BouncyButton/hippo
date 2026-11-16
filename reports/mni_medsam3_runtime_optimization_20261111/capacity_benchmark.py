"""Paired full-objective benchmark; zero changes to campaign parameters or outputs."""
import gc
import statistics
import time
import torch
from capacity_retention import retention, selected_indices
from retention_benchmark import state, restore, equal, save


def run(output, engine, model, processor, criterion, parameters, slabs, p, weights, mode=None):
    from scripts.cross_dataset_campaign.foundation_fast import validation_sampling
    modes = [8, 12, 28]
    receipt = dict(status='testing', timings={}, equivalence={}, gpu=torch.cuda.get_device_name(),
                   total_bytes=torch.cuda.get_device_properties(0).total_memory,
                   free_bytes=torch.cuda.mem_get_info()[0], campaign_updates=0,
                   objective='Disposable native+bands+edge+SDF+nonzero depth audit')
    initial = engine.parameter_digest(model, True)
    engine.reset_seed(7319)
    rng = state()
    schedule = slabs.schedule(p)

    def backward():
        model.zero_grad(set_to_none=True)
        for micro in range(p['accumulation']):
            case, anchor = schedule[micro]
            losses = engine.forward_slab(model, criterion, slabs, case, anchor)
            total = losses['native'] + sum(weights[n] * losses[n] for n in weights)
            (total / p['accumulation']).backward()
        return {n: float(v.detach()) for n, v in losses.items()}

    with retention(model, 8):
        backward()  # Warm kernels/cache before paired timing.
    for count in modes + list(reversed(modes)):
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        restore(rng)
        try:
            with retention(model, count):
                torch.cuda.synchronize(); began = time.perf_counter()
                losses = backward()
                torch.cuda.synchronize()
                row = dict(seconds=time.perf_counter()-began,
                           peak_bytes=torch.cuda.max_memory_allocated(), losses=losses,
                           indices=selected_indices(model, count))
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            row = dict(error='out_of_memory')
        receipt['timings'].setdefault(str(count), []).append(row)
        print('TIMING', count, row, flush=True)
        save(output/'candidate_result.json', receipt)

    reference = None
    baseline = statistics.median(x['seconds'] for x in receipt['timings']['8'])
    for count in modes:
        timings = receipt['timings'][str(count)]
        if any('error' in x for x in timings):
            continue
        if count != 8 and statistics.median(x['seconds'] for x in timings) >= baseline / 1.02:
            receipt['equivalence'][str(count)] = dict(status='rejected_no_2percent_gain')
            continue
        model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); restore(rng)
        with validation_sampling(), retention(model, count):
            case, anchor = slabs.calibration_schedule(4)[0]
            losses = engine.forward_slab(model, criterion, slabs, case, anchor)
            names = ['native', 'bands', 'edge', 'sdf', 'depth']
            values = {n: float(losses[n].detach()) for n in names}
            assert values['depth'] != 0, 'Depth audit must exercise a nonzero loss'
            gradients = {}
            for i, name in enumerate(names):
                gs = torch.autograd.grad(losses[name], parameters,
                                         retain_graph=i < len(names)-1, allow_unused=True)
                gradients[name] = [None if g is None else g.detach().cpu() for g in gs]
                del gs
            final_rng = state()
            del losses
        if count == 8:
            reference = values, gradients, final_rng
            receipt['equivalence']['8'] = dict(status='reference', losses=values)
        else:
            receipt['equivalence'][str(count)] = dict(
                exact=equal(values, reference[0]) and equal(gradients, reference[1]) and equal(final_rng, reference[2]),
                losses_exact=equal(values, reference[0]), gradients_exact=equal(gradients, reference[1]),
                rng_exact=equal(final_rng, reference[2]), depth_loss=values['depth'])
        print('EQUIVALENCE', count, receipt['equivalence'][str(count)], flush=True)
        save(output/'candidate_result.json', receipt)
    receipt.update(status='complete', parameters_unchanged=initial == engine.parameter_digest(model, True))
    save(output/'candidate_result.json', receipt)
