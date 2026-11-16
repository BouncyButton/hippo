"""Zero-update replay of the established native graph and strict CUDA audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from contextlib import nullcontext


def main():
    import torch
    from scripts.cross_dataset_campaign import foundation as original, foundation_msd as matched
    from scripts.colab_hfh.data import Volumes
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/content/hippo_hfh'))
    parser.add_argument('--empty-grid-audit', action='store_true')
    args = parser.parse_args()
    root = args.root
    output = root/'medsam_constraints_v1'/('determinism_probe_empty_grid.json' if args.empty_grid_audit else 'determinism_probe.json')
    p = json.loads((root/'bundle/medsam_protocols/f0_s0.json').read_text())
    p['arm'] = 'dice'
    torch.set_num_threads(4)
    matched.configure_retained_backward()
    original.reset_seed(p['seed'])
    model, processor = matched.load_foundation(p, root/'sam3.pt')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original.is_adapter(name))
    params = [v for v in model.parameters() if v.requires_grad]
    criterion = original.native_loss(model)
    slabs = original.SupportSlabs(Volumes(p), p, processor)
    schedule = slabs.schedule(p)
    before = original.parameter_digest(model, True)
    result = dict(purpose='Zero optimizer updates; replay native full zero-auxiliary graph',
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), replays=[],
                  empty_grid_audit=args.empty_grid_audit,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def buffers_digest():
        h = hashlib.sha256()
        for name, value in model.named_buffers():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def replay(strict):
        torch.use_deterministic_algorithms(strict)
        original.reset_seed(p['seed']); model.train(); model.zero_grad(set_to_none=True)
        rows = []
        for case, anchor in schedule[:p['accumulation']]:
            losses = original.forward_slab(model, criterion, slabs, case, anchor)
            total = losses['native'] + min(1/p['constraint_warmup_updates'], 1.) * (
                0. * losses['bands'] + 0. * losses['edge'])
            (total/p['accumulation']).backward()
            rows.append(float(total.detach()))
        gradients = torch.cat([(torch.zeros_like(v) if v.grad is None else v.grad).detach().float().cpu().flatten()
                               for v in params])
        return dict(loss=sum(rows)/len(rows), gradient_norm=float(gradients.double().norm()),
                    gradient_sha256=hashlib.sha256(gradients.numpy().tobytes()).hexdigest()), gradients

    initial_buffers = buffers_digest()
    first = None
    for strict in (False, False, True, True):
        try:
            torch.use_deterministic_algorithms(strict)
            if strict and args.empty_grid_audit:
                from scripts.colab_medsam3.audit_numerics import audit_empty_point_grid
                context = audit_empty_point_grid()
            else:
                context = nullcontext(None)
            with context as audit:
                row, vector = replay(strict)
            if audit is not None: row['empty_grid_audit'] = audit
            row.update(strict=strict, status='complete')
            if first is None:
                first = vector
            else:
                difference = vector-first
                row.update(max_absolute_gradient_difference=float(difference.abs().max()),
                           relative_gradient_l2_difference=float(difference.double().norm()/first.double().norm()))
            result['replays'].append(row)
        except Exception as error:
            result['replays'].append(dict(strict=strict,status='error',type=type(error).__name__,
                                          error=str(error)[-6000:],traceback=traceback.format_exc()[-8000:]))
            if not strict: break
        result.update(adapter_unchanged=original.parameter_digest(model,True)==before,
                      buffers_unchanged=buffers_digest()==initial_buffers)
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['replays'][-1]),flush=True)
    print('PROBE_RESULT',str(output),flush=True)
    from scripts.colab_hfh.medsam_recovery import copy_verified
    copy_verified(output, Path('/content/hfh_storage/results_medsam3_constraints_v1/monitor')/output.name)


if __name__ == '__main__': main()
