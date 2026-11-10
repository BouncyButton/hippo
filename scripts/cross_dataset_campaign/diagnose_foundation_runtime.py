"""Read-only GPU reproduction of the foundation equivalence-gate failure."""
import argparse
import hashlib
import time
from pathlib import Path

import torch

from scripts.cross_dataset_campaign import foundation_fast as fast
from scripts.cross_dataset_campaign.common import read, write


def compare(left, right, names):
    rows = []
    for name, a, b in zip(names, left, right):
        if a is None or b is None:
            if (a is None) != (b is None):
                rows.append(dict(parameter=name, connectivity_differs=True))
            continue
        diff = (a.double() - b.double())
        if not torch.equal(a, b):
            rows.append(dict(parameter=name, allclose=torch.allclose(a,b,rtol=1e-5,atol=1e-7),
                max_abs=float(diff.abs().max()), relative_l2=float(diff.norm()/a.double().norm().clamp_min(1e-30)),
                a_norm=float(a.double().norm()), b_norm=float(b.double().norm())))
    return dict(changed_parameters=len(rows), failed_parameters=sum(not r.get('allclose',False) for r in rows),
        worst=sorted(rows,key=lambda x:x.get('max_abs',float('inf')),reverse=True)[:12])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-grid-check',action='store_true')
    parser.add_argument('--deterministic',action='store_true')
    args=parser.parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    if args.cpu_grid_check:
        import torch.nn.functional as functional
        grid_sample=functional.grid_sample
        def stable_grid_sample(input,grid,**kwargs):
            dtype=torch.float32 if input.dtype in (torch.float16,torch.bfloat16) else input.dtype
            return grid_sample(input.to(device='cpu',dtype=dtype),grid.to(device='cpu',dtype=dtype),**kwargs).to(input.device)
        functional.grid_sample=stable_grid_sample
    torch.set_num_threads(4)
    p={**read(args.stage/'protocols/ADNI/protocol.json'),'arm':'dice'}
    fast.original.reset_seed(p['seed'])
    model,processor=fast.load_foundation(p,args.base)
    for name,value in model.named_parameters():value.requires_grad_(fast.original.is_adapter(name))
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    parameters=[v for _,v in named];names=[n for n,_ in named]
    criterion=fast.original.native_loss(model)
    slabs=fast.SupportSlabs(fast.original.Volumes(p),p,processor)
    case,anchor=slabs.calibration_schedule(4)[0]
    model.train()
    result=dict(torch=torch.__version__,device=torch.cuda.get_device_name(),case=case,anchor=anchor,passes=[])
    def run(label,fn,cache):
        processor.transform.enabled=cache
        fast.original.reset_seed(7319)
        started=time.monotonic()
        losses=fn(model,criterion,slabs,case,anchor)
        grads=torch.autograd.grad(losses['native'],parameters,allow_unused=True)
        grads=[None if g is None else g.detach().cpu() for g in grads]
        row=dict(label=label,loss=float(losses['native'].detach()),seconds=time.monotonic()-started,
            cuda_rng=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
        result['passes'].append(row)
        write(args.output,result);print(row,flush=True)
        return grads
    a=run('reference_a',fast.REFERENCE_SLAB,False)
    b=run('reference_b',fast.REFERENCE_SLAB,False)
    result['reference_repeat']=compare(a,b,names);write(args.output,result)
    c=run('optimized_a',fast.forward_slab,True)
    result['reference_optimized']=compare(a,c,names);write(args.output,result)
    d=run('optimized_b',fast.forward_slab,True)
    result['optimized_repeat']=compare(c,d,names);write(args.output,result)
    del a,b,c,d
    try:
        if args.cpu_grid_check:
            write(args.output,result)
            return
        torch.use_deterministic_algorithms(True)
        a=run('deterministic_reference',fast.REFERENCE_SLAB,False)
        b=run('deterministic_optimized',fast.forward_slab,True)
        result['deterministic_comparison']=compare(a,b,names)
    except Exception as exc:
        result['deterministic_error']=repr(exc)
    write(args.output,result)
    print(result,flush=True)


if __name__=='__main__':main()
