"""Read-only microbenchmark on synthetic gradients with real adapter shapes."""
import argparse
import json
import statistics
import time
import torch
from scripts.cross_dataset_campaign.foundation_fast import gram_matrix


def reference(gradients):
    return [[sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(ga,gb)
                 if a is not None and b is not None) for gb in gradients] for ga in gradients]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(17011)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    shapes=[v.shape for v in checkpoint['adapter'].values()];del checkpoint
    output=[]
    for count in (2,3,4,5):
        gradients=[tuple(torch.randn(shape,device='cuda') if (i+j)%19 else None for j,shape in enumerate(shapes)) for i in range(count)]
        expected=reference(gradients);actual=gram_matrix(gradients)
        assert actual==expected,(count,'not bit-exact')
        timings={'reference':[],'batched':[]}
        for repeat in range(4):
            methods=[('reference',reference),('batched',gram_matrix)]
            if repeat%2:methods.reverse()
            for name,method in methods:
                torch.cuda.synchronize();start=time.perf_counter();value=method(gradients);torch.cuda.synchronize()
                timings[name].append(time.perf_counter()-start)
                assert value==expected
        a=statistics.median(timings['reference']);b=statistics.median(timings['batched'])
        output.append(dict(losses=count,adapter_tensors=len(shapes),exact=True,reference_seconds=a,batched_seconds=b,speedup=a/b))
        del gradients
    print(json.dumps(dict(status='passed',benchmark='Gram scalar-transfer batching only; synthetic gradients, no optimizer updates',
        gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),results=output)),flush=True)


if __name__=='__main__':main()
