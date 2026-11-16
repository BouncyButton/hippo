"""Bounded diagnostic on one allocated GPU; never writes campaign checkpoints."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def supervisor(args):
    """Resume the paused worker even if the diagnostic fails or times out."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Diagnostic GPU binding unavailable; campaign not paused')
    if torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
        raise RuntimeError('Full A100 required; campaign not paused')
    candidates = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry/'cmdline').read_bytes().split(b'\0')
            if command and Path(command[0].decode()).name.startswith('python') and b'mni_runtime' in command and b'worker' in command and any(
                args.run.encode() in part for part in command):
                candidates.append(int(entry.name))
        except (OSError, PermissionError):
            pass
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one live target worker: {candidates}')
    target = candidates[0]
    started = time.time()
    receipt = args.output/'supervision.json'
    write(receipt, dict(target_pid=target, state='preparing', started_at=started))
    def interrupted(signum, frame):
        raise RuntimeError(f'Diagnostic interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # Separate watchdog owns both the timeout and unconditional SIGCONT.
    os.kill(target, signal.SIGSTOP)
    child = None
    try:
        child = subprocess.Popen([sys.executable, __file__, 'benchmark', '--output', str(args.output),
            '--stage', str(args.stage), '--run', args.run, '--mode', args.mode])
        write(receipt, dict(target_pid=target, diagnostic_pid=child.pid, state='profiling', started_at=started))
        try:
            result = child.wait(timeout=720)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            raise
        if result:
            raise RuntimeError(f'Diagnostic exited {result}')
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        os.kill(target, signal.SIGCONT)
        write(receipt, dict(target_pid=target, state='resumed', started_at=started,
                           stopped_seconds=time.time()-started))


def benchmark(args):
    import torch
    from scripts.cross_dataset_campaign import foundation as engine
    from scripts.cross_dataset_campaign import foundation_msd as matched
    from scripts.colab_mni_medsam3.optimized import Volumes
    from scripts.mni_medsam3_cluster.constraint_support import install
    from scripts.cross_dataset_campaign.common import read
    from torch.profiler import profile, ProfilerActivity, record_function
    torch.set_num_threads(4)
    p = read(args.stage/'protocols'/args.run.rsplit('_', 1)[-1]/'protocol.json')
    p['arm'] = 'dice_bands_edge_sdf_depth'  # Disposable full-loss audit, never a campaign update.
    engine.reset_seed(p['seed'])
    matched.configure_retained_backward()
    install(p)
    engine.Volumes = Volumes
    print('Loading disposable profiling model', flush=True)
    model, processor = matched.load_foundation(p, Path('/tmp')/f"hippo-sam3-{os.getuid()}-{os.environ['SLURM_JOB_ID']}.pt")
    for n, parameter in model.named_parameters():
        parameter.requires_grad_(engine.is_adapter(n))
    parameters = [v for v in model.parameters() if v.requires_grad]
    criterion = engine.native_loss(model)
    slabs = engine.SupportSlabs(Volumes(p), p, processor)
    model.train()
    # Fixed coefficients exercise every loss in a disposable paired benchmark.
    weights = {'bands': 1.0, 'edge': 1.0, 'sdf': 1.0, 'depth': 1.0}
    schedule = slabs.schedule(p)
    if args.mode in ('candidates','casts','selective'):
        from capacity_benchmark import run
        run(args.output, engine, model, processor, criterion, parameters, slabs, p, weights, mode=args.mode)
        return
    optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], weight_decay=.01)
    before = engine.parameter_digest(model, True)
    def update(index):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(p['accumulation']):
            case, anchor = schedule[index*p['accumulation']+micro]
            with record_function('HIPPO_FORWARD_SLAB'):
                losses = engine.forward_slab(model, criterion, slabs, case, anchor)
            with record_function('HIPPO_LOSS_BACKWARD'):
                total = losses['native']+sum(weights[n]*losses[n] for n in weights)
                if not torch.isfinite(total): raise ValueError('Nonfinite diagnostic loss')
                (total/p['accumulation']).backward()
                logged = {k:float(v.detach()) for k,v in losses.items()}
        with record_function('HIPPO_CLIP_AND_OPTIMIZER'):
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
        return logged
    timings = []
    for i in range(3):
        torch.cuda.synchronize(); start=time.perf_counter()
        values=update(0)
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-start)
        print('Unprofiled update', i, timings[-1], flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        update(0)
        torch.cuda.synchronize()
    averages = prof.key_averages()
    (args.output/'profile_cpu.txt').write_text(averages.table(sort_by='self_cpu_time_total',row_limit=45))
    (args.output/'profile_cuda.txt').write_text(averages.table(sort_by='self_device_time_total',row_limit=45))
    events=[dict(key=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,cpu_total_us=e.cpu_time_total,
                 self_device_us=e.self_device_time_total,device_total_us=e.device_time_total) for e in averages]
    result=dict(status='profiled',gpu=torch.cuda.get_device_name(),threads=torch.get_num_threads(),
        protocol_sha256=engine.digest(args.stage/'runs'/args.run/'protocol.json'),timings=timings,
        events=events,peak_allocated=torch.cuda.max_memory_allocated(),cache=dict(hits=processor.transform.hits,
        misses=processor.transform.misses),initial_adapter_digest=before,
        disposable_optimizer_updates=4,campaign_optimizer_updates=0)
    write(args.output/'profile.json',result)
    print('PROFILE COMPLETE',json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['supervisor','benchmark'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--mode',choices=['profile','candidates','casts','selective'],default='profile')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command=='supervisor': supervisor(args)
    else: benchmark(args)
