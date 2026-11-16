"""Versioned operations overlay; preserve immutable experimental sources."""
import argparse
import ast
import inspect
import os
from pathlib import Path
import sys
import time
import re
import subprocess
from unittest.mock import patch

OPS=Path(__file__).resolve().parent


def submission_flags(config):
    """Keep every matrix and candidate allocation on the requested GPU pair."""
    if config.get('required_gpu_profile') != 'balanced' or config.get('workers') != 2:
        raise ValueError('Continuation requires the fixed two-40-GiB GPU profile')
    from dual_gpu import allocation_flags
    return ['--nodes=1', '--ntasks=2', '--cpus-per-task=4', '--mem=80G',
            *allocation_flags('balanced')]


def preferred_profile(text,allow_dual=True):
    from dual_gpu import free_full_nodes
    if allow_dual and len(free_full_nodes(text)) >= 2:return 'dual_full'
    for line in text.splitlines():
        fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
        if fields.get('State') not in ('IDLE','MIXED'):continue
        if int(fields.get('CPUTot','0'))-int(fields.get('CPUAlloc','0'))<8:continue
        if int(fields.get('RealMemory','0'))-int(fields.get('AllocMem','0'))<81920:continue
        total={k:int(v) for k,v in re.findall(r'gpu:([^:,()]+):(\d+)',fields.get('Gres',''))}
        used={k:int(v) for k,v in re.findall(r'gpu:([^:,()]+):(\d+)',fields.get('GresUsed',''))}
        if total.get('7g.80gb',0)-used.get('7g.80gb',0)>=1 and total.get('4g.40gb',0)-used.get('4g.40gb',0)>=1:
            return 'mixed'
    return 'balanced'


def lane_command(command,lane,profile=None):
    profile=profile or os.environ.get('HIPPO_GPU_PROFILE','balanced')
    if profile=='dual_full':
        from dual_gpu import dual_lane
        nodes=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
        return dual_lane(command,lane,nodes)
    if profile not in ('mixed','balanced'):raise ValueError('Unknown GPU profile')
    if profile=='mixed' and lane==0:
        return [v.replace('gpu:4g.40gb:1','gpu:7g.80gb:1').replace('--gpus-per-task=4g.40gb:1','--gpus-per-task=7g.80gb:1') for v in command]
    return command


def verify():
    from scripts.cross_dataset_campaign.common import read,digest
    for name,sha in read(OPS/'manifest.json').items():
        if digest(OPS/name)!=sha:raise ValueError('Runtime overlay changed: '+name)
    return read(OPS/'config.json')


def reference_gram(gradients):
    return [[sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(ga,gb)
                 if a is not None and b is not None) for gb in gradients] for ga in gradients]


def replace_gram(reference,helper):
    """Replace only the scalar-transfer expression, retaining the full trainer."""
    tree=ast.parse(inspect.getsource(reference));changed=0
    expected=ast.parse('gram = [[sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(ga,gb) if a is not None and b is not None) for gb in gradients] for ga in gradients]').body[0]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.ListComp) and any(isinstance(t,ast.Name) and t.id=='gram' for t in node.targets):
            if ast.dump(node,include_attributes=False)!=ast.dump(expected,include_attributes=False):
                raise ValueError('Reference calibration expression changed')
            node.value=ast.Call(func=ast.Name(id='_exact_batched_gram',ctx=ast.Load()),args=[ast.Name(id='gradients',ctx=ast.Load())],keywords=[])
            changed+=1
    if changed!=1:raise ValueError('Expected exactly one Gram expression')
    namespace=dict(reference.__globals__,_exact_batched_gram=helper)
    exec(compile(ast.fix_missing_locations(tree),str(OPS/'mni_runtime.py')+':calibration','exec'),namespace)
    return namespace[reference.__name__]


def install_worker():
    config=verify()
    from scripts.cross_dataset_campaign import foundation,adni_foundation_full_runtime as runtime
    from scripts.cross_dataset_campaign.foundation_fast import gram_matrix
    from scripts.cross_dataset_campaign.common import digest,write,read
    from selective_retention import retention
    import torch
    if digest(foundation.__file__)!=config['foundation_sha256']:raise ValueError('Unexpected scientific engine')
    policy=config['activation_retention']
    if policy!='text_vision8':raise ValueError('Only the audited retention policy is allowed')
    evidence=read(OPS/'benchmark_evidence.json')
    from statistics import median
    if (evidence.get('status')!='complete' or not evidence.get('parameters_unchanged')
            or not evidence['equivalence'][policy].get('exact')
            or max(r['peak_bytes'] for r in evidence['timings'][policy])>32*1024**3
            or median(r['seconds'] for r in evidence['timings'][policy])>=median(r['seconds'] for r in evidence['timings']['reference'])/1.02):
        raise ValueError('Activation retention lacks a valid speed/memory/equivalence audit')
    original_train=foundation.train
    def train(model,criterion,parameters,slabs,protocol,calibration,arm,output):
        timing_path=output/'runtime_step_timings.json'
        from scripts.cross_dataset_campaign.common import read
        timings=read(timing_path) if timing_path.exists() else []
        previous=None
        save=foundation.save_checkpoint
        def timed_save(path,state):
            nonlocal previous
            began=time.monotonic();save(path,state)
            row=dict(update=state['step'],elapsed_training_seconds=state['elapsed_seconds'],
                checkpoint_seconds=time.monotonic()-began,time=time.time(),
                job_id=os.environ.get('SLURM_JOB_ID'),operations_manifest_sha256=digest(OPS/'manifest.json'))
            if previous and row['update']>previous['update']:
                row['seconds_per_update']=(row['elapsed_training_seconds']-previous['elapsed_training_seconds'])/(row['update']-previous['update'])
            timings.append(row);write(timing_path,timings);previous=row
        proof=dict(policy=policy,status='running',operations_manifest_sha256=digest(OPS/'manifest.json'),
            architecture_flags_changed=False,experimental_settings_changed=False,started_at=time.time())
        write(output/'runtime_retention.json',proof)
        try:
            with retention(model,policy),patch.object(foundation,'save_checkpoint',timed_save):
                result=original_train(model,criterion,parameters,slabs,protocol,calibration,arm,output)
            proof['status']='training_complete'
            return result
        except BaseException as exc:
            proof['status']='checkpointed_stop' if isinstance(exc,SystemExit) and exc.code==75 else 'failed'
            raise
        finally:
            proof.update(finished_at=time.time(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
            write(output/'runtime_retention.json',proof)
    foundation.train=train
    original=runtime.REFERENCE_CALIBRATE
    def calibrate(model,criterion,parameters,slabs,protocol,arm,output):
        proof=dict(status='enabled',operation='Batch scalar transfers; unchanged FP64 tensor reductions and ordered Python sums',
            operations_manifest_sha256=digest(OPS/'manifest.json'),foundation_sha256=config['foundation_sha256'],
            calls=0,guard=None,gram_seconds=0.,reference_seconds=None,batched_seconds=None)
        def helper(gradients):
            start=time.perf_counter()
            if proof['guard'] is None:
                began=time.perf_counter();expected=reference_gram(gradients)
                proof['reference_seconds']=time.perf_counter()-began
                began=time.perf_counter();actual=gram_matrix(gradients)
                proof['batched_seconds']=time.perf_counter()-began
                proof['guard']='bit_exact' if actual==expected else 'fallback_reference'
                value=actual if actual==expected else expected
            else:
                value=gram_matrix(gradients) if proof['guard']=='bit_exact' else reference_gram(gradients)
            proof['calls']+=1;proof['gram_seconds']+=time.perf_counter()-start
            return value
        function=replace_gram(original,helper)
        started=time.perf_counter()
        try:
            with retention(model,policy):
                return function(model,criterion,parameters,slabs,protocol,arm,output)
        finally:
            proof['calibration_seconds']=time.perf_counter()-started
            write(output/'runtime_optimization.json',proof)
    runtime.REFERENCE_CALIBRATE=calibrate


def install_controller():
    config=verify()
    from scripts.cross_dataset_campaign import adni_foundation_full_pool as pool
    from scripts.cross_dataset_campaign.common import read,write,digest
    from scripts.mni_medsam3_cluster import constraint_campaign as matrix
    from scripts.mni_medsam3_cluster.offline_ops import offline_ops as transport
    submit=transport.submit
    def optimized_submit(stage,plan):
        previous=transport.OPS;transport.OPS=OPS
        original_check=transport.subprocess.check_output
        def checked(command,*args,**kwargs):
            if isinstance(command,list) and command and command[0]=='sbatch':
                command[1:1]=submission_flags(config)
            return original_check(command,*args,**kwargs)
        try:
            with patch.object(transport.subprocess,'check_output',checked):result=submit(stage,plan)
            from dual_gpu import record_pending
            record_pending(stage,result)
            return result
        finally:transport.OPS=previous
    transport.submit=optimized_submit
    command=matrix.command
    def worker_command(*args,**kwargs):
        result=command(*args,**kwargs)
        index=result.index('scripts.mni_medsam3_cluster.constraint_worker')
        result[index:index+1]=['mni_runtime','worker']
        return lane_command(result,args[4] if len(args)>4 else kwargs['lane'])
    matrix.command=worker_command
    from scripts.mni_medsam3_cluster import adaptive_campaign
    baseline_command=adaptive_campaign.command
    def candidate_command(*args,**kwargs):
        result=baseline_command(*args,**kwargs)
        index=result.index('scripts.mni_medsam3_cluster.adaptive_worker')
        result[index:index+1]=['mni_runtime','baseline-worker']
        return lane_command(result,args[4] if len(args)>4 else kwargs['lane'])
    adaptive_campaign.command=candidate_command
    def gpu_probe(stage,plan):
        job=os.environ['SLURM_JOB_ID'];records=[]
        for lane in range(2):
            folder=stage/'gpu_probe'/job/f'lane{lane}'
            command=lane_command(pool.step_prefix(),lane)+['env',f'HIPPO_GPU_WORKER={lane}',
                plan['foundation_python'],'-m','scripts.cross_dataset_campaign.foundation_gpu_worker','--probe-directory',str(folder)]
            subprocess.run(command,cwd=stage/'source',check=True)
            records.append(read(folder/'0.json'))
        if any(r['status']!='passed' or r['visible_devices']!=1 or r['slurm_job_id']!=job for r in records) or len({r['device_uuid'] for r in records})!=2:
            raise ValueError('GPU isolation failed')
        expected={'dual_full':[75,75],'mixed':[75,38],'balanced':[38,38]}[os.environ.get('HIPPO_GPU_PROFILE','balanced')]
        if os.environ.get('HIPPO_GPU_PROFILE')=='dual_full' and len({r['host'] for r in records})!=2:
            raise ValueError('Full GPU lanes must use distinct nodes')
        if any(r['total_memory']<gib*1024**3 for gib,r in zip(expected,records)):
            raise ValueError('Assigned GPU profile differs')
        write(stage/'gpu_allocation.json',dict(status='passed',workers=records,job_id=job,
            manifest_sha256=digest(stage/'manifest.json'),operations_manifest_sha256=digest(OPS/'manifest.json'),
            gpu_profile=os.environ.get('HIPPO_GPU_PROFILE','balanced')))
    pool.verify_gpu_probe=gpu_probe
    run=pool.run
    def allocated_run(stage,plan,*args):
        job=os.environ['SLURM_JOB_ID']
        actual=dict(plan,graceful_stop_seconds=config['graceful_stop_seconds'],max_job_hours=config['max_job_hours'])
        write(stage/'runtime_allocations'/f'{job}.json',dict(job_id=job,
            original_manifest_sha256=digest(stage/'manifest.json'),operations_manifest_sha256=digest(OPS/'manifest.json'),
            max_job_hours=config['max_job_hours'],graceful_stop_seconds=config['graceful_stop_seconds'],
            gpu_profile=os.environ.get('HIPPO_GPU_PROFILE','balanced'),
            workers=2,scientific_protocol_changed=False,time=time.time()))
        return run(stage,actual,*args)
    pool.run=allocated_run


def main():
    command=sys.argv[1]
    if command in ('worker','baseline-worker'):
        sys.argv=[sys.argv[0],*sys.argv[2:]];install_worker()
        from scripts.cross_dataset_campaign.common import write,digest
        output=Path(sys.argv[sys.argv.index('--output')+1])
        write(output/'runtime_worker.json',dict(operations_manifest_sha256=digest(OPS/'manifest.json'),
            scientific_protocol_changed=False,calibration='Per-run exact scalar-transfer batching with reference fallback',time=time.time()))
        if command=='baseline-worker':
            from scripts.mni_medsam3_cluster.adaptive_worker import main as worker
        else:
            from scripts.mni_medsam3_cluster.constraint_worker import main as worker
        worker();return
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['matrix-serve','search-serve','run'])
    parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();stage=args.stage.resolve()
    install_controller()
    from scripts.cross_dataset_campaign.common import read
    from scripts.mni_medsam3_cluster import constraint_campaign as matrix
    if args.command=='matrix-serve':matrix.serve(stage)
    elif args.command=='search-serve':
        from scripts.mni_medsam3_cluster import seed_search
        seed_search.serve(stage)
    elif 'candidate_seed' in read(stage/'plan.json'):
        from scripts.mni_medsam3_cluster import seed_search
        seed_search.run_baseline(stage)
    else:matrix.run(stage)


if __name__=='__main__':main()
