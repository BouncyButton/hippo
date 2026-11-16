"""Two full A100s on separate nodes, with bounded pending time and safe fallback."""
import re


def free_full_nodes(text):
    nodes = []
    for line in text.splitlines():
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if fields.get('State') not in ('IDLE', 'MIXED'):
            continue
        if int(fields.get('CPUTot', '0')) - int(fields.get('CPUAlloc', '0')) < 4:
            continue
        if int(fields.get('RealMemory', '0')) - int(fields.get('AllocMem', '0')) < 40960:
            continue
        total = {k: int(v) for k, v in re.findall(r'gpu:([^:,()]+):(\d+)', fields.get('Gres', ''))}
        used = {k: int(v) for k, v in re.findall(r'gpu:([^:,()]+):(\d+)', fields.get('GresUsed', ''))}
        if total.get('7g.80gb', 0) > used.get('7g.80gb', 0):
            nodes.append(fields['NodeName'])
    return sorted(set(nodes))


def allocation_flags(profile):
    if profile == 'dual_full':
        # GRES and memory are per node; exactly one GPU on each of two nodes.
        return ['--nodes=2', '--ntasks=2', '--ntasks-per-node=1', '--mem=40G',
                '--gres=gpu:7g.80gb:1', '--export=ALL,HIPPO_GPU_PROFILE=dual_full']
    if profile not in ('mixed', 'balanced'):
        raise ValueError('Unknown GPU profile')
    gres = 'gpu:7g.80gb:1,gpu:4g.40gb:1' if profile == 'mixed' else 'gpu:4g.40gb:2'
    return ['--gres=' + gres, '--export=ALL,HIPPO_GPU_PROFILE=' + profile]


def dual_lane(command, lane, nodes):
    if len(nodes) != 2 or len(set(nodes)) != 2 or lane not in (0, 1):
        raise ValueError('Two distinct allocated nodes required')
    result = [v.replace('gpu:4g.40gb:1', 'gpu:7g.80gb:1')
              .replace('--gpus-per-task=4g.40gb:1', '--gpus-per-task=7g.80gb:1')
              for v in command]
    result.insert(1, '--nodelist=' + nodes[lane])
    return result


def pending_decision(record, queue, now, timeout=900):
    """Pure decision: never cancel a running job or count dependency wait."""
    result = dict(record)
    job = str(record['job_id'])
    own = next((row for row in queue if row[0] == job), None)
    if own is None or own[1] != 'PENDING' or record.get('status') == 'fallback_cancelled':
        return result, False
    dependency = record.get('dependency')
    if dependency and any(row[0] == dependency for row in queue):
        result.pop('eligible_seen_at', None)
        return result, False
    result.setdefault('eligible_seen_at', now)
    return result, now - result['eligible_seen_at'] >= timeout


def guard_pending(stage, queue):
    import time
    import subprocess
    from scripts.cross_dataset_campaign.common import read, write
    from scripts.cross_dataset_campaign import controller
    path = stage/'dual_full_pending.json'
    if not path.exists():
        return
    record, cancel = pending_decision(read(path), queue, time.time())
    if cancel:
        fresh = controller.queue()
        # Check again at the mutation boundary: a newly RUNNING job is protected.
        record, cancel = pending_decision(record, fresh, time.time())
        if cancel:
            subprocess.run(['scancel', '--state=PENDING', str(record['job_id'])], check=True)
            record.update(status='fallback_cancelled', cancelled_at=time.time(),
                          disable_dual_until=time.time()+21600)
    write(path, record)


def record_pending(stage, submission):
    from scripts.cross_dataset_campaign.common import write
    if submission and '--export=ALL,HIPPO_GPU_PROFILE=dual_full' in submission['command']:
        write(stage/'dual_full_pending.json', dict(job_id=submission['job_id'],
              dependency=submission['dependency'], submitted_at=submission['submitted_at'],
              status='waiting', maximum_eligible_wait_seconds=900))


def queue_successor(stage, parent, ops):
    import fcntl
    import subprocess
    import time
    from scripts.cross_dataset_campaign import controller
    from scripts.cross_dataset_campaign.common import read, write, digest
    from scripts.cross_dataset_campaign.storage import quota
    parent = str(parent)
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = controller.queue()
        if len(current) != 1 or current[0][0] != parent or current[0][1] != 'RUNNING':
            raise ValueError('Expected only the authorized running predecessor')
        if not (stage/'submissions'/f'{parent}.json').exists():
            raise ValueError('Predecessor does not belong to this campaign')
        if read(stage/'control.json').get('pause'):
            raise ValueError('Campaign is paused')
        preflight = read(ops/'preflight.json')
        if preflight['status'] != 'passed' or preflight['operational_manifest_sha256'] != digest(ops/'manifest.json'):
            raise ValueError('Matching runtime preflight required')
        if quota()['free_gib'] < read(stage/'plan.json')['storage']['minimum_free_gib']:
            raise ValueError('Insufficient quota headroom')
        command = ['sbatch', *allocation_flags('dual_full'), '--parsable',
                   '--dependency=afterany:'+parent, '--chdir='+str(stage),
                   '--output='+str(stage/'slurm-%j.out'), '--error='+str(stage/'slurm-%j.err'),
                   str(ops/'run.sbatch'), str(stage)]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected Slurm job id')
        record = dict(job_id=job, dependency=parent, command=command, submitted_at=time.time(),
                      manifest_sha256=digest(stage/'manifest.json'),
                      operational_manifest_sha256=digest(ops/'manifest.json'), laptop_required=False)
        write(stage/'submissions'/f'{job}.json', record)
        write(stage/'submission.json', record)
        record_pending(stage, record)
        return record
