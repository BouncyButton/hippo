"""Bound this campaign's bulk upload volume without changing model updates."""
import fcntl
import json
import os
from pathlib import Path
import time

WINDOW_SECONDS=24*3600
# Keep every old reservation. The larger, bounded ceiling provides migration
# headroom after the old 200-GiB guard fired; it is NOT a Google quota estimate.
DEFAULT_LIMIT=320*1024**3
RECOVERY_RESERVE=32*1024**3
POLICY='checkpoint_15min_reserve_v3'


class RoutineBudgetExceeded(RuntimeError):
    """Drain workers into the recovery reserve instead of failing their saves."""
    checkpoint_budget_pause=True


def reserved_window(state, size, name, now, limit=DEFAULT_LIMIT, *, priority='recovery'):
    entries=[e for e in state.get('reservations',[]) if e['time']>now-WINDOW_SECONDS]
    used=sum(e['bytes'] for e in entries)
    reserve=min(RECOVERY_RESERVE,limit//10)
    if priority=='routine' and used+size>limit-reserve:
        raise RoutineBudgetExceeded('Routine checkpoint allowance reached; saving active workers from the reserved allowance.')
    if used+size>limit:
        from store import DriveBlocked
        raise DriveBlocked('The notebook bulk-upload budget for the last 24 hours is exhausted. '
                           'Stopping before uploading more; local checkpoints and server charges are preserved.')
    entries.append(dict(time=now,bytes=size,name=name,priority=priority))
    history=list(state.get('policy_history',[]))
    if state.get('policy')!=POLICY:
        history.append(dict(time=now,previous_limit_bytes=state.get('limit_bytes'),
                            new_limit_bytes=limit,prior_reserved_bytes=used,
                            reason='15-minute snapshots and dedicated clean-stop reserve; no reservations reset'))
    return dict(limit_bytes=limit,window_seconds=WINDOW_SECONDS,reservations=entries,
                reserved_bytes=used+size,reserve_bytes=reserve,policy=POLICY,policy_history=history,
                note='Reservations include failed/unknown upload attempts. Small ledgers and other campaigns are outside this bulk-volume guard; provider throttling still stops work.')


def reserve_upload(cloud, path, properties=None):
    from store import atomic
    config=json.loads(Path(os.environ['HIPPO_CLOUD_CONFIG']).read_text())
    local=Path(config['local_root']);local.mkdir(parents=True,exist_ok=True)
    with (local/'traffic_window.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        found=cloud.execute(cloud.service().files().list(
            q="'"+cloud.folder+"' in parents and trashed=false and name='bulk_upload_window.json'",
            fields='files(id)',pageSize=100)).get('files',[])
        if len(found)>1:raise RuntimeError('Duplicate upload-budget state; stop and preserve work')
        target=local/'bulk_upload_window.json'
        if found:
            cloud.download(found[0]['id'],target)
            state=json.loads(target.read_text())
        else:state={}
        properties=properties or {}
        priority=('routine' if properties.get('kind')=='checkpoint'
                  and properties.get('upload_priority')!='recovery' else 'recovery')
        value=reserved_window(state,Path(path).stat().st_size,Path(path).name,time.time(),priority=priority)
        atomic(target,value)
        # This compact journal is always below the bulk-upload threshold.
        if target.stat().st_size>=4*1024**2:raise RuntimeError('Upload-budget journal unexpectedly large')
        cloud.upload(target,'bulk_upload_window.json',{'kind':'traffic_budget'},
                     file_id=found[0]['id'] if found else None)
        print('BULK UPLOAD BUDGET',round(value['reserved_bytes']/1024**3,2),
              '/320 GiB in last 24h | 32 GiB reserved for recovery |',priority,flush=True)
