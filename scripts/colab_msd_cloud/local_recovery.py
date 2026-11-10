"""Audit an interrupted local checkpoint before allowing the frozen resume gate."""
import json
from pathlib import Path

from store import atomic, digest


def checkpoint_ledger(checkpoint, configuration, ledger, protocol, calibration, calibration_sha):
    """Derive only charges; leave all checkpoint/model/optimizer/RNG bytes intact."""
    from scripts.capped_depth_delay.schedule import training_flops
    if checkpoint['configuration'] != configuration:
        raise ValueError('Local checkpoint configuration differs; preserve for audit')
    step = checkpoint['charged_attempts']
    if checkpoint['step'] != step or not isinstance(step, int) or step < 0:
        raise ValueError('Local checkpoint has discarded or invalid updates')
    schedule = protocol['models'][protocol['trainer_alias']]['objectives'][configuration['objective']]
    if step > ledger['charged_attempts'] or ledger['training_flops'] != training_flops(ledger['charged_attempts'], schedule):
        raise ValueError('Local checkpoint/charge arithmetic differs')
    if ledger['charged_attempts'] > schedule['max_steps'] or ledger['training_flops'] > protocol['budget_flops']:
        raise ValueError('Local charge exceeds the frozen cap')
    if ledger['calibration_flops'] != ledger['calibration_attempts'] * schedule['calibration_flops']:
        raise ValueError('Local calibration charge arithmetic differs')
    restored = dict(ledger, charged_attempts=step, training_flops=training_flops(step, schedule))
    if ledger['calibration_attempts']:
        if (not calibration or calibration_sha != ledger.get('calibration_sha256')
                or calibration.get('status') != 'complete'
                or calibration.get('config_sha256') != ledger['config_sha256']
                or calibration.get('anchor_training_attempts') != schedule['prefix_updates']
                or calibration.get('anchor_successful_updates') != schedule['prefix_updates']
                or step < schedule['prefix_updates']):
            raise ValueError('Local calibration is not recoverable at this checkpoint; preserve for audit')
    elif configuration['objective'] != 'dice' and step > schedule['prefix_updates']:
        raise ValueError('Constraint checkpoint lacks completed calibration')
    return restored


def reconcile_local(cloud, group, output, root, local, cache, phase, task, event_factory):
    """Commit lost-tail evidence BEFORE reducing any active charge ledger."""
    import torch
    current = json.loads((output/'budget_ledger.json').read_text())
    checkpoint_path = output/'checkpoint_latest.pt'
    if not checkpoint_path.exists():
        if current.get('charged_attempts', 0) or current.get('calibration_attempts', 0):
            raise ValueError('Local charged work has no checkpoint; preserve for recovery audit: '+task['id'])
        return
    configuration = json.loads((output/'config.json').read_text())
    if (configuration['augmentation'] != 'none' or configuration['budget_flops'] != 31976450667090
            or (configuration['model_identity'], configuration['objective'], configuration['fold'], configuration['seed'])
            != (task['model'], task['arm'], task['fold'], task['seed'])):
        raise ValueError('Local recovery scientific identity differs')
    cfg_sha = digest(output/'config.json')
    if current['config_sha256'] != cfg_sha:
        raise ValueError('Local recovery configuration hash differs')
    state = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False, mmap=True)
    protocol = json.loads((root/'protocols'/task['protocol']/'protocol.json').read_text())
    calibration_path = output/'calibration.json'
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    saved = checkpoint_ledger(state, configuration, current, protocol, calibration,
                              digest(calibration_path) if calibration else None)
    del state
    ledger_objects = [o for o in group if o['appProperties']['kind'] == 'ledger']
    if not ledger_objects and saved == current:
        # Setup may be repeated after an audited historical import but before
        # its first new update (and therefore before its first new ledger).
        prior = local/'recovery_events'/(task['id']+'.json')
        if prior.exists() and json.loads(prior.read_text()).get('restored_checkpoint_charge') == current:
            return
        if not current['charged_attempts'] and not current['calibration_attempts']:
            return
    if len(ledger_objects) != 1:
        raise ValueError('Local work needs exactly one server charge ledger: '+task['id'])
    obj = ledger_objects[0]
    server_path = cloud.download(obj['id'], cache/(obj['id']+'.json'))
    server = json.loads(server_path.read_text())
    recovery = local/phase/'recovery'/task['id']/'charged_ledger.json'
    candidates = [current, server]
    if recovery.exists():
        candidates.append(json.loads(recovery.read_text()))
    keys = ('charged_attempts', 'training_flops', 'calibration_flops', 'calibration_attempts')
    for candidate in candidates:
        if candidate.get('config_sha256') != cfg_sha or any(
                not isinstance(candidate.get(k), int) or candidate[k] < 0 for k in keys):
            raise ValueError('Recovery charge evidence differs from configuration')
    # A single monotone ledger must dominate every known charge. Never invent a
    # synthetic combination or ignore a possibly committed timed-out request.
    charged = next((a for a in candidates if all(all(a[k] >= b[k] for k in keys)
                                               for b in candidates)), None)
    if charged is None:
        raise ValueError('Conflicting local/server charge histories')
    if current == saved and server == saved and (not recovery.exists() or candidates[-1] == saved):
        return
    evidence = dict(local_checkpoint_sha256=digest(checkpoint_path),
                    local_ledger_before=current, server_ledger_id=obj['id'])
    event = event_factory(task['id'], charged, saved, evidence, 'server_store')
    event['recovery_method'] = 'same_VM_checkpoint_exact_bytes'
    event_path = cache/(task['id']+'_local_tail_recovery.json')
    atomic(event_path, event)
    cloud.put_immutable(event_path, 'recovery_event', task['id'], phase)
    # If interrupted here, the next attempt recommits identical evidence before
    # touching any charge. This event is content-addressed and is not duplicated.
    restored_path = cache/(task['id']+'_restored_local_charge.json')
    atomic(restored_path, saved)
    receipt = cloud.upload(restored_path, file_id=obj['id'])
    atomic(output.parent/(task['id']+'_ledger_id.json'), receipt)
    atomic(recovery, saved)
    atomic(output/'budget_ledger.json', saved)
    atomic(local/'recovery_events'/(task['id']+'_local.json'), event)
    print('RECOVERED LOCAL CHECKPOINT', task['id'], 'at update', saved['charged_attempts'],
          '| charged tail retained as recovery overhead:', event['lost_charged_attempts'], flush=True)
