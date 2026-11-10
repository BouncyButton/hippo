"""Explicit, one-time recovery of the user-approved October 22 interruption."""
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

TASK = 'GeoFWI3D_SAM3_f1s2_dice_bands_edge_sdf_u25_v1'
EVENT = '20261022_f1s2_bands_edge_sdf_restart0'
PROTOCOL = 'eb713458097972644f7497e63f1bed1099f80055c284b3e60da8b8a17029de08'
RECOVERY_ID = '1UZa9VcSqDTX2R2u8S4s8-v4vNPcyN4j0'
RECOVERY_SHA = '44f8230add3df566585f5e49e0b0633f47a933a943fb40e54d7ff855f38bc70f'


def repair(transport, unpack, digest, load_checkpoint):
    """Restore only the reviewed checkpoint; never reset an unrelated charge."""
    state = transport.state
    if TASK in state['completed']:
        print('Recovery unnecessary: the run is already complete.')
        return
    if EVENT in state.get('recovery_adjustments', {}):
        print('One-time recovery already recorded; continuing normally.')
        return
    charge = state['charges'].get(TASK)
    if charge != dict(step=1, protocol_sha256=PROTOCOL):
        raise RuntimeError('Recovery state changed; refusing an unreviewed rollback.')
    root = transport.root
    output = root/'work'/TASK
    checkpoint = output/'checkpoint_latest.pt'
    ledger = output/'checkpoint_ledger.json'
    if checkpoint.exists() and ledger.exists() and json.loads(ledger.read_text()) == charge:
        saved = load_checkpoint(checkpoint)
        if saved['step'] == 1 and saved['protocol_sha256'] == PROTOCOL:
            print('Matching local update-1 checkpoint exists; no rollback needed.')
            return
    receipt = state['objects'].get('recovery/'+TASK, {})
    if receipt.get('id') != RECOVERY_ID or receipt.get('local_sha256') != RECOVERY_SHA:
        raise RuntimeError('Recovery archive changed; refusing an unreviewed rollback.')
    if 'progress/'+TASK in state['objects']:
        raise RuntimeError('Additional progress archive needs review before recovery.')
    audit = root/'durable/recovery_audits'/EVENT
    audit.mkdir(parents=True, exist_ok=True)
    archive = audit/'verified_recovery.zip'
    transport.cloud.download(RECOVERY_ID, archive, RECOVERY_SHA)
    staged = audit/'staged'
    unpack(archive, staged)  # Validates every member before extracting.
    restored = staged/'work'/TASK
    recovered_charge = json.loads((restored/'checkpoint_ledger.json').read_text())
    saved = load_checkpoint(restored/'checkpoint_latest.pt')
    history = json.loads((restored/'training_history.json').read_text())
    if (recovered_charge != dict(step=0, protocol_sha256=PROTOCOL)
            or saved['step'] != 0 or saved['protocol_sha256'] != PROTOCOL
            or [r['update'] for r in saved['history']] != []
            or history != saved['history']
            or digest(restored/'protocol.json') != PROTOCOL):
        raise RuntimeError('Checkpoint contents differ from the reviewed update-0 state.')
    # Preserve local interruption evidence before replacing the active directory.
    if output.exists():
        output.rename(audit/('local_before_'+uuid.uuid4().hex))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(restored, output)
    record = dict(
        event=EVENT, task=TASK, time=time.time(),
        authorization='User instructed: Just restart the notebook, after the missing update and possible one-update replay were explained on 2026-10-22.',
        previous_charge=copy.deepcopy(charge), restored_checkpoint=recovered_charge,
        recovery_receipt=copy.deepcopy(receipt),
        checkpoint_sha256=digest(output/'checkpoint_latest.pt'),
        repeated_update_indices=[1],
        extra_update_attempts_upper_bound=1,
        exact_completed_lost_updates_unknown=True,
        policy='Resume restored optimizer/RNG at update 1; finish at update 25. '
               'Report repeated compute separately; this run is not strict no-replay compute.')
    record_path = audit/'recovery_audit.json'
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    transport.publish('audit/'+EVENT, [record_path])
    # Adjustment and audit marker are acknowledged in the same state write.
    state.setdefault('recovery_adjustments', {})[EVENT] = record
    state['charges'][TASK] = recovered_charge
    transport.save()
    transport.probe()
    print('RECOVERY VERIFIED: resume fold 1 / seed 2 / bands+edge+SDF at update 1; '
          'completed runs skipped; repeated compute recorded.')


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    import runtime
    runtime.activate(root)
    from parallel import SweepLock
    import replication_store
    replication_store.ACCOUNT = "filippofocaccia2@gmail.com"
    replication_store.RESULTS = "1ABU0S3cjFNozoYH4tD1DR_5aVp0z-kBz"
    from replication_store import Transport, unpack
    import store
    import torch
    with SweepLock(root):
        transport = Transport(root)
        repair(transport, unpack, store.digest,
               lambda p: torch.load(p, map_location='cpu', weights_only=False))


if __name__ == '__main__':
    main()
