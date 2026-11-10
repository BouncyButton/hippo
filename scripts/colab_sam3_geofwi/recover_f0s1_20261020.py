"""Explicit, one-time recovery of the user-approved October 20 interruption."""
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

TASK = 'GeoFWI3D_SAM3_f0s1_dice_bands_edge_sdf_u25_v1'
EVENT = '20261020_f0s1_bands_edge_sdf_resume5'
PROTOCOL = '61daeba9674030acef62ee3235fa48a912334eb4d21abf71cd08b94f9c9822b5'
RECOVERY_ID = '1o51LRQEktLXndXkBlfll0swD28U4y6Nh'
RECOVERY_SHA = 'd42cc01ba8afebd388445a79bde6940af61952239a9fa375a10f8efe05f0362f'


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
    if charge != dict(step=10, protocol_sha256=PROTOCOL):
        raise RuntimeError('Recovery state changed; refusing an unreviewed rollback.')
    root = transport.root
    output = root/'work'/TASK
    checkpoint = output/'checkpoint_latest.pt'
    ledger = output/'checkpoint_ledger.json'
    if checkpoint.exists() and ledger.exists() and json.loads(ledger.read_text()) == charge:
        saved = load_checkpoint(checkpoint)
        if saved['step'] == 10 and saved['protocol_sha256'] == PROTOCOL:
            print('Matching local update-10 checkpoint exists; no rollback needed.')
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
    if (recovered_charge != dict(step=5, protocol_sha256=PROTOCOL)
            or saved['step'] != 5 or saved['protocol_sha256'] != PROTOCOL
            or [r['update'] for r in saved['history']] != list(range(1, 6))
            or history != saved['history']
            or digest(restored/'protocol.json') != PROTOCOL):
        raise RuntimeError('Checkpoint contents differ from the reviewed update-5 state.')
    # Preserve local interruption evidence before replacing the active directory.
    if output.exists():
        output.rename(audit/('local_before_'+uuid.uuid4().hex))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(restored, output)
    record = dict(
        event=EVENT, task=TASK, time=time.time(),
        authorization='User approved recovery after explanation in Codex on 2026-10-20.',
        previous_charge=copy.deepcopy(charge), restored_checkpoint=recovered_charge,
        recovery_receipt=copy.deepcopy(receipt),
        checkpoint_sha256=digest(output/'checkpoint_latest.pt'),
        repeated_update_indices=[6, 7, 8, 9, 10],
        extra_update_attempts_upper_bound=5,
        exact_completed_lost_updates_unknown=True,
        policy='Resume restored optimizer/RNG at update 6; finish at update 25. '
               'Report repeated compute separately; this run is not strict no-replay compute.')
    record_path = audit/'recovery_audit.json'
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    transport.publish('audit/'+EVENT, [record_path])
    # Adjustment and audit marker are acknowledged in the same state write.
    state.setdefault('recovery_adjustments', {})[EVENT] = record
    state['charges'][TASK] = recovered_charge
    transport.save()
    transport.probe()
    print('RECOVERY VERIFIED: resume fold 0 / seed 1 / bands+edge+SDF at update 6; '
          'completed runs skipped; repeated compute recorded.')


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    import runtime
    runtime.activate(root)
    from parallel import SweepLock
    from replication_store import Transport, unpack
    import store
    import torch
    with SweepLock(root):
        transport = Transport(root)
        repair(transport, unpack, store.digest,
               lambda p: torch.load(p, map_location='cpu', weights_only=False))


if __name__ == '__main__':
    main()
