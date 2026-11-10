"""Conditional exact resume of the protected zero-step Swin baseline.

The final-matrix builder must freeze recovery/metadata.json and the byte-exact
old protocol under recovery/protocol. These helpers are not used by live UNETR
studies. The caller routes this task's training command to that old protocol.
"""
import fcntl
import os
from pathlib import Path
import shutil
import time

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive

OLD_NAME='mni_common0519_shared53t_3fold_20261108'
TASK_ID='MNI_swinunetr_dice_f1s19'
KNOWN_CHECKPOINT_SHA='892147f16c4da309e50069af8f5370dbe7cd5cc53a13a93c435cd2faf795de4e'
CORE='source/scripts/cross_dataset_campaign/capped_core.py'
CAP=53305109592450


def inputs(stage: Path, *, expected_origin: Path | None = None):
    metadata=read(stage/'recovery/metadata.json');cfg=metadata['configuration']
    # Preparation can verify a local frozen copy against its remote origin.
    # Runtime callers omit this override and resolve the actual sibling stage.
    origin=expected_origin if expected_origin is not None else stage.parent/OLD_NAME
    protocol=stage/'recovery/protocol'
    assert metadata['status']=='read_only_paused_recovery_verified'
    assert metadata['task']['id']==TASK_ID
    assert metadata['step']==metadata['charged_attempts']==0
    assert metadata['checkpoint_sha256']==KNOWN_CHECKPOINT_SHA
    assert Path(metadata['protocol_dir']).resolve()==(origin/'protocols/swinunetr_dice').resolve()
    assert (cfg['model_identity'],cfg['objective'],cfg['seed'],cfg['fold'])==('swinunetr','dice',19,1)
    assert cfg['initial_lr']==.0003 and cfg['augmentation']=='xy_d4_v1'
    assert cfg['budget_flops']==CAP and cfg['max_steps']==cfg['schedule']['max_steps']==343
    assert cfg['lr_horizon_updates']==343
    assert digest(protocol/'protocol.json')==cfg['protocol_sha256']
    assert digest(stage/CORE)==metadata['core_sha256']==cfg['script_sha256']
    assert digest(stage/'data/MNI.pkl')==cfg['dataset_sha256']
    return metadata,cfg,origin,protocol


def prepare_run(stage: Path):
    """Copy the original recovery once; never overwrite resumed progress."""
    metadata,cfg,origin,protocol=inputs(stage)
    output=stage/'runs'/TASK_ID;output.mkdir(parents=True,exist_ok=True)
    with (output/'recovery_prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (output/'cleanup.json').exists():
            assert read(output/'cleanup.json')['status']=='complete'
            return read(output/'recovery_provenance.json')
        marker=output/'recovery_provenance.json'
        if marker.exists():
            provenance=read(marker)
            assert provenance['original_checkpoint_sha256']==KNOWN_CHECKPOINT_SHA
            assert read(output/'config.json')==cfg
            # An interrupted resumed checkpoint can have advanced. The trainer
            # validates its configuration/RNG/ledger; do not replace it with step0.
            assert (output/'checkpoint_latest.pt').exists()
            return provenance
        assert read(origin/'control.json')['pause']
        old=origin/'runs'/TASK_ID
        assert read(old/'config.json')==cfg
        source=old/'checkpoint_latest.pt'
        assert digest(source)==KNOWN_CHECKPOINT_SHA
        assert not (output/'completion.json').exists(), 'Unclassified completion must be preserved'
        if (output/'config.json').exists():assert read(output/'config.json')==cfg
        if (output/'budget_ledger.json').exists():
            assert read(output/'budget_ledger.json')['charged_attempts']==0
        if (output/'checkpoint_latest.pt').exists():
            # Recover a crash between the atomic copy and provenance write.
            # An advanced or otherwise different checkpoint is never overwritten.
            assert digest(output/'checkpoint_latest.pt')==KNOWN_CHECKPOINT_SHA
        for name in ('config.json','checkpoint_latest.pt'):
            temporary=output/(name+'.recovery_tmp')
            shutil.copy2(old/name,temporary)
            assert digest(temporary)==digest(old/name)
            if name=='checkpoint_latest.pt':assert digest(temporary)==KNOWN_CHECKPOINT_SHA
            else:assert read(temporary)==cfg
            os.replace(temporary,output/name)
        assert read(origin/'control.json')['pause']
        provenance=dict(status='exact_zero_step_recovery_copied',task_id=TASK_ID,
            original_stage=str(origin),original_checkpoint_sha256=KNOWN_CHECKPOINT_SHA,
            original_checkpoint_bytes=source.stat().st_size,original_step=0,original_charged_attempts=0,
            protocol_sha256=cfg['protocol_sha256'],metadata_sha256=digest(stage/'recovery/metadata.json'),
            copied_at=time.time(),retirement_policy='Keep original until this resumed run completes required evaluations, durable archive acknowledgment and checkpoint cleanup.')
        write(marker,provenance)
        return provenance


def retire_original_after_archive(stage: Path):
    """Retire only the exact original checkpoint, after verified resumed completion."""
    metadata,cfg,origin,_=inputs(stage)
    output=stage/'runs'/TASK_ID
    provenance=read(output/'recovery_provenance.json')
    assert provenance['original_checkpoint_sha256']==KNOWN_CHECKPOINT_SHA
    assert provenance['metadata_sha256']==digest(stage/'recovery/metadata.json')
    assert provenance['original_stage']==str(origin)
    assert read(output/'config.json')==cfg
    completed=read(output/'completion.json')
    assert completed['status']=='complete' and completed['updates']==cfg['max_steps']
    assert completed['discarded_attempts']==0 and completed['consumed_training_flops']<=CAP
    assert read(output/'cleanup.json')['status']=='complete'
    assert not list(output.glob('*.pt'))
    manifest=read(output/'archive_manifest.json');ack=read(output/'archive_acknowledgment.json')
    assert ack['status']=='verified' and ack['source_verified']
    assert ack['manifest_sha256']==digest(output/'archive_manifest.json')
    assert not ack['durable_archive'].startswith(('/mnt/beegfs','/home/3160552','/tmp'))
    required={'config.json','completion.json','recovery_provenance.json','validation_latest.json',
              'validation_best.json','training_final.json','evaluation_diagnostics.json'}
    assert required<=set(manifest['metadata'])
    verify_archive(output,manifest)
    assert read(origin/'control.json')['pause']
    assert read(origin/'runs'/TASK_ID/'config.json')==cfg
    checkpoint=origin/'runs'/TASK_ID/'checkpoint_latest.pt'
    receipt=stage/'recovery_resolution.json'
    with (stage/'recovery_resolution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            previous=read(receipt)
            assert previous['original_checkpoint_sha256']==KNOWN_CHECKPOINT_SHA
            assert previous['resumed_archive_sha256']==digest(output/'archive_manifest.json')
            assert not checkpoint.exists()
            return previous
        released=0
        if checkpoint.exists():
            assert not checkpoint.is_symlink()
            assert digest(checkpoint)==KNOWN_CHECKPOINT_SHA
            released=checkpoint.stat().st_size
            checkpoint.unlink()
        resolution=dict(status='resolved_after_verified_resume',task_id=TASK_ID,
            original_checkpoint=str(checkpoint),original_checkpoint_sha256=KNOWN_CHECKPOINT_SHA,
            released_file_bytes=released,original_already_absent=released==0,
            resumed_archive_sha256=digest(output/'archive_manifest.json'),
            durable_resumed_archive=ack['durable_archive'],resolved_at=time.time())
        write(receipt,resolution)
        return resolution
