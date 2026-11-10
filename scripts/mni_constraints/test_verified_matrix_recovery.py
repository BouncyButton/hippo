"""Exercise interrupted-copy recovery and the deletion gate using tiny temp files."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import verified_matrix_recovery as recovery
from scripts.mni_constraints import collect_verified_matrix as collector


def rejected(call):
    try:call()
    except (AssertionError,FileNotFoundError):return
    raise AssertionError('Unsafe operation was accepted')


def main():
    original_sha=recovery.KNOWN_CHECKPOINT_SHA
    source=Path(__file__).resolve().parents[2]/'reports/mni_unetr_constraint_goal_20261108/paused_swin_recovery_metadata.json'
    try:
        with TemporaryDirectory() as temporary:
            root=Path(temporary);stage=root/'test_matrix';origin=root/recovery.OLD_NAME
            old=origin/'runs'/recovery.TASK_ID;old.mkdir(parents=True)
            checkpoint=old/'checkpoint_latest.pt';checkpoint.write_bytes(b'original zero-step checkpoint fixture')
            recovery.KNOWN_CHECKPOINT_SHA=digest(checkpoint)
            write(origin/'control.json',dict(pause=True));(old/'unrelated.txt').write_text('keep')
            protocol=stage/'recovery/protocol/protocol.json';write(protocol,dict(fixture=True))
            core=stage/recovery.CORE;core.parent.mkdir(parents=True);core.write_text('fixture core')
            data=stage/'data/MNI.pkl';data.parent.mkdir();data.write_bytes(b'fixture data')
            metadata=deepcopy(read(source));cfg=metadata['configuration']
            cfg.update(protocol_sha256=digest(protocol),script_sha256=digest(core),dataset_sha256=digest(data))
            metadata.update(checkpoint_sha256=recovery.KNOWN_CHECKPOINT_SHA,core_sha256=digest(core),
                protocol_dir=str(origin/'protocols/swinunetr_dice'))
            write(stage/'recovery/metadata.json',metadata);write(old/'config.json',cfg)
            proof=recovery.prepare_run(stage);output=stage/'runs'/recovery.TASK_ID
            assert digest(output/'checkpoint_latest.pt')==digest(checkpoint)
            assert checkpoint.exists() and proof['original_step']==0
            (output/'checkpoint_latest.pt').write_bytes(b'advanced resumed checkpoint')
            recovery.prepare_run(stage)
            assert (output/'checkpoint_latest.pt').read_bytes()==b'advanced resumed checkpoint'
            (output/'recovery_provenance.json').unlink()
            rejected(lambda:recovery.prepare_run(stage))
            assert (output/'checkpoint_latest.pt').read_bytes()==b'advanced resumed checkpoint'
            (output/'checkpoint_latest.pt').write_bytes(checkpoint.read_bytes())
            recovery.prepare_run(stage)  # Interrupted exact copy is safely adopted.
            (output/'checkpoint_latest.pt').unlink()
            write(output/'completion.json',dict(status='complete',updates=343,discarded_attempts=0,
                consumed_training_flops=cfg['schedule']['consumed_training_flops']))
            write(output/'cleanup.json',dict(status='complete'))
            for name in ('validation_latest.json','validation_best.json','training_final.json','evaluation_diagnostics.json'):
                write(output/name,dict(fixture=True))
            names=['config.json','completion.json','recovery_provenance.json','validation_latest.json',
                   'validation_best.json','training_final.json','evaluation_diagnostics.json']
            manifest=dict(metadata={name:dict(bytes=(output/name).stat().st_size,sha256=digest(output/name)) for name in names})
            write(output/'archive_manifest.json',manifest)
            rejected(lambda:recovery.retire_original_after_archive(stage));assert checkpoint.exists()
            ack=dict(status='verified',source_verified=True,manifest_sha256='wrong',durable_archive='/Users/test/reports/recovery')
            write(output/'archive_acknowledgment.json',ack)
            rejected(lambda:recovery.retire_original_after_archive(stage));assert checkpoint.exists()
            ack['manifest_sha256']=digest(output/'archive_manifest.json');ack['durable_archive']='/mnt/beegfs/not_durable'
            write(output/'archive_acknowledgment.json',ack)
            rejected(lambda:recovery.retire_original_after_archive(stage));assert checkpoint.exists()
            ack['durable_archive']='/Users/test/reports/recovery';write(output/'archive_acknowledgment.json',ack)
            completed=read(output/'completion.json')
            for field,value in (('updates',342),('discarded_attempts',1),
                                ('consumed_training_flops',recovery.CAP+1)):
                invalid=deepcopy(completed);invalid[field]=value
                write(output/'completion.json',invalid)
                rejected(lambda:recovery.retire_original_after_archive(stage));assert checkpoint.exists()
            write(output/'completion.json',completed)
            incomplete=deepcopy(manifest)
            del incomplete['metadata']['evaluation_diagnostics.json']
            write(output/'archive_manifest.json',incomplete)
            ack['manifest_sha256']=digest(output/'archive_manifest.json')
            write(output/'archive_acknowledgment.json',ack)
            rejected(lambda:recovery.retire_original_after_archive(stage));assert checkpoint.exists()
            write(output/'archive_manifest.json',manifest)
            ack['manifest_sha256']=digest(output/'archive_manifest.json')
            write(output/'archive_acknowledgment.json',ack)
            original=checkpoint.read_bytes();checkpoint.write_bytes(b'changed protected recovery')
            rejected(lambda:recovery.retire_original_after_archive(stage));assert checkpoint.read_bytes()==b'changed protected recovery'
            checkpoint.write_bytes(original);write(origin/'control.json',dict(pause=False))
            rejected(lambda:recovery.retire_original_after_archive(stage));assert checkpoint.exists()
            write(origin/'control.json',dict(pause=True))
            receipt=recovery.retire_original_after_archive(stage)
            assert not checkpoint.exists() and receipt['released_file_bytes']==len(original)
            assert (old/'unrelated.txt').read_text()=='keep' and (old/'config.json').exists()
            assert recovery.retire_original_after_archive(stage)==receipt
            assert recovery.prepare_run(stage)==read(output/'recovery_provenance.json')
    finally:recovery.KNOWN_CHECKPOINT_SHA=original_sha
    old_snapshot,old_remote=collector.ORIGINAL_SNAPSHOT,collector.collect.remote
    try:
        collector.ORIGINAL_SNAPSHOT=lambda stage:{'status.json':{'status':'running'}}
        collector.collect.remote=lambda args:b'null'
        assert 'recovery_resolution.json' not in collector.snapshot(Path('/fixture'))
        collector.collect.remote=lambda args:json.dumps({'status':'resolved_after_verified_resume'}).encode()
        assert collector.snapshot(Path('/fixture'))['recovery_resolution.json']['status']=='resolved_after_verified_resume'
    finally:collector.ORIGINAL_SNAPSHOT,collector.collect.remote=old_snapshot,old_remote
    print('Passed: exact copy, advanced-resume preservation, interrupted-copy adoption, archive/durability/hash/pause deletion gates, narrow/idempotent retirement, collector receipt mirroring. Real recovery untouched.')


if __name__=='__main__':main()
