"""Scheduling guards for the ADNI follow-on; no real scheduler calls."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.cross_dataset_campaign import adni_mixed_losses_job as job
from scripts.cross_dataset_campaign.common import write, digest


def completed_predecessor(tmp_path, monkeypatch):
    stage=tmp_path/'msd';stage.mkdir()
    write(stage/'manifest.json',{'frozen':'fixture'})
    write(stage/'control.json',dict(pause=False))
    write(stage/'status.json',dict(status='complete',completed=56))
    tasks=[dict(id=f'run_{i}') for i in range(56)]
    monkeypatch.setattr(job.msd,'verify',lambda _:dict(tasks=tasks))
    sha=digest(stage/'manifest.json')
    for task in tasks:
        root=stage/'runs'/task['id']
        write(root/'completion.json',dict(status='complete'))
        write(root/'cleanup.json',dict(status='complete'))
        write(root/'archive_manifest.json',dict(metadata={}))
        write(root/'archive_acknowledgment.json',dict(status='verified',source_verified=True,
            manifest_sha256=digest(root/'archive_manifest.json'),source_manifest_sha256=sha))
    return dict(predecessor=dict(stage=str(stage),name=job.msd.NAME,manifest_sha256=sha)),stage


def test_start_requires_every_msd_archive_and_cleanup(tmp_path,monkeypatch):
    plan,stage=completed_predecessor(tmp_path,monkeypatch)
    assert job.predecessor_complete(plan)
    write(stage/'runs/run_55/cleanup.json',dict(status='in_progress'))
    with pytest.raises(ValueError,match='incomplete'):job.predecessor_complete(plan)


def test_archive_hash_mismatch_blocks_followon(tmp_path,monkeypatch):
    plan,stage=completed_predecessor(tmp_path,monkeypatch)
    write(stage/'runs/run_0/archive_manifest.json',dict(metadata={'changed':True}))
    with pytest.raises(ValueError,match='incomplete'):job.predecessor_complete(plan)


def test_paused_or_partial_msd_never_submits(tmp_path,monkeypatch):
    plan,stage=completed_predecessor(tmp_path,monkeypatch)
    write(stage/'control.json',dict(pause=True))
    assert not job.predecessor_complete(plan)
    write(stage/'control.json',dict(pause=False))
    write(stage/'status.json',dict(status='running',completed=55))
    monkeypatch.setattr(job,'verify',lambda _:plan)
    submit=Mock();monkeypatch.setattr(job,'BASE_SUBMIT',submit)
    with pytest.raises(RuntimeError,match='Wait for all56'):job.submit(tmp_path)
    submit.assert_not_called()


def test_repeated_start_reuses_own_job(tmp_path,monkeypatch):
    monkeypatch.setattr(job,'verify',lambda _:dict())
    monkeypatch.setattr(job,'predecessor_complete',lambda _:True)
    monkeypatch.setattr(job.controller,'queue',lambda:[['123','RUNNING','hippo-adni']])
    receipt=dict(job_id='123',manifest_sha256='fixture')
    write(tmp_path/'submissions/123.json',receipt)
    submit=Mock();monkeypatch.setattr(job,'BASE_SUBMIT',submit)
    assert job.submit(tmp_path)==receipt
    submit.assert_not_called()


def test_command_keeps_resume_and_uses_adni_only(tmp_path):
    write(tmp_path/'protocols/nnunet/protocol.json',dict(trainer_alias='nnunet',data='/adni.pkl'))
    task=dict(protocol='nnunet',id='baseline',kind='conventional',arm='dice',fold=0,seed=0)
    command=job.command_for(tmp_path,dict(conventional_python='/science/python'),task,Path('/unused'))
    assert command[:3]==['/science/python','-m','scripts.cross_dataset_campaign.adni_mixed_losses']
    assert command[-1]=='--resume'
    assert command[command.index('--fold')+1]=='0'
    assert command[command.index('--seed')+1]=='0'
