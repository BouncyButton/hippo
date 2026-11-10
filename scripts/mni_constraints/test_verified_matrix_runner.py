"""Check exact recovery routing and ordering without starting training."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import write
from scripts.mni_constraints import verified_matrix_runner as runner


def main():
    original=(runner.ORIGINAL_COMMAND_FOR,runner.ORIGINAL_RUN_ONE,
              runner.recovery.inputs,runner.recovery.prepare_run,
              runner.recovery.retire_original_after_archive)
    calls=[]
    try:
        runner.ORIGINAL_COMMAND_FOR=lambda *args:['python','-m','matrix','train',
            '--protocol-dir','new/protocol','--resume']
        runner.recovery.inputs=lambda stage:(None,None,None,stage/'recovery/protocol')
        task={'id':runner.recovery.TASK_ID}
        assert runner.command_for(Path('/stage'),{},task)[5]=='/stage/recovery/protocol'
        ordinary=runner.command_for(Path('/stage'),{},dict(id='MNI_nnunet_dice_f0s0'))
        assert ordinary[5]=='new/protocol'
        with TemporaryDirectory() as temporary:
            stage=Path(temporary);write(stage/'plan.json',dict(tasks=[task]))
            args=SimpleNamespace(stage=stage,task=task['id'])
            runner.recovery.prepare_run=lambda stage:calls.append('prepare')
            runner.ORIGINAL_RUN_ONE=lambda args:calls.append('train_evaluate_archive_cleanup')
            runner.recovery.retire_original_after_archive=lambda stage:calls.append('retire_original')
            runner.run_one(args)
            assert calls==['prepare','train_evaluate_archive_cleanup','retire_original']
            calls.clear()
            def interrupted(args):
                calls.append('interrupted');raise SystemExit(75)
            runner.ORIGINAL_RUN_ONE=interrupted
            try:runner.run_one(args)
            except SystemExit as error:assert error.code==75
            else:raise AssertionError('Expected interrupted recovery')
            assert calls==['prepare','interrupted']
            calls.clear();runner.ORIGINAL_RUN_ONE=lambda args:calls.append('ordinary')
            runner.run_one(SimpleNamespace(stage=stage,task='MNI_nnunet_dice_f0s0'))
            assert calls==['ordinary']
    finally:
        (runner.ORIGINAL_COMMAND_FOR,runner.ORIGINAL_RUN_ONE,runner.recovery.inputs,
         runner.recovery.prepare_run,runner.recovery.retire_original_after_archive)=original
    print('Passed: ordinary command unchanged; exact recovery protocol/resume routing; archive-before-retirement ordering; interruption preserves original. No training executed.')


if __name__=='__main__':main()
