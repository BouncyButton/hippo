"""Final-matrix runner hooks for the one protected, exact Swin recovery."""
from scripts.cross_dataset_campaign.common import read
from scripts.mni_constraints import campaign
from scripts.mni_constraints import verified_matrix_recovery as recovery
from scripts.unetr_subfields import cap_extension as batch

ORIGINAL_COMMAND_FOR=campaign.command_for
ORIGINAL_RUN_ONE=batch.run_one


def command_for(stage,plan,task,base_path=None):
    command=ORIGINAL_COMMAND_FOR(stage,plan,task,base_path)
    if task['id']==recovery.TASK_ID:
        _,_,_,protocol=recovery.inputs(stage)
        index=command.index('--protocol-dir')+1
        command[index]=str(protocol)
        assert '--resume' in command
    return command


def run_one(args):
    is_recovery=args.task==recovery.TASK_ID
    if is_recovery:
        plan=read(args.stage/'plan.json')
        assert any(t['id']==recovery.TASK_ID for t in plan['tasks'])
        recovery.prepare_run(args.stage)
    # Keep the established evaluation -> durable archive ack -> cleanup path.
    # An interrupted run raises/returns without deleting its original recovery.
    ORIGINAL_RUN_ONE(args)
    if is_recovery:
        recovery.retire_original_after_archive(args.stage)


def install():
    campaign.command_for=command_for
    batch.run_one=run_one
