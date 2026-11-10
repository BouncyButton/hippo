"""Audit real archived cells for conditional reuse; do not fabricate a winner."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import shutil

from scripts.cross_dataset_campaign.common import read,write,digest,ARMS,REMOTE
from scripts.mni_constraints import verified_unetr_full_matrix as matrix


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',action='store_true')
    variants=parser.add_mutually_exclusive_group()
    variants.add_argument('--diceonly',action='store_true');variants.add_argument('--ce025',action='store_true')
    variants.add_argument('--ce025-cosine',action='store_true')
    variants.add_argument('--ce050-cosine',action='store_true')
    variants.add_argument('--lower-lr',action='store_true')
    parser.add_argument('--recipe-key',choices=('lr045_cosine','lr030_cosine'))
    args=parser.parse_args()
    if args.diceonly:
        from scripts.mni_constraints import verified_diceonly_full_matrix
        assert verified_diceonly_full_matrix.matrix is matrix
    if args.ce025:
        from scripts.mni_constraints import verified_ce025_full_matrix
        assert verified_ce025_full_matrix.matrix is matrix
    if args.ce025_cosine:
        from scripts.mni_constraints import verified_ce025_cosine_full_matrix
        assert verified_ce025_cosine_full_matrix.matrix is matrix
    if args.ce050_cosine:
        from scripts.mni_constraints import verified_ce050_cosine_full_matrix
        assert verified_ce050_cosine_full_matrix.matrix is matrix
    if args.lower_lr:
        from scripts.mni_constraints import verified_lower_lr_full_matrix
        assert verified_lower_lr_full_matrix.matrix is matrix
    assert args.recipe_key is None or args.lower_lr
    result=read(matrix.ROOT/'reports'/matrix.confirmation.NAME/matrix.RESULT_FILENAME)
    if result['status']!='complete' or result['outcome']!='nine_pair_success':
        try:matrix.gate_from_report()
        except AssertionError:pass
        else:raise AssertionError('Unconfirmed gate accepted')
    recipe=matrix.confirmation.RECIPES[0] if args.diceonly or args.ce025 or args.ce025_cosine or args.ce050_cosine or args.lower_lr else matrix.recipe_helper.COSINE_RECIPE
    if args.recipe_key:
        recipe=next(r for r in matrix.confirmation.RECIPES if r['key']==args.recipe_key)
    tasks=matrix.matrix(recipe);assert len(tasks)==504
    assert sum(t['model']=='unetr_vit_b16' and t['augmentation']=='none' for t in tasks)==126
    assert sum(t['model']=='swinunetr' and t['augmentation']=='xy_d4_v1' for t in tasks)==126
    parent=matrix.ROOT/'experiments'/matrix.confirmation.NAME
    original=matrix.ROOT/'experiments'/matrix.legacy.ORIGINAL
    available={r['id']:r for r in result['archived_runs']}
    checked=[]
    with TemporaryDirectory() as temporary:
        stage=Path(temporary)
        (stage/'source').symlink_to(parent/'source',target_is_directory=True)
        (stage/'data').symlink_to(parent/'data',target_is_directory=True)
        for model in matrix.legacy.MODELS:
            for arm in ARMS:
                key=model+'_'+arm;folder=stage/'protocols'/key
                shutil.copytree(original/'protocols'/key,folder)
                old=read(folder/'protocol.json')
                if model=='unetr_vit_b16':
                    shutil.copy2(parent/'profile_evidence'/matrix.PROFILE_FILENAME,folder/'profile.json')
                write(folder/'protocol.json',matrix.recipe_helper.changed_protocol(old,read(folder/'profile.json'),
                    recipe,stage_name=matrix.NAME,profile_sha256=digest(folder/'profile.json')))
        from scripts.mni_constraints.nnunet_repair import runtime_source
        repaired=matrix.ROOT/'experiments'/matrix.legacy.NN_CONFIRM/matrix.recovery.CORE
        assert repaired.read_text()==runtime_source((stage/matrix.recovery.CORE).read_text())
        write(stage/'nnunet_reuse_equivalence.json',dict(original_sha256=digest(stage/matrix.recovery.CORE),
            legacy_sha256=digest(repaired)))
        (stage/'reused_runs').mkdir();(stage/'reuse_protocols').mkdir()
        for task in tasks:
            if task['model']=='unetr_vit_b16':
                id=matrix.confirmation.identifier(recipe['key'],task['arm'],task['seed'],task['fold'])
                if id not in available:continue
                origin=matrix.ROOT/'reports'/matrix.confirmation.NAME/'runs'/id
                protocol=stage/'reuse_protocols'/task['protocol']
                if not protocol.exists():
                    protocol.symlink_to(parent/'protocols'/(recipe['key']+'_'+task['arm']),target_is_directory=True)
            else:
                origin=matrix.legacy.reuse_origin(task)
                if origin is None:continue
            (stage/'reused_runs'/task['id']).symlink_to(origin,target_is_directory=True)
            entry=dict(task=task,source_archive=str(origin),manifest_sha256=digest(origin/'archive_manifest.json'))
            matrix.verify_reused(stage,entry);checked.append(task['id'])
        (stage/'recovery').mkdir()
        shutil.copy2(matrix.ROOT/'reports/mni_unetr_constraint_goal_20261108/paused_swin_recovery_metadata.json',stage/'recovery/metadata.json')
        shutil.copytree(matrix.ROOT/'experiments'/matrix.legacy.PARENT/'protocols/swinunetr_dice',stage/'recovery/protocol')
        matrix.recovery.inputs(stage,expected_origin=REMOTE/matrix.recovery.OLD_NAME)
        if args.runtime:
            evidence=matrix.check_runtime(stage)
            evidence.update(scope='Conditional scratch protocols with actual archived evidence; no nine-pair gate claim, no stage freeze or submission.',
                winner_gate_passed=False,source_sha256=digest(Path(matrix.__file__)))
            filename=(f"conditional_{recipe['key']}_matrix_runtime_preflight.json" if args.lower_lr else
                'conditional_ce050_cosine_matrix_runtime_preflight.json' if args.ce050_cosine else
                'conditional_ce025_cosine_matrix_runtime_preflight.json' if args.ce025_cosine else
                'conditional_ce025_matrix_runtime_preflight.json' if args.ce025 else
                'conditional_diceonly_matrix_runtime_preflight.json' if args.diceonly else
                'conditional_matrix_runtime_preflight.json')
            write(matrix.ROOT/'reports/mni_unetr_constraint_goal_20261108'/filename,evidence)
    assert len([id for id in checked if id.startswith('MNI_nnunet_')])==9
    print(f'Checked all504 logical cells/56protocols and {len(checked)} real archive reuse candidates, including all9 repaired NN controls; frozen recovery source/data/protocol match. No stage built, checkpoint touched or job submitted.')


if __name__=='__main__':main()
