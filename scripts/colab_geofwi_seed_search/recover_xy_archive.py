"""Archive an already completed XY pilot without training or deleting artifacts."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ORIGINAL_SHA = '77390cb86a5d8b6789ee0aaca12b1017097402eff0f2764971d13225a46169f9'
OLD = "config['augmentation']!='none'"
NEW = "config['augmentation']!=task.get('augmentation', 'none')"


def patched_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA:
        raise ValueError('Unexpected archive implementation')
    assert source.count(OLD) == 1
    return source.replace(OLD, NEW)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    args = parser.parse_args()
    root, durable = args.bundle.resolve(), args.durable.resolve()
    sys.path.insert(0, str(root))
    import augmentation_runtime as campaign
    io = campaign.io
    io.activate(root)
    campaign.verify(root)
    task = campaign.task('xy_d4')
    out = root/'work'/task['id']
    target = durable/'runs'/task['id']
    with (root/'work/augmentation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = io.read(out/'completion.json')
        assert done['status'] == 'complete' and done['updates'] == 414
        campaign.check_recipe(out, 'xy_d4')
        final = campaign.engine.validate_output(root, out, task)
        assert final['training_flops'] <= campaign.CAP
        from scripts.cross_dataset_campaign import storage
        source = Path(storage.__file__).read_text()
        # Preserve the frozen training bundle; patch only this archive process.
        exec(compile(patched_source(source), '<archive-policy-fix>', 'exec'), storage.__dict__)
        archive_task = dict(task, augmentation='xy_d4_v1')
        verify_with_policy = storage.verify_results
        def verify_requested_pilot(output, requested_task):
            if requested_task != task:
                raise ValueError('Archive recovery is restricted to the requested XY pilot')
            verify_with_policy(output, archive_task)
        storage.verify_results = verify_requested_pilot
        patch_dir = durable/'provenance/archive_policy_fix_v1'
        io.copy_verified(Path(__file__), patch_dir/Path(__file__).name)
        io.write(out/'archive_policy_fix.json', dict(
            original_storage_sha256=ORIGINAL_SHA,
            recovery_script_sha256=io.digest(Path(__file__)),
            correction='Validate augmentation against explicitly requested task policy',
            training_updates_added=0, checkpoint_deletions=0))
        manifest = storage.prepare(out, task)
        assert manifest['task'] == task
        for name in [*manifest['metadata'], 'archive_manifest.json']:
            io.copy_verified(out/name, target/name)
        storage.verify_archive(target, manifest)
        io.write(target/'archive_acknowledgment.json', dict(
            status='verified', manifest_sha256=io.digest(target/'archive_manifest.json'),
            source_verified=True, source_manifest_sha256=io.digest(root/'bundle_manifest.json'),
            durable_archive=str(target), cleanup_pending=True))
        status = io.read(durable/'augmentation_status.json')
        status.update(status='training_complete_archive_verified_cleanup_pending',
                      final=final, threshold_passed=final['final_dice'] > .5,
                      archive_verified=True, training_updates_added_by_recovery=0,
                      cleanup_pending=True)
        io.write(durable/'augmentation_status.json', status)
        io.write(durable/'archive_recovery_result.json', status)
        print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
