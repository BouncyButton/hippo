"""Use mounted local NVMe for new work without moving existing run artifacts."""
import json
from pathlib import Path
import shutil


GIB = 1024**3


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def choose_work_store(root, durable, scientific_sha, mount=Path('/mnt/local-scratch')):
    """Called under the unchanged sweep lock; mount absence never creates a fake SSD."""
    mount = Path(mount)
    if not mount.is_mount() or mount.stat().st_dev == root.stat().st_dev:
        print('STORAGE local-scratch is unavailable; new work stays on the runtime disk.', flush=True)
        return None
    if shutil.disk_usage(mount).free < 12*GIB:
        raise RuntimeError('Local-scratch has less than 12 GiB free; existing work preserved')
    store = mount/('hippo_msd_'+scientific_sha[:16])
    store.mkdir(exist_ok=True)
    owner = dict(bundle=str(root.resolve()), durable=str(durable.resolve()),
                 scientific_bundle_sha256=scientific_sha)
    marker = store/'owner.json'
    if marker.exists():
        if json.loads(marker.read_text()) != owner:
            raise RuntimeError('NVMe work belongs to a different bundle/results directory; preserved')
    elif any(store.iterdir()):
        raise RuntimeError('NVMe work exists without its ownership record; preserved for inspection')
    else:
        atomic_json(marker, owner)
    (store/'work').mkdir(exist_ok=True)
    print('STORAGE new runs use local NVMe:', store/'work',
          '| existing run directories remain in place; Drive recovery/accounting unchanged.', flush=True)
    return store/'work'


def output_location(root, task, store):
    """Use logical paths for compatibility, leaving the controller lock in place."""
    identity = task['id']
    if Path(identity).name != identity or identity in ('.', '..'):
        raise ValueError('Invalid task identity')
    output = root/'work'/identity
    if output.is_symlink():
        if store is None or not output.exists() or output.resolve() != (store/identity).resolve():
            raise RuntimeError('Existing work storage is unavailable or unexpected; preserve it: '+identity)
        return output.resolve()
    if output.exists():
        if not output.is_dir():
            raise RuntimeError('Expected a run directory; existing path preserved: '+identity)
        return output
    return store/identity if store is not None else output


def ensure_output(root, task, store):
    """Create only a missing run path; never move, replace or empty existing work."""
    output = root/'work'/task['id']
    target = output_location(root, task, store)
    if target != output and not output.is_symlink():
        if target.is_symlink():
            raise RuntimeError('NVMe run target is a symlink; preserve it for inspection')
        target.mkdir(exist_ok=True)
        # A killed setup may leave this owned target before linking it. Reuse its
        # contents, which will still pass the normal charge/checkpoint audit.
        output.symlink_to(target, target_is_directory=True)
    return output


def record_location(output, store):
    # Once an archive manifest exists, its metadata inventory must remain frozen.
    if (output/'archive_manifest.json').exists():
        return
    path = output/'local_storage.json'
    if not path.exists():
        atomic_json(path, dict(logical_output=str(output), physical_output=str(output.resolve()),
            backend='local_nvme' if store is not None and output.is_symlink() else 'existing_runtime_disk',
            existing_work_moved=False, checkpoint_format_changed=False,
            checkpoint_cadence_changed=False, durable_accounting_changed=False))
