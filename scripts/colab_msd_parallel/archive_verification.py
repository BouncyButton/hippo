"""Stage each Drive archive once, then apply the unchanged scientific verifier."""
from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def verified_snapshot(runtime, durable, local_parent, task, binding, progress=None, check=None):
    """Yield a fully verified local results root, or None for unfinished cleanup.

    Every invocation reads fresh Drive bytes. Only this invocation's disposable
    local copy is removed on exit; durable archives and recovery are read-only.
    The caller must consume result rows/profiles before leaving the context.
    """
    task_id = task['id']
    if Path(task_id).name != task_id or task_id in ('', '.', '..'):
        raise ValueError('Unsafe archive task ID: '+task_id)
    source = durable/'runs'/task_id
    local_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='archive_', dir=local_parent) as temporary:
        local = Path(temporary)
        target = local/'runs'/task_id
        target.mkdir(parents=True)
        runtime.write(local/'environment_binding.json', binding)
        copied = copied_bytes = 0

        def update(**fields):
            if progress is not None:
                progress(announce=False, **fields)

        def copy(name):
            nonlocal copied, copied_bytes
            if check is not None:
                check()
            update(current_file=name)
            destination = target/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # No source hash pass or readback on Drive: the frozen verifier below
            # hashes and parses these exact bytes locally, with all original gates.
            size = 0
            with (source/name).open('rb') as reader, destination.open('wb') as writer:
                while block := reader.read(8*1024**2):
                    writer.write(block)
                    size += len(block)
            copied += 1
            copied_bytes += size
            update(progress=True, archive_files_copied=copied, archive_bytes_copied=copied_bytes)
            if check is not None:
                check()

        update(current_task=task_id, current_file=None, archive_files_copied=0,
               archive_files_total=None, archive_bytes_copied=0, archive_operation='reading Drive archive')
        try:
            copy('cleanup.json')
        except FileNotFoundError:
            yield None
            return
        if runtime.read(target/'cleanup.json')['status'] != 'complete':
            yield None
            return
        copy('archive_manifest.json')
        manifest = runtime.read(target/'archive_manifest.json')
        names = manifest['metadata']
        reserved = {'cleanup.json', 'archive_manifest.json', 'archive_acknowledgment.json'}
        for name in names:
            relative = Path(name)
            if (not relative.parts or relative.is_absolute() or '..' in relative.parts
                    or relative.as_posix() != name or relative.parts[0] in reserved):
                raise ValueError('Unsafe archive member: '+name)
        update(archive_files_total=len(names)+3)
        for name in names:
            copy(name)
        copy('archive_acknowledgment.json')
        update(current_file=None, archive_operation='checking hashes and readability locally')
        if not runtime.completed(local, task):
            raise ValueError('Archive proof did not verify: '+task_id)
        if check is not None:
            check()
        update(progress=True, archive_operation='verified locally')
        yield local
