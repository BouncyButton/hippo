"""Idempotently install a scheduler alongside the original scientific package."""


def stage_parallel(package, data_zip, base_file, bundle, controller, results,
                   package_sha, manifest_sha, data_sha, base_sha):
    import fcntl
    import hashlib
    import json
    from pathlib import Path
    import shutil
    import zipfile

    package, data_zip, base_file, bundle, controller, results = map(
        Path, (package, data_zip, base_file, bundle, controller, results))

    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
        return h.hexdigest()

    def safe(name):
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe package path')
        return p

    def install_bytes(target, value, expected):
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError('Package member hash mismatch: '+str(target))
        if target.is_symlink():
            raise ValueError('Refusing to overwrite symlink: '+str(target))
        if target.exists():
            if sha(target) != expected:
                raise ValueError('Existing file differs; preserved without replacement: '+str(target))
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name+'.staging')
        temp.write_bytes(value)
        temp.replace(target)

    if sha(package) != package_sha:
        raise ValueError('Upload hash mismatch: '+package.name)
    (bundle/'work').mkdir(parents=True, exist_ok=True)
    with (bundle/'work/sweep.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A sweep is active. Stop the old Run cell once and wait for its recovery upload before staging.') from None
        existing = bundle/'bundle_manifest.json'
        if existing.exists() and sha(existing) != manifest_sha:
            raise ValueError('Existing bundle identity differs. No work has been replaced.')
        with zipfile.ZipFile(package) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate package members')
            for info in archive.infolist():
                safe(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlink in package')
            package_manifest = json.loads(archive.read('package_manifest.json'))
            if set(names) != set(package_manifest)|{'package_manifest.json'}:
                raise ValueError('Package file list differs')
            payload = archive.read('bundle/bundle_manifest.json')
            if hashlib.sha256(payload).hexdigest() != manifest_sha:
                raise ValueError('Scientific manifest differs')
            # Verify every existing scientific file before writing anything.
            for name, expected in package_manifest.items():
                part = safe(name)
                if part.parts[0] not in ('bundle', 'controller'):
                    raise ValueError('Unexpected package tree')
                target = (bundle if part.parts[0] == 'bundle' else controller)/Path(*part.parts[1:])
                if target.exists() and (target.is_symlink() or sha(target) != expected):
                    raise ValueError('Existing file differs; preserved: '+str(target))
            for name, expected in package_manifest.items():
                part = safe(name)
                target = (bundle if part.parts[0] == 'bundle' else controller)/Path(*part.parts[1:])
                install_bytes(target, archive.read(name), expected)
        data_manifest = json.loads((bundle/'data_manifest.json').read_text())
        missing = [name for name in data_manifest if not (bundle/'data'/safe(name)).exists()]
        for name, expected in data_manifest.items():
            path = bundle/'data'/safe(name)
            if path.exists() and sha(path) != expected:
                raise ValueError('Existing data changed: '+name)
        if missing:
            if sha(data_zip) != data_sha:
                raise ValueError('Data ZIP upload differs')
            with zipfile.ZipFile(data_zip) as archive:
                for name in missing:
                    install_bytes(bundle/'data'/safe(name), archive.read('data/'+name), data_manifest[name])
        base = bundle/'weights/sam3.pt'
        if base.exists():
            if sha(base) != base_sha:
                raise ValueError('Existing base checkpoint differs; preserved.')
        else:
            if shutil.disk_usage(bundle).free < 12*1024**3:
                raise RuntimeError('Need 12 GiB runtime disk space to stage the base checkpoint.')
            base.parent.mkdir(parents=True, exist_ok=True)
            temp = base.with_name('sam3.pt.staging')
            shutil.copyfile(base_file, temp)
            if sha(temp) != base_sha:
                raise ValueError('Base checkpoint copy differs')
            temp.replace(base)
        results.mkdir(parents=True, exist_ok=True)
        (results/'parallel_package.json').write_text(json.dumps(dict(
            package_sha256=package_sha, original_manifest_sha256=manifest_sha,
            science_source_changed=False, existing_results_preserved=True), indent=2))
    print('Original bundle and work preserved. Parallel supervisor staged separately.', flush=True)
