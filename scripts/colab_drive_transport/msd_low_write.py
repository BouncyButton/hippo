"""Reuse server-verified immutable objects and stable ZIP bytes; charges untouched."""
import hashlib
import json
import os
from pathlib import Path
import zipfile


def cache_file(folder, name):
    config = os.environ.get('HIPPO_CLOUD_CONFIG')
    if not config: return None
    local = Path(json.loads(Path(config).read_text())['local_root'])/'immutable_upload_index'
    key = hashlib.sha256((folder+'\0'+name).encode()).hexdigest()
    return local/(key+'.json')


def find_verified(cloud, path, name):
    cache = cache_file(cloud.folder, name)
    if cache is None or not cache.exists(): return None
    saved = json.loads(cache.read_text())
    try: metadata = cloud.metadata(saved['id'])
    except Exception as exc:
        if getattr(getattr(exc,'resp',None),'status',None) == 404:
            cache.unlink()
            return None
        raise
    if (metadata['name'] != name or metadata.get('parents') != [cloud.folder]
            or not cloud.matches(path, metadata)):
        raise ValueError('Previously verified immutable Drive object changed')
    from store import digest
    return dict(metadata, local_sha256=digest(path))


def remember(cloud, name, receipt):
    cache = cache_file(cloud.folder, name)
    if cache is not None:
        from store import atomic
        atomic(cache, dict(id=receipt['id']))
    return receipt


def stable_zip(output, manifest):
    """Byte-identical metadata stays byte-identical after restaging on a new VM."""
    archive = output.parent/(output.name+'_results.zip')
    names = [*manifest['metadata'], 'archive_manifest.json']
    names += [n for n in ('archive_acknowledgment.json','cleanup.json') if (output/n).exists()]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted(set(names)):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, (output/name).read_bytes())
    return archive
