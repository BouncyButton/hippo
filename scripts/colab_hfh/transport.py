"""Split large immutable inputs for the connector's 100 MiB transfer limit."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import time

PART_BYTES = 96 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def assemble_asset(name, home, destination, transport, fetch_part=None):
    """Stream verified parts to SSD, publishing only the verified whole file."""
    spec = transport['assets'][name]
    if destination.exists():
        if sha(destination) != spec['sha256']:
            raise ValueError('Existing assembled input differs; retained: '+str(destination))
        return
    temporary = destination.with_name(destination.name+'.assembling')
    whole = hashlib.sha256()
    total = 0
    started = time.monotonic()
    with temporary.open('wb') as output:
        for index, part in enumerate(spec['parts'], 1):
            relative = Path(part['name'])
            if relative.name != part['name']:
                raise ValueError('Unsafe input part name')
            checksum = hashlib.sha256()
            size = 0
            print(f'Staging {name}: part {index}/{len(spec["parts"])} '
                  f'({part["bytes"]/1024**2:.1f} MiB)', flush=True)
            source_path = fetch_part(part) if fetch_part else home/relative
            with source_path.open('rb') as source:
                for block in iter(lambda: source.read(8*1024**2), b''):
                    checksum.update(block); whole.update(block)
                    output.write(block); size += len(block)
            if size != part['bytes'] or checksum.hexdigest() != part['sha256']:
                raise ValueError('Input part failed verification: '+part['name'])
            total += size
            elapsed = time.monotonic()-started
            print(f'Verified {total/1024**2:.1f}/{spec["bytes"]/1024**2:.1f} MiB '
                  f'in {elapsed:.1f}s', flush=True)
    if total != spec['bytes'] or whole.hexdigest() != spec['sha256']:
        raise ValueError('Assembled input failed verification: '+name)
    temporary.replace(destination)


def prepare(root):
    original = json.loads((root/'upload_manifest.json').read_text())
    assets = {}
    for row in original['files']:
        if row['file_name'] not in ('sam3.pt', 'hfh_exploration_bundle.zip'):
            continue
        path = Path(row['local_path'])
        parts, offset = [], 0
        whole = hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                block = stream.read(PART_BYTES)
                if not block:
                    break
                whole.update(block)
                parts.append(dict(name=path.name+f'.part{len(parts):03d}', offset=offset,
                    bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
                offset += len(block)
        if offset != row['bytes'] or whole.hexdigest() != row['sha256']:
            raise ValueError('Frozen input changed')
        assets[row['file_name']] = dict(local_path=str(path), bytes=offset,
            sha256=whole.hexdigest(), parts=parts)
    transport = dict(version=1, part_bytes=PART_BYTES, assets=assets,
        reason='Google Drive connector rejects input files above104857600 bytes; scientific bytes remain unchanged')
    path = root/'transport_manifest.json'
    path.write_text(json.dumps(transport, indent=2)+'\n')
    helper = ("TRANSPORT_SHA="+repr(sha(path))+"\n"
              "if sha(HOME/'transport_manifest.json')!=TRANSPORT_SHA:raise ValueError('Transport manifest differs')\n"
              "transport=json.loads((HOME/'transport_manifest.json').read_text())\n"+
              inspect.getsource(assemble_asset)+'\n')
    for notebook in root.glob('*.ipynb'):
        doc = json.loads(notebook.read_text())
        stage = doc['cells'][2]['source']
        if 'TRANSPORT_SHA=' in stage:
            raise ValueError('Notebook already adapted; avoid duplicate patching')
        stage = helper+stage
        old = 'if not local.exists():shutil.copyfile(source,local)'
        if stage.count(old) != 1:
            raise ValueError('Unexpected package staging cell')
        stage = stage.replace(old, "assemble_asset(source.name,HOME,local,transport)")
        old = "if not base.exists():shutil.copyfile(HOME/'sam3.pt',base)"
        if stage.count(old) != 1:
            raise ValueError('Unexpected checkpoint staging cell')
        stage = stage.replace(old, "assemble_asset('sam3.pt',HOME,base,transport)")
        ast.parse(stage)
        doc['cells'][2]['source'] = stage
        doc['cells'][0]['source'] += '\nLarge inputs are stored as37 transport parts (2 bundle +35 base checkpoint); this notebook verifies and joins them automatically on local SSD. Do not rename the parts.\n'
        notebook.write_text(json.dumps(doc,indent=2)+'\n')
    return transport


def stage(root, index):
    transport = json.loads((root/'transport_manifest.json').read_text())
    entries = [(asset, part) for asset in transport['assets'].values() for part in asset['parts']]
    asset, part = entries[index]
    directory = root/'upload_staging'
    directory.mkdir(exist_ok=True)
    target = directory/part['name']
    with Path(asset['local_path']).open('rb') as source:
        source.seek(part['offset'])
        data = source.read(part['bytes'])
    if hashlib.sha256(data).hexdigest() != part['sha256']:
        raise ValueError('Source part changed')
    target.write_bytes(data)
    print(json.dumps(dict(path=str(target), **part)))


def retire(root, index):
    transport = json.loads((root/'transport_manifest.json').read_text())
    entries = [part for asset in transport['assets'].values() for part in asset['parts']]
    part = entries[index]
    receipt = json.loads((root/'uploaded_parts'/(part['name']+'.json')).read_text())
    if receipt['status'] != 'metadata_verified' or int(receipt['metadata']['size']) != part['bytes']:
        raise ValueError('Upload not verified; retain staging')
    path = root/'upload_staging'/part['name']
    if sha(path) != part['sha256']:
        raise ValueError('Local upload part changed')
    path.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--retire', type=int)
    args = parser.parse_args()
    if args.retire is not None:
        retire(args.root, args.retire)
    elif args.stage is None:
        value = prepare(args.root)
        print(json.dumps({name:len(spec['parts']) for name,spec in value['assets'].items()}))
    else:
        stage(args.root, args.stage)
