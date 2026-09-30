"""Fail closed on quota or incomplete MedSAM3 inputs before submission/run."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import hashlib

BASE_REVISION = '3c879f39826c281e95690f02c7821c4de09afae7'
UPSTREAM_REVISION = '84f43118b5b67b50abad81d7c946681ebb97dcda'
ADAPTER_SHA256 = '499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def quota_free_bytes(output):
    """Use the tightest applicable user/group space limit; fail on unknown user quota."""
    remaining = []
    saw_user = False
    units = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3, 'TiB': 1024**4}
    for line in output.splitlines():
        if not re.search(r'\b(user|group)\b', line):
            continue
        is_user = bool(re.search(r'\buser\b', line))
        match = re.search(r'([\d.]+)([KMGT]?i?B)/([\d.]+)([KMGT]?i?B)', line)
        if match:
            used, unit, limit, limit_unit = match.groups()
            remaining.append(float(limit)*units[limit_unit] - float(used)*units[unit])
            saw_user = saw_user or is_user
        elif re.search(r'([\d.]+)([KMGT]?i?B)/∞', line):
            saw_user = saw_user or is_user
        else:
            raise RuntimeError('Cannot parse an applicable quota row.')
    if not saw_user or not remaining:
        raise RuntimeError('Cannot establish finite quota reserve.')
    return min(remaining)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--minimum-free-gib', type=float, default=1.)
    p.add_argument('--base-weights', type=Path)
    p.add_argument('--allow-download', action='store_true',
                   help='Check authorized HF access instead of requiring a persistent base file.')
    args = p.parse_args()
    manifest = json.loads((args.root/'source/SOURCE_MANIFEST.json').read_text())
    for name, expected in manifest.items():
        if digest(args.root/'source'/name) != expected:
            raise RuntimeError(f'Frozen source changed: {name}')
    commit = subprocess.check_output(['git', '-C', str(args.root/'MedSAM3'), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_REVISION:
        raise RuntimeError('Upstream revision differs from the reviewed version.')
    if digest(args.root/'weights/best_lora_weights.pt') != ADAPTER_SHA256:
        raise RuntimeError('Medical adapter differs from the verified release.')
    output = subprocess.check_output(
        ['beegfs', 'quota', 'list-usage', '--uids', 'current', '--gids', 'current'], text=True)
    free = quota_free_bytes(output)
    print(json.dumps(dict(quota_output=output, free_gib=free/1024**3,
                          minimum_free_gib=args.minimum_free_gib)), flush=True)
    if free < args.minimum_free_gib*1024**3:
        raise RuntimeError('Insufficient or unknown BeeGFS quota reserve.')
    required = ['weights/best_lora_weights.pt',
                'MedSAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz']
    missing = [f for f in required if not (args.root/f).is_file() or (args.root/f).stat().st_size == 0]
    base = args.base_weights or args.root/'weights/sam3.pt'
    access_error = None
    if not base.is_file():
        if args.allow_download:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url, get_token
            try:
                metadata = get_hf_file_metadata(hf_hub_url('facebook/sam3', 'sam3.pt', revision=BASE_REVISION), token=get_token())
                if metadata.size != 3450062241:
                    access_error = 'Unexpected official base-checkpoint size.'
            except Exception as exc:
                # Do not persist signed URLs or credential-bearing exception text.
                access_error = type(exc).__name__ + ': approved facebook/sam3 access required.'
        else:
            missing.append(str(base))
    elif base.stat().st_size != 3450062241:
        access_error = 'Base checkpoint size differs from pinned official release.'
    record = dict(quota_output=output, free_gib=free/1024**3, minimum_free_gib=args.minimum_free_gib,
                  missing=missing, base_access_error=access_error,
                  status='passed' if not missing and not access_error else 'blocked')
    (args.root/'preflight.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))
    if missing or access_error:
        raise RuntimeError('Required model inputs missing; do not submit.')


if __name__ == '__main__':
    main()
