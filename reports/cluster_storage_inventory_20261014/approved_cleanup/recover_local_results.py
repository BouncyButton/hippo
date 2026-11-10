"""Recover byte-identical result/source files already present in durable local folders."""
from pathlib import Path
import collections,hashlib,json,os,shutil
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
manifest=json.loads((BASE/'content_manifest.json').read_text())['files']
cache=BASE/'transfer_blobs'
wanted=collections.defaultdict(set)
for name,r in manifest.items():
 if not (cache/r['sha256']).exists():wanted[(Path(name).name,r['bytes'])].add(r['sha256'])
wanted_names={key[0] for key in wanted}
count=total=0
for current,dirs,files in os.walk(ROOT):
 dirs[:]=[n for n in dirs if n not in {'.git','.venv','node_modules','.cache','MedSam3_weights','__pycache__'} and Path(current)/n != BASE]
 for name in files:
  if name not in wanted_names:continue
  path=Path(current)/name
  if path.is_symlink():continue
  try:
   key=(name,path.stat().st_size)
   hashes=wanted.get(key)
   if not hashes:continue
   data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
   if digest in hashes and not (cache/digest).exists():
    (cache/digest).write_bytes(data);count+=1;total+=len(data)
  except (OSError,PermissionError):continue
print(json.dumps(dict(recovered_files=count,recovered_mib=total/2**20)),flush=True)
