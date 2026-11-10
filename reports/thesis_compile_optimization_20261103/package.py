"""Package the current thesis and its figure dependencies for Overleaf."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path('/Users/filippofocaccia/Desktop/hippo')
PROJECT = ROOT / 'Template_tesi'
AUDIT = Path(__file__).resolve().parent
source = PROJECT / 'thesis-revision.tex'
files = [source, PROJECT / 'OVERLEAF_README.txt', PROJECT / 'abstract.txt']
supplement = PROJECT / 'thesis-supplementary-results.zip'
if supplement.is_file():
    files.append(supplement)
files.extend(sorted(p for p in (PROJECT / 'figures').rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts
                    and p.suffix != '.pyc'))
for dependency in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source.read_text()):
    path = PROJECT / dependency
    assert path.is_file(), f'Missing figure: {dependency}'
    if path not in files:
        files.append(path)
output = PROJECT / 'thesis-overleaf-fast.zip'
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(PROJECT).as_posix())
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert archive.read('thesis-revision.tex') == source.read_bytes()
    for path in files:
        assert archive.read(path.relative_to(PROJECT).as_posix()) == path.read_bytes()
record = dict(zip=str(output), bytes=output.stat().st_size, members=len(files),
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(AUDIT / 'package.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
