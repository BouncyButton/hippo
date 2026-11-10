"""Copy independent approved result archives, then publish completed files."""
import importlib.util
import os
from pathlib import Path
import sys

base = Path(__file__).resolve().parent
requested = sys.argv[1:]
sys.argv = [str(base / 'archive_cleanup.py'), 'archive_results']
spec = importlib.util.spec_from_file_location('retirement_archive', base / 'archive_cleanup.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert requested and len(requested) == len(set(requested))
assert set(requested).issubset(module.TARGETS)
module.ARCHIVE_DIR = 'parallel_results_archives'
for target in requested:
    destination = base / 'results_archives' / (target + '.tar.gz')
    if destination.exists():
        continue
    module.TARGETS = [target]
    module.archive()
    source = base / module.ARCHIVE_DIR / destination.name
    if destination.exists():
        # Another stream finished the same original snapshot; keep its complete archive.
        source.unlink()
    else:
        os.replace(source, destination)
