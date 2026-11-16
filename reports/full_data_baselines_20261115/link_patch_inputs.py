"""Link the frozen model source and initialization already installed in this Colab."""
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
native = Path(sys.argv[1]).resolve()
assert here != native and (native / "jobs.json").exists()
for key in ("source", "entry.py", "runtime.py", "initialization_reference.json"):
    obj = native / key
    files = obj.rglob("*") if obj.is_dir() else [obj]
    for path in files:
        dest = here / path.relative_to(native)
        if path.is_file() and "__pycache__" not in path.parts and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, dest)
