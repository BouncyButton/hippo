#!/usr/bin/env python3
"""Execute an experiment only after binding imports to a pre-launch source digest."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_ENV = "HIPPO_EXPECTED_SOURCE_SHA256"


def source_digest() -> str:
    constraint_root = REPO_ROOT / "thesis" / "new_constraints"
    source_paths = [
        path
        for path in constraint_root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh"}
        and not path.name.startswith("test_")
        and "__pycache__" not in path.parts
    ]
    source_paths.append(REPO_ROOT / "baselines" / "swin_unetr" / "swin_unetr.py")
    aggregate = hashlib.sha256()
    for path in sorted(source_paths):
        relative_path = str(path.relative_to(REPO_ROOT))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: source_bootstrap.py TARGET.py [arguments ...]")
    target = Path(sys.argv[1]).expanduser().resolve()
    if not target.is_file() or REPO_ROOT not in target.parents:
        raise SystemExit("Bootstrap target must be a repository Python file.")
    environment = os.environ.copy()
    environment[PROVENANCE_ENV] = source_digest()
    os.execve(
        sys.executable,
        [sys.executable, str(target), *sys.argv[2:]],
        environment,
    )


if __name__ == "__main__":
    main()
