"""Download verified GeoFWI3D archive 0 and its model metadata, with resume."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import ssl
import time
import urllib.request

ARCHIVE_NAME = "models_batch_0.tar.gz"
ARCHIVE_URL = f"https://zenodo.org/records/20148778/files/{ARCHIVE_NAME}?download=1"
ARCHIVE_BYTES = 7358564728
ARCHIVE_MD5 = "57013080521bfa96069db03b98e2e553"
METADATA_URL = "https://raw.githubusercontent.com/aaspip/geofwi3d/main/parameters.csv"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def download(root: Path) -> dict:
    cache = root / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    metadata_path = cache / "parameters.csv"
    if not metadata_path.exists():
        with urllib.request.urlopen(METADATA_URL, timeout=60, context=context) as response:
            payload = response.read()
        if not payload.startswith(b"sample_index,"):
            raise RuntimeError("Unexpected metadata response")
        temporary = metadata_path.with_suffix(".csv.part")
        temporary.write_bytes(payload)
        temporary.replace(metadata_path)
    destination = cache / ARCHIVE_NAME
    partial = cache / (ARCHIVE_NAME + ".part")
    if not destination.exists():
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > ARCHIVE_BYTES:
                raise RuntimeError("Partial archive exceeds expected length")
            if offset == ARCHIVE_BYTES:
                break
            request = urllib.request.Request(ARCHIVE_URL, headers={
                "User-Agent": "GeoFWI3D-research-download/1.0",
                **({"Range": f"bytes={offset}-"} if offset else {})})
            started, transferred, last_report = time.monotonic(), 0, 0.0
            try:
                with urllib.request.urlopen(request, timeout=90, context=context) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("Server resumed from an unexpected byte")
                        mode = "ab"
                    elif response.status == 200:
                        offset, mode = 0, "wb"
                    else:
                        raise RuntimeError(f"Unexpected download status {response.status}")
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise RuntimeError("Archive endpoint returned HTML")
                    with partial.open(mode) as stream:
                        while block := response.read(4 * 1024 * 1024):
                            stream.write(block)
                            transferred += len(block)
                            now = time.monotonic()
                            if now - last_report >= 20:
                                progress = {"state": "downloading", "bytes": offset + transferred,
                                            "total_bytes": ARCHIVE_BYTES,
                                            "megabytes_per_second": round(transferred / max(now-started, .001) / 1e6, 2)}
                                atomic_json(root / "download_status.json", progress)
                                print(json.dumps(progress), flush=True)
                                last_report = now
                break
            except (TimeoutError, OSError) as error:
                atomic_json(root / "download_status.json", {"state": "retrying", "attempt": attempt+1, "error": str(error)})
                if attempt == 2:
                    raise
                time.sleep(2)
        if not partial.exists() or partial.stat().st_size != ARCHIVE_BYTES:
            raise RuntimeError("Downloaded archive size does not match release")
        atomic_json(root / "download_status.json", {"state": "verifying_md5", "bytes": ARCHIVE_BYTES})
        if digest(partial, "md5") != ARCHIVE_MD5:
            raise RuntimeError("Archive MD5 does not match the published Zenodo checksum")
        partial.replace(destination)
    if destination.stat().st_size != ARCHIVE_BYTES or digest(destination, "md5") != ARCHIVE_MD5:
        raise RuntimeError("Cached archive fails published size/MD5 verification")
    receipt = {"state": "complete", "archive": str(destination.resolve()),
               "url": ARCHIVE_URL, "bytes": ARCHIVE_BYTES, "md5": ARCHIVE_MD5,
               "sha256": digest(destination), "metadata_url": METADATA_URL,
               "metadata_sha256": digest(metadata_path), "archive_index": 0}
    atomic_json(root / "download_receipt.json", receipt)
    atomic_json(root / "download_status.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/Dataset203_GeoFWI3D"))
    args = parser.parse_args()
    print(json.dumps(download(args.root.resolve()), indent=2), flush=True)


if __name__ == "__main__":
    main()
