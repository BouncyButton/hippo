"""Read a bounded archive prefix for two native GeoFWI3D examples."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://zenodo.org/records/20148778/files/models_batch_0.tar.gz?download=1"
META = "https://raw.githubusercontent.com/aaspip/geofwi3d/main/parameters.csv"
LIMIT = 96 * 1024 * 1024


class BoundedReader:
    def __init__(self, response):
        self.response = response
        self.total = 0

    def read(self, size=-1):
        if size < 0:
            size = LIMIT - self.total
        if self.total + size > LIMIT:
            raise RuntimeError("Preview download exceeded 96 MiB safety limit")
        data = self.response.read(size)
        self.total += len(data)
        return data


with urllib.request.urlopen(META, timeout=40) as response:
    rows = list(csv.DictReader(io.StringIO(response.read().decode())))
metadata = {int(row["sample_index"]): row for row in rows}
targets = {1, 2}
receipts = []
with urllib.request.urlopen(URL, timeout=40) as response:
    reader = BoundedReader(response)
    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
        for member in archive:
            match = re.search(r"model_(\d+)/(image3d|rgt3d)\.bin$", member.name)
            if match is None or int(match[1]) not in targets:
                continue
            index, modality = int(match[1]), match[2]
            assert member.size == 96 ** 3 * 4, (member.name, member.size)
            data = archive.extractfile(member).read()
            assert len(data) == member.size
            destination = ROOT / f"model_{index:04d}"
            destination.mkdir(exist_ok=True)
            (destination / f"{modality}.bin").write_bytes(data)
            receipt = {"member": member.name, "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
            if len(receipts) == 4:
                break
    transferred = reader.total
assert len(receipts) == 4, receipts
(ROOT / "provenance.json").write_text(json.dumps({
    "source": URL, "metadata_source": META,
    "compressed_bytes_read": transferred, "files": receipts,
    "models": {str(index): metadata[index] for index in sorted(targets)},
    "shape": [96, 96, 96], "dtype": "little-endian float32",
    "mask_definition": "rgt3d == 0", "purpose": "Illustration; not model predictions"
}, indent=2))
print(json.dumps({"compressed_bytes_read": transferred}), flush=True)
