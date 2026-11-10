"""Cross-check implicit salt masks against the physical-model modality."""
import hashlib
import json
from pathlib import Path
import re
import ssl
import tarfile
import urllib.request
import certifi
import numpy as np
from scipy import ndimage

root = Path(__file__).resolve().parent
meta = json.loads((root / "provenance.json").read_text())
found = 0
context = ssl.create_default_context(cafile=certifi.where())
with urllib.request.urlopen(meta["source"], timeout=40, context=context) as response:
    with tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            match = re.search(r"model_(\d+)/vp3d\.bin$", member.name)
            if not match or int(match[1]) not in [1, 2]:
                continue
            index = int(match[1])
            data = archive.extractfile(member).read()
            assert len(data) == 96 ** 3 * 4
            (root / f"model_{index:04d}" / "vp3d.bin").write_bytes(data)
            vp = np.frombuffer(data, dtype="<f4").reshape(96, 96, 96)
            rgt = np.fromfile(root / f"model_{index:04d}" / "rgt3d.bin", dtype="<f4").reshape(96, 96, 96)
            salt_vel = float(meta["models"][str(index)]["salt_vel"])
            zero, structural = rgt == 0, vp == np.float32(salt_vel)
            cc, count = ndimage.label(zero)
            sizes = np.bincount(cc.ravel())[1:]
            mismatch = zero & ~structural
            report = {
                "case": index, "salt_velocity": salt_vel,
                "rgt_zero_voxels": int(zero.sum()),
                "velocity_salt_voxels": int(structural.sum()),
                "zero_but_not_salt_velocity": int(mismatch.sum()),
                "nonzero_rgt_but_salt_velocity": int((~zero & structural).sum()),
                "mismatch_velocities": np.unique(vp[mismatch]).tolist()[:15],
                "component_sizes": sorted(sizes.tolist(), reverse=True),
                "vp_sha256": hashlib.sha256(data).hexdigest()
            }
            print(json.dumps(report), flush=True)
            (root / f"model_{index:04d}" / "label_audit.json").write_text(json.dumps(report, indent=2))
            found += 1
            if found == 2:
                break
assert found == 2
