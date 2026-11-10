"""Prepare a compact display from native, co-registered seismic and RGT arrays."""
import base64
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
cases = []
provenance = json.loads((ROOT / "provenance.json").read_text())
cross = ndimage.generate_binary_structure(3, 1)
for index in [2, 1]:
    folder = ROOT / f"model_{index:04d}"
    image = np.fromfile(folder / "image3d.bin", dtype="<f4").reshape(96, 96, 96)
    rgt = np.fromfile(folder / "rgt3d.bin", dtype="<f4").reshape(96, 96, 96)
    mask = rgt == 0
    assert np.isfinite(image).all() and np.isfinite(rgt).all()
    assert 0 < mask.mean() < 1
    inner = mask & ~ndimage.binary_erosion(mask, cross, iterations=2)
    outer = ndimage.binary_dilation(mask, cross, iterations=2) & ~mask
    # Exclude observation-window faces from the illustrative training bands.
    valid = np.ones_like(mask)
    for axis in range(3):
        for slab in [slice(0, 2), slice(-2, None)]:
            selection = [slice(None)] * 3
            selection[axis] = slab
            valid[tuple(selection)] = False
    codes = mask.astype(np.uint8)
    codes[inner & valid] = 2
    codes[outer & valid] = 3
    scale = float(np.percentile(np.abs(image), 99.5))
    gray = np.rint((np.clip(image / scale, -1, 1) + 1) * 127.5).astype(np.uint8)
    # Keep every second voxel for the small interactive preview only.
    view_gray = gray[::2, ::2, ::2].copy()
    view_codes = codes[::2, ::2, ::2].copy()
    counts = mask.sum(axis=(0, 2))
    selected = int(np.argmax(counts[::2]))
    meta = provenance["models"][str(index)]
    case = {
        "id": f"model_{index:04d}",
        "label": f"{meta['nsalt']} generated salt {'body' if meta['nsalt']=='1' else 'bodies'}"
                 + (" + faults" if meta['yn_fault']=='1' else ""),
        "gray": base64.b64encode(view_gray.tobytes()).decode(),
        "codes": base64.b64encode(view_codes.tobytes()).decode(),
        "defaultSlice": selected,
        "nativeForegroundVoxels": int(mask.sum()),
        "foregroundFraction": float(mask.mean()),
        "nativeConnectedComponents": int(ndimage.label(mask, cross)[1]),
        "clipAbsAmplitude": scale,
    }
    cases.append(case)
    print(json.dumps({k: v for k, v in case.items() if k not in ['gray', 'codes']}))
(ROOT / "view_data.json").write_text(json.dumps(cases, separators=(",", ":")))
(ROOT / "view_provenance.json").write_text(json.dumps({
    "native_shape": [96, 96, 96], "preview_shape": [48, 48, 48],
    "downsampling": "Every second native voxel; preview only",
    "amplitude_display": "Symmetric clipping at per-volume 99.5th absolute percentile",
    "mask": "rgt3d == 0; includes all salt voxels, not only surfaces",
    "bands": "2 native voxels, 6-connected; exclude outer 2-voxel observation frame",
    "no_predictions": True,
    "models": [{k: v for k, v in c.items() if k not in ['gray', 'codes']} for c in cases]
}, indent=2))
