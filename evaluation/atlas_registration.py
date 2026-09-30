"""Image-only crop registration and multi-atlas label transfer for a frozen pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT / "experiments/atlas_pilot_20260924/dependencies"
sys.path.insert(0, str(DEPENDENCIES))
import SimpleITK as sitk

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
DATASET = ROOT / "datasets/Dataset101_MSD"
CACHE = ROOT / "experiments/atlas_pilot_20260924"
REPORT = ROOT / "docs/experiments/atlas_pilot_20260924"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def normalize(image):
    lo, hi = np.quantile(image, [.01, .99])
    return np.clip((image - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


def itk_image(array):
    image = sitk.GetImageFromArray(np.asarray(array, np.float32).transpose(2, 1, 0).copy())
    image.SetOrigin(tuple(-(np.array(array.shape) - 1) / 2))
    return image


def array(image):
    return sitk.GetArrayFromImage(image).transpose(2, 1, 0)


def native(name, labels=False):
    path = DATASET / ("labelsTr" if labels else "imagesTr") / f"{name}{'' if labels else '_0000'}.nii.gz"
    image = nib.load(path)
    if nib.aff2axcodes(image.affine) != ("R", "A", "S") or not np.allclose(image.header.get_zooms(), 1):
        raise ValueError(f"Expected native 1-mm RAS: {path}")
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all() or (labels and not np.isin(values, [0, 1, 2]).all()):
        raise ValueError(path)
    return values


def nmi(fixed, moving, valid=None):
    a, b = np.asarray(fixed).ravel(), np.asarray(moving).ravel()
    if valid is not None:
        a, b = a[np.asarray(valid).ravel()], b[np.asarray(valid).ravel()]
    hist = np.histogram2d(a, b, bins=32, range=((0, 1), (0, 1)))[0]
    hist /= max(hist.sum(), 1)
    entropy = lambda p: float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    ha, hb, joint = entropy(hist.sum(0)), entropy(hist.sum(1)), entropy(hist)
    return (ha + hb - joint) / max((ha + hb) / 2, 1e-12)


def resample(moving, fixed, transform, default=0.):
    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, float(default), sitk.sitkFloat32)


def compose(affine, residual):
    """SimpleITK AddTransform copies: compose only AFTER optimizing residual."""
    transform = sitk.CompositeTransform(3)
    transform.AddTransform(affine)
    transform.AddTransform(residual)
    return transform


def select_atlases(target_image, bank, count=3):
    """Select distinct source IDs and optional LR reflection; no labels argument."""
    fixed = itk_image(normalize(target_image))
    fixed_arr = array(fixed)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    ranked = []
    for name in bank:
        image = normalize(native(name))
        options = []
        for mirror in (False, True):
            moving = itk_image(image[::-1] if mirror else image)
            warped = array(resample(moving, fixed, identity))
            valid = array(resample(itk_image(np.ones(image.shape)), fixed, identity)) > .99
            options.append(dict(source=name, mirror=mirror, initial_nmi=nmi(fixed_arr, warped, valid)))
        ranked.append(max(options, key=lambda x: (x["initial_nmi"], not x["mirror"])))
    return sorted(ranked, key=lambda x: (-x["initial_nmi"], x["source"]))[:count]


def metric_registration():
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(32)
    registration.SetMetricSamplingStrategy(registration.NONE)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1., 0.])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return registration


def register(fixed, moving):
    """Estimate fixed-to-moving affine and residual transforms using images only."""
    start = time.monotonic()
    identity = sitk.Transform(3, sitk.sitkIdentity)
    affine = sitk.AffineTransform(3)
    info = dict(affine_accepted=False, deformable_accepted=False)
    reg = metric_registration()
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1., minStep=.01,
        numberOfIterations=60, relaxationFactor=.5, gradientMagnitudeTolerance=1e-6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(affine, inPlace=True)
    try:
        reg.Execute(fixed, moving)
        matrix = np.array(affine.GetMatrix()).reshape(3, 3)
        singular = np.linalg.svd(matrix, compute_uv=False)
        determinant = float(np.linalg.det(matrix))
        translation = float(np.linalg.norm(affine.GetTranslation()))
        info.update(affine_singular_values=singular.tolist(), affine_determinant=determinant,
                    affine_translation_mm=translation, affine_stop=reg.GetOptimizerStopConditionDescription())
        info["affine_accepted"] = bool(singular.min() >= .65 and singular.max() <= 1.5
            and .4 <= determinant <= 2.5 and translation <= 12.)
    except RuntimeError as error:
        info["affine_error"] = str(error)
    affine_transform = affine if info["affine_accepted"] else identity
    bspline = sitk.BSplineTransformInitializer(fixed, [2, 3, 2])
    reg = metric_registration()
    reg.SetMovingInitialTransform(affine_transform)
    reg.SetInitialTransform(bspline, inPlace=True)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=25,
                           maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=150)
    composed = affine_transform
    try:
        reg.Execute(fixed, moving)
        composed = compose(affine_transform, bspline)
        displacement = sitk.TransformToDisplacementField(bspline, sitk.sitkVectorFloat64,
            fixed.GetSize(), fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())
        jac = array(sitk.DisplacementFieldJacobianDeterminant(displacement))
        max_displacement = float(np.linalg.norm(sitk.GetArrayFromImage(displacement), axis=-1).max())
        valid = array(resample(itk_image(np.ones(array(moving).shape)), fixed, composed))
        coverage = float((valid > .99).mean())
        info.update(residual_min_jacobian=float(jac.min()), residual_max_displacement_mm=max_displacement,
                    coverage=coverage, deformable_stop=reg.GetOptimizerStopConditionDescription())
        info["deformable_accepted"] = bool(jac.min() > 0 and max_displacement <= 8 and coverage >= .5)
    except RuntimeError as error:
        info["deformable_error"] = str(error)
    final = composed if info["deformable_accepted"] else affine_transform
    info["seconds"] = time.monotonic() - start
    return dict(centered=identity, affine=affine_transform, deformable=final), info


def transfer_labels(labels, fixed, transform):
    foreground = np.stack([array(resample(itk_image(labels == cls), fixed, transform)) for cls in (1, 2)])
    probabilities = np.concatenate(((1 - foreground.sum(0))[None], foreground), axis=0)
    probabilities = np.clip(probabilities, 0, 1)
    return probabilities / np.maximum(probabilities.sum(0), 1e-9)


def build_prior(target, bank, provenance):
    if target in bank:
        raise ValueError("Target appears in atlas bank")
    path = CACHE / "priors" / f"{target}.npz"
    record_path = REPORT / "registrations" / f"{target}.json"
    input_paths = [DATASET / "imagesTr" / f"{name}_0000.nii.gz" for name in [target] + bank]
    input_paths += [DATASET / "labelsTr" / f"{name}.nii.gz" for name in bank]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    if path.exists():
        record = json.loads(record_path.read_text())
        if record["provenance"] != provenance or record["input_hashes"] != input_hashes or record["prior_sha256"] != sha(path):
            raise ValueError("Prior provenance changed")
        return record
    image = native(target)
    fixed = itk_image(normalize(image))
    chosen = select_atlases(image, bank)
    results = {kind: [] for kind in ("centered", "affine", "deformable")}
    diagnostics = []
    for choice in chosen:
        source = choice["source"]
        moving_arr = normalize(native(source))
        if choice["mirror"]:
            moving_arr = moving_arr[::-1]
        transforms, info = register(fixed, itk_image(moving_arr))
        # Source labels are loaded only AFTER estimating transforms.
        labels = native(source, labels=True)
        if choice["mirror"]:
            labels = labels[::-1]
        for kind, transform in transforms.items():
            results[kind].append(transfer_labels(labels, fixed, transform))
            transform_path = CACHE / "transforms" / target / f"{source}_{kind}.tfm"
            transform_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteTransform(transform, str(transform_path))
        info.update(choice)
        for kind, transform in transforms.items():
            moved = array(resample(itk_image(moving_arr), fixed, transform))
            valid = array(resample(itk_image(np.ones(moving_arr.shape)), fixed, transform)) > .99
            info[f"{kind}_nmi"] = nmi(array(fixed), moved, valid)
        diagnostics.append(info)
        print(f"  {target} <- {source}, mirror={choice['mirror']}, {info['seconds']:.1f}s, nonlinear={info['deformable_accepted']}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{kind: np.mean(values, axis=0).astype(np.float32) for kind, values in results.items()})
    record = dict(target=target, sources=diagnostics, provenance=provenance,
                  input_hashes=input_hashes, prior_sha256=sha(path))
    save_json(record_path, record)
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("smoke", "calibration", "assessment", "development", "train"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cohort = json.loads((REPORT / "cohort.json").read_text())
    names = cohort["calibration"][:1] if args.subset == "smoke" else (cohort["calibration"] + cohort["assessment"] if args.subset == "train" else cohort[args.subset])
    paths = [Path(__file__), REPORT / "PROTOCOL.md", REPORT / "cohort.json"]
    provenance = dict(files={str(p.relative_to(ROOT)): sha(p) for p in paths}, sitk=sitk.Version_VersionString())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for record in executor.map(lambda name: build_prior(name, cohort["atlas_bank"], provenance), names):
            print(f"Completed prior: {record['target']}", flush=True)
