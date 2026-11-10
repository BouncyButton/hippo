"""Exercise crop geometry without running the builder's download/upload steps."""
import ast
from pathlib import Path

import nibabel as nib
import numpy as np


def crop_function():
    source = Path(__file__).with_name('create_cobra_dataset.py')
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'crop_and_save')
    namespace = {'np': np, 'nib': nib, 'TARGET_SIZE': np.array([4, 6, 8])}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['crop_and_save']


def test_subsampled_crop_preserves_world_coordinates(tmp_path):
    crop = crop_function()
    data = np.arange(12**3, dtype=np.float32).reshape(12, 12, 12)
    affine = np.array([[0, -2, 0, 30], [1, 0, 0, -10], [0, 0, 3, 4], [0, 0, 0, 1.]])
    target = tmp_path/'crop.nii.gz'
    result = crop(data, affine, np.array([4, 5, 6]), target)
    np.testing.assert_array_equal(result, data[2:6:2, 2:8:2, 2:10:2])
    saved = nib.load(target)
    for voxel in ([0, 0, 0], [1, 2, 3]):
        source_voxel = np.array([2, 2, 2]) + 2*np.array(voxel)
        np.testing.assert_allclose(nib.affines.apply_affine(saved.affine, voxel),
                                   nib.affines.apply_affine(affine, source_voxel))


def test_translated_annotation_crop_matches_image_with_padding(tmp_path):
    crop = crop_function()
    label = np.zeros((12, 12, 12), dtype=np.float32)
    label[0:4, 0:4, 0:4] = 5
    offset = np.array([2, 4, 2])
    image = np.pad(label, ((2, 0), (4, 0), (2, 0)))
    image_affine = np.diag([.5, .75, 1.25, 1.])
    label_affine = image_affine.copy()
    label_affine[:3, 3] = nib.affines.apply_affine(image_affine, offset)
    center = np.array([1, 1, 1])
    image_target, label_target = tmp_path/'image.nii.gz', tmp_path/'label.nii.gz'
    image_crop = crop(image, image_affine, center + offset, image_target)
    label_crop = crop(label, label_affine, center, label_target)
    np.testing.assert_array_equal(image_crop, label_crop)
    np.testing.assert_allclose(nib.load(image_target).affine, nib.load(label_target).affine)
