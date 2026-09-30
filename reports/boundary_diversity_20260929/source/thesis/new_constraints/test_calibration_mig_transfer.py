"""The explicit MIG transfer changes hardware identity, never numerical policy."""
import pytest

from thesis.new_constraints.train_swinunetr_constraints import validate_calibration_execution


def execution(profile):
    return dict(device_type="cuda", deterministic_algorithms=False,
                cudnn_benchmark=False, cudnn_deterministic=True,
                float32_matmul_precision="highest", cuda_device_index=0,
                cuda_device_name=f"NVIDIA A100 80GB PCIe MIG {profile}.40gb",
                cuda_compute_capability=[8, 0], cuda_total_memory=42144366592)


def test_transfer_requires_opt_in_and_preserves_metadata():
    old, new = execution("4g"), execution("3g")
    original = old.copy()
    validate_calibration_execution(old, old)
    with pytest.raises(ValueError, match="compute device"):
        validate_calibration_execution(old, new)
    validate_calibration_execution(old, new, allow_mig_profile_change=True)
    assert old == original and new["cuda_device_name"].endswith("3g.40gb")


@pytest.mark.parametrize("key,value", [
    ("cuda_device_name", "NVIDIA A100 80GB PCIe MIG 2g.20gb"),
    ("cuda_total_memory", 20 * 1024**3),
    ("cuda_total_memory", None),
    ("cuda_compute_capability", [9, 0]),
    ("cudnn_deterministic", False),
    ("float32_matmul_precision", "high"),
    ("device_type", "cpu"),
])
def test_transfer_does_not_relax_other_requirements(key, value):
    new = execution("3g")
    new[key] = value
    with pytest.raises(ValueError, match="compute device"):
        validate_calibration_execution(execution("4g"), new, allow_mig_profile_change=True)
