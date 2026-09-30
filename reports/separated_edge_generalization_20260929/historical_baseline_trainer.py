#!/usr/bin/env python3
"""Train SwinUNETR with a selectable auxiliary constraint on an exact CV fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_PROVENANCE_ENV = "HIPPO_EXPECTED_SOURCE_SHA256"
if __name__ == "__main__" and SOURCE_PROVENANCE_ENV not in os.environ:
    os.execve(
        sys.executable,
        [
            sys.executable,
            str(REPO_ROOT / "thesis" / "new_constraints" / "source_bootstrap.py"),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        os.environ.copy(),
    )

import monai  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from monai.data import DataLoader, Dataset as MonaiDataset  # noqa: E402
from monai.networks.nets import SwinUNETR  # noqa: E402

from baselines.swin_unetr.swin_unetr import (  # noqa: E402
    _build_monai_dataset_from_pkl,
    _load_pkl_dataframe,
    _load_splits_json,
    build_optimizer_and_scheduler,
)
from thesis.new_constraints import (  # noqa: E402
    NewConstraintConfig,
    NewConstraintObjective,
)
from thesis.new_constraints.constraint_result import ConstraintResult  # noqa: E402
from thesis.new_constraints.early_stopping import EarlyStopping  # noqa: E402
from thesis.new_constraints.ap_cut import AP_CUT_METRICS  # noqa: E402
from thesis.new_constraints.ap_plane import (  # noqa: E402
    AP_CONDITIONAL_CE_METRICS,
    AP_PLANE_LOCATION_METRICS,
)
from thesis.new_constraints.teacher import TEACHER_METRICS  # noqa: E402
from thesis.new_constraints.translation_augmentation import augment_translation  # noqa: E402
from thesis.new_constraints.training_augmentation import MILD_V1, augment_mild  # noqa: E402
from thesis.new_constraints.supervised import (  # noqa: E402
    CalibrationDiagnostics,
    build_supervised_loss,
    supervised_loss_config,
    calibration_diagnostics_config,
)
from thesis.new_constraints.training_telemetry import (  # noqa: E402
    TELEMETRY_EPOCH_FIELDS,
    TELEMETRY_GRADIENT_FIELDS,
    BoundaryTelemetryAccumulator,
    probe_component_gradients,
)

CONSTRAINT_WEIGHTS = {"none": 0.0, "equivariance": 0.10, "translation": 0.10}
CONSTRAINT_CHOICES = (
    "none", "equivariance", "bands", "onecut", "translation", "teacher", "ap_cut",
    "ap_plane", "ap_plane_location", "ap_plane_ce_control", "perimeter_profile",
    "ray_moment",
)
AGREEMENT_CONSTRAINT_NAMES = (
    "translation_equivariance", "translation_teacher_kl", "ap_cut_posterior",
)
BAND_METRICS = (
    "raw_loss",
    "inner_loss",
    "outer_loss",
    "inner_voxels",
    "outer_voxels",
    "class_1_soft_fp",
    "class_1_soft_fn",
    "class_2_soft_fp",
    "class_2_soft_fn",
    "valid_patient",
    "skipped_patient",
    "edge_touching",
)
ONECUT_METRICS = (
    "raw_loss",
    "normalized_truth",
    "allowed_cut_mass",
    "ray_count",
    "valid_patient",
    "skipped_patient",
    "edge_touching",
)
AP_PLANE_METRICS = (
    "raw_loss",
    "selected_cut",
    "candidate_count",
    "valid_patient",
    "skipped_patient",
)
PERIMETER_PROFILE_METRICS = ("raw_loss",)
RAY_MOMENT_METRICS = ("raw_loss",)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest used to bind runs to exact input contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FileSnapshot:
    """Private read-only copy whose digest describes the bytes later consumed."""

    original_path: Path
    path: Path
    sha256: str
    _temporary_directory: tempfile.TemporaryDirectory[str]
    _closed: bool = False

    def cleanup(self) -> None:
        if not self._closed:
            self._temporary_directory.cleanup()
            self._closed = True

    def __enter__(self) -> FileSnapshot:
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        self.cleanup()


def snapshot_file(path: Path) -> FileSnapshot:
    """Copy a file once while hashing, then consume only the private copy."""

    original_path = path.expanduser().resolve()
    temporary_directory = tempfile.TemporaryDirectory(prefix="hippo-input-")
    snapshot_path = Path(temporary_directory.name) / original_path.name
    digest = hashlib.sha256()
    try:
        with original_path.open("rb") as source, snapshot_path.open("xb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        snapshot_path.chmod(0o400)
    except Exception:
        temporary_directory.cleanup()
        raise
    return FileSnapshot(
        original_path=original_path,
        path=snapshot_path,
        sha256=digest.hexdigest(),
        _temporary_directory=temporary_directory,
    )


def collect_source_manifest() -> dict[str, str]:
    """Hash only the defining source files, without invoking Git."""

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
    return {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in sorted(source_paths)
    }


def source_manifest_sha256(manifest: dict[str, str]) -> str:
    aggregate = hashlib.sha256()
    for relative_path, digest in manifest.items():
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def validate_preimport_source_digest() -> str | None:
    """Match the imported process to the digest captured by the bootstrap."""

    expected_digest = os.environ.get(SOURCE_PROVENANCE_ENV)
    if expected_digest is None:
        return None
    current_digest = source_manifest_sha256(collect_source_manifest())
    if expected_digest != current_digest:
        raise RuntimeError("Source files changed between pre-launch hashing and import.")
    return expected_digest


PREIMPORT_SOURCE_SHA256 = validate_preimport_source_digest()


def collect_source_provenance() -> dict[str, Any]:
    """Describe the local source files that define training and calibration."""

    manifest = collect_source_manifest()
    source_sha256 = source_manifest_sha256(manifest)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        git_status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                *manifest.keys(),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        source_git_dirty = bool(git_status.strip())
    except (OSError, subprocess.CalledProcessError):
        source_git_dirty = None
    try:
        repository_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        repository_git_dirty = bool(repository_status.strip())
    except (OSError, subprocess.CalledProcessError):
        repository_git_dirty = None
    return {
        "sha256": source_sha256,
        "preimport_sha256": PREIMPORT_SOURCE_SHA256,
        "files": manifest,
        "git_commit": commit,
        # Keep the historical key for old report readers, but make its scope explicit.
        "git_dirty": source_git_dirty,
        "source_git_dirty": source_git_dirty,
        "repository_git_dirty": repository_git_dirty,
    }


def collect_runtime_provenance() -> dict[str, Any]:
    """Describe the numerical software stack that can affect experiment results."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "monai": monai.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }


def collect_execution_provenance(device: torch.device) -> dict[str, Any]:
    """Describe the resolved compute target and numerical backend policy."""

    payload: dict[str, Any] = {
        "device_type": device.type,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_compute_capability": [properties.major, properties.minor],
                "cuda_total_memory": properties.total_memory,
            }
        )
    return payload


def canonical_sha256(payload: Any) -> str:
    """Hash a JSON-compatible identity with stable ordering and no NaN values."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_source_provenance_unchanged(expected: dict[str, Any]) -> None:
    """Fail if defining source files changed after the initial provenance pass."""

    current_digest = source_manifest_sha256(collect_source_manifest())
    if current_digest != expected.get("sha256"):
        raise RuntimeError("Experiment source files changed during process startup.")


def patient_id_from_case(case_name: str, dataset: str) -> str:
    """Return the patient group encoded in a dataset case name."""

    dataset = dataset.upper()
    if dataset == "MSD":
        return case_name
    patterns = {
        "MNI": r"s\d+",
        "ADNI": r"adni_\d+",
        "COBRA": r"cobra_\d+",
    }
    if dataset not in patterns:
        raise ValueError(f"Unknown dataset: {dataset}")
    match = re.search(patterns[dataset], case_name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not derive a patient ID from case name: {case_name}")
    return match.group().lower()


def build_swinunetr(
    spatial_size: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    *,
    drop_rate: float = 0.0,
    activation_checkpointing: bool = True,
) -> SwinUNETR:
    """Build SwinUNETR across MONAI versions with and without ``img_size``."""
    if not math.isfinite(drop_rate) or not 0 <= drop_rate < 1:
        raise ValueError("--drop-rate must be finite and in [0, 1).")
    kwargs: dict[str, Any] = {
        "in_channels": 1,
        "out_channels": num_classes,
        "use_checkpoint": activation_checkpointing,
        "drop_rate": drop_rate,
    }
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = spatial_size
    return SwinUNETR(**kwargs).to(device)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    fold: int
    seed: int
    translation_seed: int
    epochs: int
    batch_size: int
    spatial_size: tuple[int, int, int]
    resize: bool
    optimizer_mode: str
    learning_rate: float | None
    weight_decay: float | None
    step_size: int
    adamw_gamma: float
    constraint_set: str
    constraint_config: dict[str, Any]
    constraint_warmup_epochs: int
    constraint_scale_knots: tuple[tuple[int, float], ...]
    constraint_eval_every: int
    amp: bool
    initial_checkpoint: str | None
    initial_checkpoint_sha256: str | None = None
    pkl_sha256: str = ""
    splits_json_sha256: str = ""
    source_sha256: str = ""
    runtime_sha256: str = ""
    execution_sha256: str = ""
    bands_calibration: str | None = None
    bands_calibration_sha256: str | None = None
    onecut_calibration: str | None = None
    onecut_calibration_sha256: str | None = None
    ap_plane_calibration: str | None = None
    ap_plane_calibration_sha256: str | None = None
    telemetry: bool = False
    telemetry_probe_epochs: tuple[int, ...] = ()
    telemetry_probe_cases: int = 0
    telemetry_spatial_cases: int = 0
    supervised_loss: str = "dice"
    ce_weight: float = 0.0
    calibration_diagnostics: bool = False
    translation_augmentation: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0005
    early_stopping_min_epochs: int = 25
    drop_rate: float = 0.0
    activation_checkpointing: bool = True
    plain_tensors: bool = False
    training_augmentation: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--dataset", choices=("MSD", "MNI", "ADNI", "COBRA"), default="MSD")
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Validation hard-Dice patience; 0 disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0005)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-rate", type=float, default=0.0,
                        help="Swin embedding/MLP/projection dropout; attention dropout and stochastic depth stay zero.")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True, help="Recompute activations to save GPU memory.")
    parser.add_argument("--plain-tensors", action="store_true",
                        help="Discard transform metadata after preprocessing to avoid model dispatch overhead.")
    parser.add_argument("--optim-mode", choices=("adamw_0.01", "nnunetv2"), default="adamw_0.01")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--adamw-gamma", type=float, default=0.5)
    parser.add_argument("--supervised-loss", choices=("dice", "dice_ce"), default="dice")
    parser.add_argument("--ce-weight", type=float, default=1.0,
                        help="CE coefficient in Dice + CE; used only for dice_ce.")
    parser.add_argument("--calibration-diagnostics", action="store_true",
                        help="Log FP32 NLL, Brier, ECE and confidence by validation stratum.")

    parser.add_argument(
        "--constraint-set",
        choices=CONSTRAINT_CHOICES,
        default="equivariance",
        help="Constraint preset. 'translation' is a deprecated alias for 'equivariance'.",
    )
    parser.add_argument("--equivariance-weight", type=float, default=None)
    parser.add_argument("--teacher-weight", type=float, default=None,
                        help="Experimental translation-teacher KL weight, chosen by a training-only audit.")
    parser.add_argument("--teacher-views", type=int, default=2)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-support", choices=("union", "common"), default="union")
    parser.add_argument("--translation-augmentation", action="store_true",
                        help="Compute-matched identity plus one supervised uniform +/-2 axis view; training only.")
    parser.add_argument("--training-augmentation", choices=("none", "mild_v1"), default="none",
                        help="Single-view spatial/intensity augmentation; training only, constraint-set none, batch 1.")
    parser.add_argument("--ap-cut-weight", type=float, default=None,
                        help="Experimental structured A/P posterior weight, chosen by a training-only audit.")
    parser.add_argument("--ap-axis", type=int, choices=(0, 1, 2), default=None,
                        help="Explicit spatial tensor axis; verify against transformed training labels.")
    parser.add_argument("--ap-anterior-side", choices=("low", "high"), default=None)
    parser.add_argument("--ap-temperature", type=float, default=1.0)
    parser.add_argument(
        "--ap-plane-weight",
        type=float,
        default=None,
        help="Positive weight from the selected training-only A/P-plane calibration.",
    )
    parser.add_argument(
        "--ap-plane-calibration-json",
        type=Path,
        default=None,
        help="Completed calibration required by ap_plane or ap_plane_location.",
    )
    parser.add_argument(
        "--ap-plane-axis",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Stored coronal tensor axis; Task04 uses axis 1.",
    )
    parser.add_argument(
        "--ap-plane-anterior-side",
        choices=("low", "high"),
        default=None,
        help="Stored anterior direction; Task04 uses high.",
    )
    parser.add_argument("--ap-plane-margin", type=float, default=0.0)
    parser.add_argument(
        "--perimeter-profile-weight",
        type=float,
        default=None,
        help="Classwise multi-axis perimeter-profile weight from a training-only audit.",
    )
    parser.add_argument(
        "--ray-moment-weight",
        type=float,
        default=None,
        help="Three-view M0 ray-thickness weight from a training-only audit.",
    )
    parser.add_argument("--translation-size", type=int, default=2)
    parser.add_argument("--equivariance-max-samples", type=int, default=1,
                        help="Maximum transformed samples per training batch; 0 uses the whole batch.")
    parser.add_argument(
        "--bands-weight",
        type=float,
        default=None,
        help="Positive calibrated weight required by --constraint-set bands.",
    )
    parser.add_argument(
        "--bands-calibration-json",
        type=Path,
        default=None,
        help="Completed calibration report required by --constraint-set bands.",
    )
    parser.add_argument(
        "--band-steps",
        type=int,
        default=2,
        help="Fixed at 2 for the canonical bands experiment.",
    )
    parser.add_argument(
        "--bands-focal-gamma",
        type=float,
        default=0.0,
        help="Focal exponent for band supervision; 0 preserves the BCE formulation.",
    )
    parser.add_argument(
        "--bands-inner-focal-gamma",
        type=float,
        default=None,
        help="Optional inner-band focal exponent; defaults to --bands-focal-gamma.",
    )
    parser.add_argument(
        "--bands-outer-focal-gamma",
        type=float,
        default=None,
        help="Optional outer-band focal exponent; defaults to --bands-focal-gamma.",
    )
    parser.add_argument(
        "--bands-loss-type",
        choices=("focal_bce", "class_tversky"),
        default="focal_bce",
        help="Boundary objective: grouped focal BCE or class-aware boundary Tversky.",
    )
    parser.add_argument("--tversky-fp-weight", type=float, default=0.60)
    parser.add_argument("--tversky-fn-weight", type=float, default=0.40)
    parser.add_argument("--foreground-class-ids", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--complement-class-ids", type=int, nargs="+", default=(0,))
    parser.add_argument(
        "--onecut-weight",
        type=float,
        default=None,
        help="Positive calibrated weight required by --constraint-set onecut.",
    )
    parser.add_argument(
        "--onecut-calibration-json",
        type=Path,
        default=None,
        help="Completed training-only calibration required by --constraint-set onecut.",
    )
    parser.add_argument("--onecut-spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--onecut-radius-mm", type=float, default=3.0)
    parser.add_argument("--onecut-ray-step-mm", type=float, default=0.5)
    parser.add_argument("--onecut-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--onecut-margin", type=float, default=0.0)
    parser.add_argument("--onecut-temperature", type=float, default=1.0)
    parser.add_argument("--onecut-max-surface-points", type=int, default=4096)
    parser.add_argument("--constraint-warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--constraint-scale-knots",
        type=float,
        nargs="+",
        default=None,
        metavar=("EPOCH", "SCALE"),
        help=(
            "Optional piecewise-linear constraint multiplier as EPOCH SCALE pairs. "
            "Requires --constraint-warmup-epochs 0; the endpoint scales are held "
            "before the first and after the last knot."
        ),
    )
    parser.add_argument("--constraint-eval-every", type=int, default=5,
                        help="Evaluate validation constraint metrics every N epochs; 0 means final only.")
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Record non-optimizing boundary focus and component-gradient diagnostics.",
    )
    parser.add_argument(
        "--telemetry-probe-epochs",
        type=int,
        nargs="+",
        default=(1, 3, 5, 8, 10, 12, 14, 16, 18, 20, 25, 30),
        help="Epochs for fixed-case component-gradient and spatial probes.",
    )
    parser.add_argument("--telemetry-probe-cases", type=int, default=2)
    parser.add_argument("--telemetry-spatial-cases", type=int, default=2)

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Optional baseline model.pt used only to initialize a new run.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="hippopotamus-project")
    parser.add_argument("--wandb-entity", default="focacciafilippo-bocconi-university")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_experiment_generators(
    seed: int,
) -> tuple[torch.Generator, torch.Generator, int]:
    """Create independent streams for data order and translation sampling."""

    translation_seed = seed + 1
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    translation_generator = torch.Generator()
    translation_generator.manual_seed(translation_seed)
    return data_generator, translation_generator, translation_seed


def rng_state(
    data_generator: torch.Generator,
    translation_generator: torch.Generator,
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "data_generator": data_generator.get_state(),
        "translation_generator": translation_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: dict[str, Any],
    data_generator: torch.Generator,
    translation_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, dict) or not isinstance(
        numpy_state.get("keys"), torch.Tensor
    ):
        raise ValueError("Resume checkpoint NumPy RNG state is malformed.")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            _exact_int(numpy_state["position"], "rng.numpy.position"),
            _exact_int(numpy_state["has_gauss"], "rng.numpy.has_gauss"),
            _finite_float(
                numpy_state["cached_gaussian"], "rng.numpy.cached_gaussian"
            ),
        )
    )
    torch.set_rng_state(state["torch"])
    data_generator.set_state(state["data_generator"])
    translation_generator.set_state(state["translation_generator"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def resolve_constraint_config(args: argparse.Namespace) -> NewConstraintConfig:
    selected = "equivariance" if args.constraint_set == "translation" else args.constraint_set
    teacher_override = getattr(args, "teacher_weight", None)
    ap_override = getattr(args, "ap_cut_weight", None)
    ap_plane_override = getattr(args, "ap_plane_weight", None)
    ap_plane_calibration = getattr(args, "ap_plane_calibration_json", None)
    perimeter_profile_override = getattr(args, "perimeter_profile_weight", None)
    ray_moment_override = getattr(args, "ray_moment_weight", None)
    if getattr(args, "translation_augmentation", False) and selected not in {"none", "equivariance", "teacher"}:
        raise ValueError("Translation augmentation supports none, equivariance, or teacher only.")
    if (
        getattr(args, "translation_augmentation", False)
        and selected == "equivariance"
        and getattr(args, "translation_size", 2) != 2
    ):
        raise ValueError("Translation augmentation with equivariance requires --translation-size 2.")
    plane_presets = {"ap_plane", "ap_plane_location", "ap_plane_ce_control"}
    for preset, override, flag in (
        ("teacher", teacher_override, "--teacher-weight"),
        ("ap_cut", ap_override, "--ap-cut-weight"),
        (
            "perimeter_profile",
            perimeter_profile_override,
            "--perimeter-profile-weight",
        ),
        ("ray_moment", ray_moment_override, "--ray-moment-weight"),
    ):
        if selected != preset and override not in (None, 0.0):
            raise ValueError(f"{flag} requires --constraint-set {preset}.")
        if selected == preset and (
            override is None or not math.isfinite(override) or override <= 0
        ):
            raise ValueError(f"--constraint-set {preset} requires a finite positive {flag}.")
    if selected not in plane_presets and ap_plane_override not in (None, 0.0):
        raise ValueError(
            "--ap-plane-weight requires an A/P-plane constraint preset."
        )
    if selected in plane_presets and (
        ap_plane_override is None
        or not math.isfinite(ap_plane_override)
        or ap_plane_override <= 0
    ):
        raise ValueError(
            f"--constraint-set {selected} requires a finite positive --ap-plane-weight."
        )
    if selected in {"bands", "onecut", *plane_presets} and getattr(args, "supervised_loss", "dice") != "dice":
        raise ValueError(
            "Existing bands/onecut calibration reports use Dice-only gradients; "
            "they cannot authorize a Dice+CE constraint run."
        )
    if selected == "ap_cut" and (
        getattr(args, "ap_axis", None) is None
        or getattr(args, "ap_anterior_side", None) is None
    ):
        raise ValueError("ap_cut requires explicit --ap-axis and --ap-anterior-side.")
    if selected in plane_presets and (
        getattr(args, "ap_plane_axis", None) is None
        or getattr(args, "ap_plane_anterior_side", None) is None
    ):
        raise ValueError(
            f"{selected} requires explicit --ap-plane-axis and "
            "--ap-plane-anterior-side."
        )
    if selected in plane_presets and ap_plane_calibration is None:
        raise ValueError(f"{selected} requires --ap-plane-calibration-json.")
    if selected not in plane_presets and ap_plane_calibration is not None:
        raise ValueError(
            "--ap-plane-calibration-json requires an A/P-plane constraint preset."
        )
    equivariance_override = getattr(args, "equivariance_weight", None)
    bands_override = getattr(args, "bands_weight", None)
    bands_calibration = getattr(args, "bands_calibration_json", None)
    onecut_override = getattr(args, "onecut_weight", None)
    onecut_calibration = getattr(args, "onecut_calibration_json", None)
    bands_loss_type = str(getattr(args, "bands_loss_type", "focal_bce"))
    tversky_fp_weight = float(getattr(args, "tversky_fp_weight", 0.60))
    tversky_fn_weight = float(getattr(args, "tversky_fn_weight", 0.40))
    bands_focal_gamma = float(getattr(args, "bands_focal_gamma", 0.0))
    inner_override = getattr(args, "bands_inner_focal_gamma", None)
    outer_override = getattr(args, "bands_outer_focal_gamma", None)
    bands_inner_focal_gamma = (
        bands_focal_gamma if inner_override is None else float(inner_override)
    )
    bands_outer_focal_gamma = (
        bands_focal_gamma if outer_override is None else float(outer_override)
    )
    for name, gamma in (
        ("--bands-focal-gamma", bands_focal_gamma),
        ("--bands-inner-focal-gamma", bands_inner_focal_gamma),
        ("--bands-outer-focal-gamma", bands_outer_focal_gamma),
    ):
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    focal_enabled = bands_inner_focal_gamma != 0.0 or bands_outer_focal_gamma != 0.0
    if bands_loss_type not in {"focal_bce", "class_tversky"}:
        raise ValueError("--bands-loss-type must be focal_bce or class_tversky.")
    if bands_loss_type == "class_tversky" and focal_enabled:
        raise ValueError("--bands-loss-type class_tversky cannot use focal exponents.")
    for name, value in (
        ("--tversky-fp-weight", tversky_fp_weight),
        ("--tversky-fn-weight", tversky_fn_weight),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isclose(
        tversky_fp_weight + tversky_fn_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Tversky FP and FN weights must sum to 1.")
    if selected in {
        "none", "teacher", "ap_cut", "ap_plane", "ap_plane_location",
        "ap_plane_ce_control",
        "perimeter_profile", "ray_moment"
    }:
        if (
            equivariance_override not in (None, 0.0)
            or bands_override not in (None, 0.0)
            or bands_calibration is not None
            or onecut_override not in (None, 0.0)
            or onecut_calibration is not None
            or focal_enabled
        ):
            raise ValueError(
                f"--constraint-set {selected} cannot be combined with a legacy constraint weight."
            )
        equivariance_weight = 0.0
        bands_weight = 0.0
        onecut_weight = 0.0
    elif selected == "equivariance":
        if (
            bands_override not in (None, 0.0)
            or bands_calibration is not None
            or focal_enabled
            or onecut_override not in (None, 0.0)
            or onecut_calibration is not None
        ):
            raise ValueError(
                "--constraint-set equivariance cannot be combined with --bands-weight."
            )
        equivariance_weight = (
            CONSTRAINT_WEIGHTS["equivariance"]
            if equivariance_override is None
            else equivariance_override
        )
        if not math.isfinite(equivariance_weight) or equivariance_weight <= 0:
            raise ValueError(
                "--constraint-set equivariance requires a finite positive "
                "--equivariance-weight."
            )
        bands_weight = 0.0
        onecut_weight = 0.0
    elif selected == "bands":
        if (
            equivariance_override not in (None, 0.0)
            or onecut_override not in (None, 0.0)
            or onecut_calibration is not None
        ):
            raise ValueError(
                "--constraint-set bands cannot be combined with --equivariance-weight."
            )
        if bands_override is None:
            raise ValueError(
                "--constraint-set bands requires --bands-weight from training-only "
                "gradient calibration."
            )
        bands_weight = bands_override
        if not math.isfinite(bands_weight) or bands_weight <= 0:
            raise ValueError(
                "--constraint-set bands requires a finite positive --bands-weight."
            )
        if getattr(args, "band_steps", 2) != 2:
            raise ValueError(
                "The canonical bands experiment requires exactly --band-steps 2."
            )
        if bands_calibration is None:
            raise ValueError(
                "--constraint-set bands requires --bands-calibration-json."
            )
        equivariance_weight = 0.0
        onecut_weight = 0.0
    elif selected == "onecut":
        if (
            equivariance_override not in (None, 0.0)
            or bands_override not in (None, 0.0)
            or bands_calibration is not None
            or focal_enabled
        ):
            raise ValueError(
                "--constraint-set onecut cannot be combined with equivariance or band options."
            )
        if onecut_override is None or not math.isfinite(onecut_override) or onecut_override <= 0:
            raise ValueError(
                "--constraint-set onecut requires a finite positive --onecut-weight."
            )
        if onecut_calibration is None:
            raise ValueError(
                "--constraint-set onecut requires --onecut-calibration-json."
            )
        equivariance_weight = 0.0
        bands_weight = 0.0
        onecut_weight = float(onecut_override)
    else:
        raise ValueError(f"Unknown constraint set: {args.constraint_set}")
    return NewConstraintConfig(
        equivariance_weight=equivariance_weight,
        translation_size=getattr(args, "translation_size", 2),
        equivariance_max_samples=(
            None
            if getattr(args, "equivariance_max_samples", None) == 0
            else getattr(args, "equivariance_max_samples", None)
        ),
        bands_weight=bands_weight,
        band_steps=getattr(args, "band_steps", 2),
        bands_focal_gamma=bands_focal_gamma,
        bands_inner_focal_gamma=bands_inner_focal_gamma,
        bands_outer_focal_gamma=bands_outer_focal_gamma,
        bands_loss_type=bands_loss_type,
        tversky_false_positive_weight=tversky_fp_weight,
        tversky_false_negative_weight=tversky_fn_weight,
        foreground_class_ids=tuple(getattr(args, "foreground_class_ids", (1, 2))),
        complement_class_ids=tuple(getattr(args, "complement_class_ids", (0,))),
        onecut_weight=onecut_weight,
        onecut_spacing=tuple(getattr(args, "onecut_spacing", (1.0, 1.0, 1.0))),
        onecut_radius_mm=float(getattr(args, "onecut_radius_mm", 3.0)),
        onecut_ray_step_mm=float(getattr(args, "onecut_ray_step_mm", 0.5)),
        onecut_tolerance_mm=float(getattr(args, "onecut_tolerance_mm", 1.0)),
        onecut_margin=float(getattr(args, "onecut_margin", 0.0)),
        onecut_temperature=float(getattr(args, "onecut_temperature", 1.0)),
        onecut_max_surface_points=int(
            getattr(args, "onecut_max_surface_points", 4096)
        ),
        onecut_geometry_seed=int(getattr(args, "seed", 0)),
        teacher_weight=float(teacher_override) if selected == "teacher" else 0.0,
        teacher_views=int(getattr(args, "teacher_views", 2)),
        teacher_temperature=float(getattr(args, "teacher_temperature", 1.0)),
        teacher_support=getattr(args, "teacher_support", "union"),
        ap_cut_weight=float(ap_override) if selected == "ap_cut" else 0.0,
        ap_axis=getattr(args, "ap_axis", None) if selected == "ap_cut" else 1,
        ap_anterior_low=getattr(args, "ap_anterior_side", "low") == "low",
        ap_temperature=float(getattr(args, "ap_temperature", 1.0)),
        ap_plane_weight=(
            float(ap_plane_override) if selected in plane_presets else 0.0
        ),
        ap_plane_axis=(
            int(getattr(args, "ap_plane_axis")) if selected in plane_presets else 1
        ),
        ap_plane_anterior_high=(
            getattr(args, "ap_plane_anterior_side", "high") == "high"
        ),
        ap_plane_margin=float(getattr(args, "ap_plane_margin", 0.0)),
        ap_plane_require_both=True,
        ap_plane_mode=(
            "location"
            if selected == "ap_plane_location"
            else "conditional_ce"
            if selected == "ap_plane_ce_control"
            else "existential"
        ),
        perimeter_profile_weight=(
            float(perimeter_profile_override)
            if selected == "perimeter_profile"
            else 0.0
        ),
        ray_moment_weight=(
            float(ray_moment_override) if selected == "ray_moment" else 0.0
        ),
    )


def constraint_warmup_scale(epoch: int, warmup_epochs: int) -> float:
    """Return the multiplier applied to the already weighted constraint loss."""

    if epoch < 1:
        raise ValueError("epoch must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def parse_constraint_scale_knots(
    raw_values: list[float] | tuple[float, ...] | None,
) -> tuple[tuple[int, float], ...]:
    """Validate CLI epoch/scale pairs and return an immutable schedule."""

    if raw_values is None:
        return ()
    if len(raw_values) < 2 or len(raw_values) % 2:
        raise ValueError(
            "--constraint-scale-knots requires one or more EPOCH SCALE pairs."
        )
    knots: list[tuple[int, float]] = []
    previous_epoch = 0
    for epoch_value, scale_value in zip(raw_values[::2], raw_values[1::2]):
        if not math.isfinite(epoch_value) or epoch_value < 1 or not float(epoch_value).is_integer():
            raise ValueError("Constraint scale knot epochs must be finite positive integers.")
        epoch = int(epoch_value)
        if epoch <= previous_epoch:
            raise ValueError("Constraint scale knot epochs must be strictly increasing.")
        if not math.isfinite(scale_value) or scale_value < 0:
            raise ValueError("Constraint scale knot values must be finite and non-negative.")
        knots.append((epoch, float(scale_value)))
        previous_epoch = epoch
    return tuple(knots)


def constraint_scale_for_epoch(
    epoch: int,
    warmup_epochs: int,
    knots: tuple[tuple[int, float], ...] = (),
) -> float:
    """Return the legacy warmup or an explicit piecewise-linear multiplier."""

    if not knots:
        return constraint_warmup_scale(epoch, warmup_epochs)
    if epoch < 1:
        raise ValueError("epoch must be positive.")
    if epoch <= knots[0][0]:
        return knots[0][1]
    for (left_epoch, left_scale), (right_epoch, right_scale) in zip(knots, knots[1:]):
        if epoch <= right_epoch:
            fraction = (epoch - left_epoch) / (right_epoch - left_epoch)
            return left_scale + fraction * (right_scale - left_scale)
    return knots[-1][1]


def validate_three_class_labels(items: list[dict[str, Any]]) -> None:
    """Reject labels that cannot be interpreted exactly as classes 0, 1, and 2."""

    allowed = {0, 1, 2}
    for item in items:
        case_name = str(item.get("case_name", "<unknown>"))
        labels = np.asarray(item["label"])
        if labels.size == 0:
            raise ValueError(f"Case {case_name} has an empty label array.")
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError(f"Case {case_name} has non-numeric labels.")
        try:
            finite = bool(np.isfinite(labels).all())
        except TypeError as error:
            raise ValueError(f"Case {case_name} has invalid labels.") from error
        if not finite:
            raise ValueError(f"Case {case_name} has nonfinite labels.")
        if np.iscomplexobj(labels) or not np.equal(labels, np.rint(labels)).all():
            raise ValueError(f"Case {case_name} has non-integer labels.")
        values = {int(value) for value in np.unique(labels)}
        invalid = sorted(values - allowed)
        if invalid:
            preview = ", ".join(str(value) for value in invalid[:5])
            raise ValueError(
                f"Case {case_name} contains labels outside {{0,1,2}}: {preview}"
            )


def validate_image_label_samples(items: list[dict[str, Any]]) -> None:
    """Validate raw image tensors before independent MONAI transforms can hide defects."""

    validate_three_class_labels(items)
    for item in items:
        case_name = str(item.get("case_name", "<unknown>"))
        image = np.asarray(item["image"])
        label = np.asarray(item["label"])
        if image.ndim != 3 or label.ndim != 3:
            raise ValueError(
                f"Case {case_name} must have raw 3-D image and label arrays; "
                f"found {image.shape} and {label.shape}."
            )
        if image.shape != label.shape:
            raise ValueError(
                f"Case {case_name} image/label shapes differ: "
                f"{image.shape} != {label.shape}."
            )
        if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
            raise ValueError(f"Case {case_name} has a non-real numeric image array.")
        try:
            image_is_finite = bool(np.isfinite(image).all())
        except TypeError as error:
            raise ValueError(f"Case {case_name} has invalid image values.") from error
        if not image_is_finite:
            raise ValueError(f"Case {case_name} has nonfinite image values.")


class PlainTensorTransform:
    """Keep preprocessing unchanged, then discard metadata unused by this trainer."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        result = self.transform(sample)
        for key in ("image", "label"):
            if hasattr(result[key], "as_tensor"):
                result[key] = result[key].as_tensor()
        return result


def build_data(
    args: argparse.Namespace,
    train_generator: torch.Generator,
    *,
    pkl_path: Path | None = None,
    splits_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, int, int, int]:
    dataframe = _load_pkl_dataframe(pkl_path or args.pkl)
    if args.num_classes not in (None, 3):
        raise ValueError(
            f"The new constraints expect exactly three classes, found {args.num_classes}."
        )
    num_classes = 3
    dataset = _build_monai_dataset_from_pkl(
        dataframe,
        args.dataset,
        num_classes,
        spatial_size=tuple(args.spatial_size),
        do_resize=args.resize,
    )
    splits = _load_splits_json(splits_path or args.splits_json)
    if args.fold < 0 or args.fold >= len(splits):
        raise ValueError(f"--fold must be in 0..{len(splits) - 1}.")
    split = splits[args.fold]
    train_list = list(split["train"])
    validation_list = list(split["val"])
    if len(train_list) != len(set(train_list)) or len(validation_list) != len(
        set(validation_list)
    ):
        raise ValueError("Split contains duplicate case names.")
    train_names = set(train_list)
    validation_names = set(validation_list)
    if train_names & validation_names:
        raise ValueError("Training and validation case names overlap.")
    train_patients = {
        patient_id_from_case(case_name, args.dataset) for case_name in train_names
    }
    validation_patients = {
        patient_id_from_case(case_name, args.dataset) for case_name in validation_names
    }
    patient_overlap = train_patients & validation_patients
    if patient_overlap:
        preview = ", ".join(sorted(patient_overlap)[:5])
        raise ValueError(
            f"Training and validation patient IDs overlap: {preview}"
        )

    available_names = {item["case_name"] for item in dataset.data}
    missing = (train_names | validation_names) - available_names
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"The split references {len(missing)} missing cases: {preview}")

    train_items = [item for item in dataset.data if item["case_name"] in train_names]
    validation_items = [
        item for item in dataset.data if item["case_name"] in validation_names
    ]
    if len(train_items) != len(train_names) or len(validation_items) != len(validation_names):
        raise ValueError("Duplicate case names in the dataframe make the split ambiguous.")
    validate_image_label_samples(train_items + validation_items)

    transform = (PlainTensorTransform(dataset.transform)
                 if getattr(args, "plain_tensors", False) else dataset.transform)
    train_dataset = MonaiDataset(data=train_items, transform=transform)
    validation_dataset = MonaiDataset(
        data=validation_items,
        transform=transform,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    return (
        train_loader,
        validation_loader,
        num_classes,
        len(train_dataset),
        len(validation_dataset),
    )


def load_initial_weights(model: SwinUNETR, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)


def update_constraint_totals(
    totals: dict[str, dict[str, Any]],
    results: dict[str, ConstraintResult],
) -> None:
    for name, result in results.items():
        truth = result.truth.detach().float()
        confidence_weighted_agreement = (
            result.details["confidence_weighted_agreement"].detach().float()
        )
        confidence_adherent = result.details["confidence_adherent"].detach().float()
        entry = totals.setdefault(
            name,
            {
                "truth": 0.0,
                "confidence_weighted_agreement": 0.0,
                "confidence_adherent": 0.0,
                "count": 0.0,
                "metric_sums": {},
                "metric_counts": {},
            },
        )
        metric_values = [
            (metric_name, torch.as_tensor(metric_value).detach().float())
            for metric_name, metric_value in result.details.get("metrics", {}).items()
        ]
        # One device-to-host transfer per constraint, instead of synchronizing
        # separately for every scalar. Preserve the original float32 reductions
        # and Python float accumulation exactly.
        def summarize(values: torch.Tensor) -> torch.Tensor:
            # Batch-one diagnostics are already scalars; launching a reduction
            # kernel for each of them adds cost without changing their value.
            return values.reshape(()) if values.numel() == 1 else values.sum()

        summaries = torch.stack([
            summarize(truth), summarize(confidence_weighted_agreement).to(truth.device),
            summarize(confidence_adherent).to(truth.device),
            *(summarize(values).to(truth.device) for _, values in metric_values),
        ]).cpu().tolist()
        entry["truth"] += summaries[0]
        entry["confidence_weighted_agreement"] += summaries[1]
        entry["confidence_adherent"] += summaries[2]
        entry["count"] += float(truth.numel())
        for (metric_name, values), total in zip(metric_values, summaries[3:], strict=True):
            entry["metric_sums"][metric_name] = entry["metric_sums"].get(
                metric_name, 0.0
            ) + total
            entry["metric_counts"][metric_name] = entry["metric_counts"].get(
                metric_name, 0.0
            ) + float(values.numel())


def averaged_constraint_metrics(
    totals: dict[str, dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, entry in totals.items():
        if name in AGREEMENT_CONSTRAINT_NAMES:
            count = max(entry["count"], 1.0)
            metrics[f"{prefix}_{name}_truth"] = entry["truth"] / count
            metrics[f"{prefix}_{name}_confidence_weighted_agreement"] = (
                entry["confidence_weighted_agreement"] / count
            )
            metrics[f"{prefix}_{name}_confidence_adherent"] = (
                entry["confidence_adherent"] / count
            )
        for metric_name, total in entry.get("metric_sums", {}).items():
            metric_count = max(entry["metric_counts"][metric_name], 1.0)
            metrics[f"{prefix}_{name}_{metric_name}"] = total / metric_count
    return metrics


def _update_validation_constraint_batch(
    model: nn.Module,
    objective: NewConstraintObjective,
    batch_index: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    case_names: list[str],
    totals: dict[str, dict[str, Any]],
    *,
    all_translation_shifts: bool,
    detail_rows: list[dict[str, Any]] | None,
) -> None:
    """Accumulate active constraints from an already computed base prediction."""

    if objective.config.bands_weight > 0:
        band_result = objective.bands(logits, labels)
        update_constraint_totals(totals, {"outer_boundary_band": band_result})
        if detail_rows is not None:
            details = band_result.details
            valid = details["valid"].detach().cpu()
            for sample_index, case_name in enumerate(case_names):
                is_valid = bool(valid[sample_index])
                detail_rows.append(
                    {
                        "constraint_name": "outer_boundary_band",
                        "case_name": case_name,
                        "inner_component_label": details.get(
                            "component_labels", ("inner_band", "outer_band")
                        )[0],
                        "outer_component_label": details.get(
                            "component_labels", ("inner_band", "outer_band")
                        )[1],
                        "band_valid": int(is_valid),
                        "band_loss": (
                            float(details["case_loss"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "inner_loss": (
                            float(details["inner_loss"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "outer_loss": (
                            float(details["outer_loss"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "inner_voxels": int(
                            details["inner_voxels"][sample_index].detach().cpu()
                        ),
                        "outer_voxels": int(
                            details["outer_voxels"][sample_index].detach().cpu()
                        ),
                        "edge_touching": int(
                            details["edge_touching"][sample_index].detach().cpu()
                        ),
                    }
                )

    if objective.config.onecut_weight > 0:
        onecut_result = objective.onecut(logits, labels)
        update_constraint_totals(totals, {"outer_onecut": onecut_result})
        if detail_rows is not None:
            details = onecut_result.details
            valid = details["valid"].detach().cpu()
            for sample_index, case_name in enumerate(case_names):
                is_valid = bool(valid[sample_index])
                detail_rows.append(
                    {
                        "constraint_name": "outer_onecut",
                        "case_name": case_name,
                        "onecut_valid": int(is_valid),
                        "onecut_loss": (
                            float(details["case_loss"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "onecut_truth": (
                            float(details["case_truth"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "onecut_allowed_cut_mass": (
                            float(details["allowed_cut_mass"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "onecut_ray_count": int(details["ray_count"][sample_index].item()),
                        "edge_touching": int(details["edge_touching"][sample_index].item()),
                    }
                )

    if objective.config.ap_cut_weight > 0:
        result = objective.ap_cut(logits, labels)
        update_constraint_totals(totals, {"ap_cut_posterior": result})
        if detail_rows is not None:
            for index, case_name in enumerate(case_names):
                detail_rows.append({
                    "constraint_name": "ap_cut_posterior", "case_name": case_name,
                    "raw_loss": float(result.details["case_loss"][index].detach().cpu()),
                    "ap_cut_abs_error": float(result.value[index].detach().cpu()),
                    "ap_gt_cut_probability": float(result.truth[index].detach().cpu()),
                })

    if objective.config.ap_plane_weight > 0:
        result = objective.ap_plane(logits, labels)
        result_name = {
            "existential": "existential_ap_plane",
            "location": "ap_plane_location",
            "conditional_ce": "ap_conditional_ce_control",
        }[objective.config.ap_plane_mode]
        update_constraint_totals(totals, {result_name: result})
        if detail_rows is not None:
            details = result.details
            for index, case_name in enumerate(case_names):
                if objective.config.ap_plane_mode == "conditional_ce":
                    detail_rows.append({
                        "constraint_name": result_name,
                        "case_name": case_name,
                        "ap_plane_valid": int(details["valid"][index].detach().cpu()),
                        "ap_plane_loss": float(
                            details["case_loss"][index].detach().cpu()
                        ),
                    })
                    continue
                row = {
                    "constraint_name": result_name,
                    "case_name": case_name,
                    "ap_plane_valid": int(details["valid"][index].detach().cpu()),
                    "ap_plane_loss": float(details["case_loss"][index].detach().cpu()),
                    "ap_plane_selected_cut": int(
                        details["selected_cut"][index].detach().cpu()
                    ),
                    "ap_plane_candidate_count": int(
                        details["candidate_count"][index].detach().cpu()
                    ),
                }
                if objective.config.ap_plane_mode == "location":
                    row.update({
                        "ap_plane_target_cut": int(
                            details["target_cut"][index].detach().cpu()
                        ),
                        "ap_plane_cut_abs_error": float(
                            details["cut_abs_error"][index].detach().cpu()
                        ),
                        "ap_plane_gt_disagreement_fraction": float(
                            details["gt_plane_disagreement_fraction"][index]
                            .detach()
                            .cpu()
                        ),
                        "ap_plane_target_cost": float(
                            details["target_cost"][index].detach().cpu()
                        ),
                        "ap_plane_target_cost_gap": float(
                            details["target_cost_gap"][index].detach().cpu()
                        ),
                        "ap_plane_target_cost_gap_fraction": float(
                            details["target_cost_gap_fraction"][index].detach().cpu()
                        ),
                        "ap_plane_raw_selected_cut": int(
                            details["raw_selected_cut"][index].detach().cpu()
                        ),
                        "ap_plane_raw_cut_abs_error": float(
                            details["raw_cut_abs_error"][index].detach().cpu()
                        ),
                        "ap_plane_raw_exact_cut": float(
                            details["raw_exact_cut"][index].detach().cpu()
                        ),
                        "ap_plane_raw_within_one_cut": float(
                            details["raw_within_one_cut"][index].detach().cpu()
                        ),
                        "ap_plane_raw_disagreement_fraction": float(
                            details["raw_plane_disagreement_fraction"][index]
                            .detach()
                            .cpu()
                        ),
                        "ap_plane_raw_has_both_ap_classes": int(
                            details["raw_has_both_ap_classes"][index].detach().cpu()
                        ),
                        "ap_plane_raw_cut_valid": int(
                            details["raw_cut_valid"][index].detach().cpu()
                        ),
                        "foreground_union_dice": float(
                            details["foreground_union_dice"][index].detach().cpu()
                        ),
                        "ap_swap_voxels": int(
                            details["ap_swap_voxels"][index].detach().cpu()
                        ),
                        "ap_swap_fraction": float(
                            details["ap_swap_fraction"][index].detach().cpu()
                        ),
                        "displaced_slab_swap_voxels": int(
                            details["displaced_slab_swap_voxels"][index].detach().cpu()
                        ),
                        "displaced_slab_swap_fraction": float(
                            details["displaced_slab_swap_fraction"][index]
                            .detach()
                            .cpu()
                        ),
                    })
                detail_rows.append(row)

    if objective.config.perimeter_profile_weight > 0:
        result = objective.perimeter_profile(logits, labels)
        update_constraint_totals(totals, {"perimeter_profile": result})
        if detail_rows is not None:
            axis_losses = result.details["axis_loss"].detach().cpu()
            for index, case_name in enumerate(case_names):
                detail_rows.append({
                    "constraint_name": "perimeter_profile",
                    "case_name": case_name,
                    "raw_loss": float(result.details["case_loss"][index].detach().cpu()),
                    "axis_0_loss": float(axis_losses[index, 0]),
                    "axis_1_loss": float(axis_losses[index, 1]),
                    "axis_2_loss": float(axis_losses[index, 2]),
                })

    if objective.config.ray_moment_weight > 0:
        result = objective.ray_moment(logits, labels)
        update_constraint_totals(totals, {"ray_moment": result})
        if detail_rows is not None:
            axis_losses = result.details["axis_loss"].detach().cpu()
            group_losses = result.details["group_loss"].detach().cpu()
            for index, case_name in enumerate(case_names):
                detail_rows.append({
                    "constraint_name": "ray_moment",
                    "case_name": case_name,
                    "raw_loss": float(result.details["case_loss"][index].detach().cpu()),
                    "axis_0_loss": float(axis_losses[index, 0]),
                    "axis_1_loss": float(axis_losses[index, 1]),
                    "axis_2_loss": float(axis_losses[index, 2]),
                    "foreground_loss": float(group_losses[index, 0]),
                    "anterior_loss": float(group_losses[index, 1]),
                    "posterior_loss": float(group_losses[index, 2]),
                })

    if objective.config.teacher_weight > 0:
        shifts = objective.teacher.shifts
        if not all_translation_shifts:
            shifts = tuple(
                shifts[(batch_index * objective.teacher.num_views + i) % len(shifts)]
                for i in range(objective.teacher.num_views)
            )
        result = objective.teacher(model, images, logits, shifts=shifts)
        update_constraint_totals(totals, {"translation_teacher_kl": result})
        if detail_rows is not None:
            for index, case_name in enumerate(case_names):
                detail_rows.append({
                    "constraint_name": "translation_teacher_kl", "case_name": case_name,
                    "raw_loss": float(result.value[index].detach().cpu()),
                    "teacher_views": len(shifts),
                })

    if objective.config.equivariance_weight <= 0:
        return
    shifts = objective.equivariance.shifts
    selected_shifts = (
        shifts if all_translation_shifts else (shifts[batch_index % len(shifts)],)
    )
    for shift in selected_shifts:
        translation_result = objective.equivariance(
            model,
            images,
            logits,
            shift=shift,
        )
        update_constraint_totals(
            totals,
            {"translation_equivariance": translation_result},
        )
        if detail_rows is None:
            continue
        optimization_values = translation_result.details[
            "optimization_classwise_dice"
        ].detach().float().cpu()
        linear_values = translation_result.value.detach().float().cpu()
        case_truth = translation_result.truth.detach().float().cpu()
        case_confidence_weighted_agreement = translation_result.details[
            "confidence_weighted_agreement"
        ].detach().float().cpu()
        case_confidence_adherent = translation_result.details[
            "confidence_adherent"
        ].detach().cpu()
        dx, dy, dz = shift
        for sample_index, case_name in enumerate(case_names):
            for class_index, class_id in enumerate(objective.equivariance.class_ids):
                linear_value = float(linear_values[sample_index, class_index])
                detail_rows.append(
                    {
                        "constraint_name": "translation_equivariance",
                        "case_name": case_name,
                        "shift_dx": dx,
                        "shift_dy": dy,
                        "shift_dz": dz,
                        "class_id": class_id,
                        "optimization_truth": float(
                            optimization_values[sample_index, class_index]
                        ),
                        "legacy_linear_value": linear_value,
                        "class_confidence_adherent": int(
                            linear_value
                            >= objective.equivariance.confidence_agreement_threshold
                        ),
                        "case_direction_truth": float(case_truth[sample_index]),
                        "case_direction_confidence_weighted_agreement": float(
                            case_confidence_weighted_agreement[sample_index]
                        ),
                        "case_direction_confidence_adherent": int(
                            case_confidence_adherent[sample_index]
                        ),
                    }
                )


@torch.no_grad()
def evaluate_constraint_metrics(
    model: SwinUNETR,
    loader: DataLoader,
    objective: NewConstraintObjective,
    device: torch.device,
    *,
    amp: bool,
    all_translation_shifts: bool,
    detail_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Standalone constraint diagnostics; training uses the combined evaluator."""

    if (
        objective.config.equivariance_weight == 0
        and objective.config.bands_weight == 0
        and objective.config.onecut_weight == 0
        and objective.config.teacher_weight == 0
        and objective.config.ap_cut_weight == 0
        and objective.config.ap_plane_weight == 0
        and objective.config.perimeter_profile_weight == 0
        and objective.config.ray_moment_weight == 0
    ):
        return {}
    totals: dict[str, dict[str, Any]] = {}
    model.eval()
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        case_names = [str(name) for name in batch["case_name"]]
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            _update_validation_constraint_batch(
                model,
                objective,
                batch_index,
                images,
                labels,
                logits,
                case_names,
                totals,
                all_translation_shifts=all_translation_shifts,
                detail_rows=detail_rows,
            )
    return averaged_constraint_metrics(totals, "val")


def _segmentation_dice_sums(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[float, float, int]:
    """Return foreground Dice sums and their valid sample/class count."""

    if num_classes < 2:
        raise ValueError("Segmentation Dice requires at least two classes.")
    if labels.ndim == logits.ndim and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    probabilities = torch.softmax(logits.float(), dim=1)
    hard_labels = torch.argmax(probabilities, dim=1)
    targets = torch.nn.functional.one_hot(
        labels.long(),
        num_classes=num_classes,
    ).movedim(-1, 1).float()
    hard_predictions = torch.nn.functional.one_hot(
        hard_labels,
        num_classes=num_classes,
    ).movedim(-1, 1).float()
    spatial_dimensions = tuple(range(2, probabilities.ndim))
    foreground_targets = targets[:, 1:]
    foreground_probabilities = probabilities[:, 1:]
    foreground_hard = hard_predictions[:, 1:]
    soft_intersection = (
        foreground_probabilities * foreground_targets
    ).sum(spatial_dimensions)
    hard_intersection = (foreground_hard * foreground_targets).sum(
        spatial_dimensions
    )
    target_volume = foreground_targets.sum(spatial_dimensions)
    soft_denominator = foreground_probabilities.sum(spatial_dimensions) + target_volume
    hard_denominator = foreground_hard.sum(spatial_dimensions) + target_volume
    # Match MONAI's default ignore-empty convention: a class with no
    # ground-truth voxels does not enter the sample/class macro mean.
    valid_scores = target_volume > 0
    soft_scores = 2.0 * soft_intersection[valid_scores] / soft_denominator[
        valid_scores
    ]
    hard_scores = 2.0 * hard_intersection[valid_scores] / hard_denominator[
        valid_scores
    ]
    return (
        float(soft_scores.sum().cpu()),
        float(hard_scores.sum().cpu()),
        int(valid_scores.sum().item()),
    )


@torch.no_grad()
def evaluate_segmentation_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> tuple[float, float]:
    """Return valid macro soft and hard Dice over foreground classes."""

    soft_total = 0.0
    hard_total = 0.0
    count = 0
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        batch_soft, batch_hard, batch_count = _segmentation_dice_sums(
            model(images), labels, num_classes
        )
        soft_total += batch_soft
        hard_total += batch_hard
        count += batch_count
    if count == 0:
        raise ValueError("The validation loader produced no foreground scores.")
    return soft_total / count, hard_total / count


@torch.no_grad()
def evaluate_validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    objective: NewConstraintObjective,
    device: torch.device,
    *,
    num_classes: int,
    amp: bool,
    evaluate_constraints: bool,
    all_translation_shifts: bool,
    detail_rows: list[dict[str, Any]] | None = None,
    telemetry: BoundaryTelemetryAccumulator | None = None,
    calibration_diagnostics: bool = False,
) -> dict[str, float]:
    """Evaluate segmentation and active constraints from one base forward per batch."""

    soft_total = 0.0
    hard_total = 0.0
    count = 0
    constraint_totals: dict[str, dict[str, Any]] = {}
    has_active_constraints = (
        objective.config.equivariance_weight > 0
        or objective.config.bands_weight > 0
        or objective.config.onecut_weight > 0
        or objective.config.teacher_weight > 0
        or objective.config.ap_cut_weight > 0
        or objective.config.ap_plane_weight > 0
        or objective.config.perimeter_profile_weight > 0
        or objective.config.ray_moment_weight > 0
    )
    calibration = CalibrationDiagnostics(num_classes) if calibration_diagnostics else None
    model.eval()
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        case_names = (
            [str(name) for name in batch["case_name"]]
            if evaluate_constraints and has_active_constraints
            else []
        )
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            if evaluate_constraints and has_active_constraints:
                _update_validation_constraint_batch(
                    model,
                    objective,
                    batch_index,
                    images,
                    labels,
                    logits,
                    case_names,
                    constraint_totals,
                    all_translation_shifts=all_translation_shifts,
                    detail_rows=detail_rows,
                )
        batch_soft, batch_hard, batch_count = _segmentation_dice_sums(
            logits, labels, num_classes
        )
        if telemetry is not None:
            telemetry.update(logits, labels)
        if calibration is not None:
            calibration.update(logits, labels)
        soft_total += batch_soft
        hard_total += batch_hard
        count += batch_count
    if count == 0:
        raise ValueError("The validation loader produced no foreground scores.")
    return {
        "val_dice_soft": soft_total / count,
        "val_dice_hard": hard_total / count,
        **averaged_constraint_metrics(constraint_totals, "val"),
        **(calibration.summary() if calibration is not None else {}),
    }


def save_json(path: Path, payload: Any) -> None:
    """Atomically publish strict JSON so config-only recovery is crash-safe."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Atomically publish a complete CSV artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def telemetry_rows_for_resume(
    path: Path,
    fieldnames: list[str],
    checkpoint_epoch: int,
    *,
    one_row_per_epoch: bool,
) -> list[dict[str, Any]]:
    """Validate and trim telemetry to the last durable checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"Resume requires {path.name}.")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"{path.name} has an incompatible schema.")
        rows = [row for row in reader if int(row["epoch"]) <= checkpoint_epoch]
    epochs = [int(row["epoch"]) for row in rows]
    if one_row_per_epoch and epochs != list(range(1, checkpoint_epoch + 1)):
        raise ValueError(f"{path.name} does not contain every durable epoch exactly once.")
    return rows


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: SwinUNETR,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    best_hard_dice: float,
    best_epoch: int,
    data_generator: torch.Generator,
    translation_generator: torch.Generator,
    run_spec: RunSpec,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (
            scheduler.state_dict()
            if callable(getattr(scheduler, "state_dict", None))
            else None
        ),
        "scaler": scaler.state_dict(),
        "best_hard_dice": best_hard_dice,
        "best_epoch": best_epoch,
        "run": asdict(run_spec),
        "rng": rng_state(data_generator, translation_generator),
    }
    save_checkpoint_payload(path, payload)


def save_checkpoint_payload(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a complete checkpoint payload."""

    save_torch_payload(path, payload)


def save_torch_payload(path: Path, payload: Any) -> None:
    """Atomically publish a torch-serialized artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def metric_fieldnames(calibration_diagnostics: bool = False, num_classes: int = 3) -> list[str]:
    fields = [
        "epoch",
        "train_loss",
        "train_supervised_loss",
        "train_constraint_loss",
        "constraint_scale",
        "learning_rate",
        "val_dice_soft",
        "val_dice_hard",
    ]
    for prefix in ("train", "val"):
        for name in AGREEMENT_CONSTRAINT_NAMES:
            for statistic in (
                "truth",
                "confidence_weighted_agreement",
                "confidence_adherent",
            ):
                fields.append(f"{prefix}_{name}_{statistic}")
        for statistic in BAND_METRICS:
            fields.append(f"{prefix}_outer_boundary_band_{statistic}")
        for statistic in ONECUT_METRICS:
            fields.append(f"{prefix}_outer_onecut_{statistic}")
        for statistic in AP_CUT_METRICS:
            fields.append(f"{prefix}_ap_cut_posterior_{statistic}")
        for statistic in AP_PLANE_METRICS:
            fields.append(f"{prefix}_existential_ap_plane_{statistic}")
        for statistic in AP_PLANE_LOCATION_METRICS:
            fields.append(f"{prefix}_ap_plane_location_{statistic}")
        for statistic in AP_CONDITIONAL_CE_METRICS:
            fields.append(f"{prefix}_ap_conditional_ce_control_{statistic}")
        for statistic in TEACHER_METRICS:
            fields.append(f"{prefix}_translation_teacher_kl_{statistic}")
        for statistic in PERIMETER_PROFILE_METRICS:
            fields.append(f"{prefix}_perimeter_profile_{statistic}")
        for statistic in RAY_MOMENT_METRICS:
            fields.append(f"{prefix}_ray_moment_{statistic}")
    if calibration_diagnostics:
        fields.extend(CalibrationDiagnostics(num_classes).summary())
    return fields


def resolve_resume_run_spec(run_spec: RunSpec, saved_run: dict[str, Any]) -> RunSpec:
    """Restore immutable initialization provenance before strict resume checks."""

    saved_initial_path = saved_run.get("initial_checkpoint")
    saved_initial_digest = saved_run.get("initial_checkpoint_sha256")
    if saved_initial_path is None:
        if saved_initial_digest is not None:
            raise ValueError("Resume initializer digest exists without a checkpoint path.")
    elif (
        not isinstance(saved_initial_path, str)
        or not saved_initial_path
        or not isinstance(saved_initial_digest, str)
        or len(saved_initial_digest) != 64
    ):
        raise ValueError("Resume initializer lacks immutable SHA-256 provenance.")
    current_run = json.loads(json.dumps(asdict(run_spec)))
    current_run["initial_checkpoint"] = saved_initial_path
    current_run["initial_checkpoint_sha256"] = saved_initial_digest
    canonical_saved_run = json.loads(json.dumps(saved_run))
    if canonical_saved_run != current_run:
        raise ValueError("Resume configuration does not match config.json.")
    return replace(
        run_spec,
        initial_checkpoint=saved_initial_path,
        initial_checkpoint_sha256=saved_initial_digest,
    )


def validate_input_file_provenance(
    config: dict[str, Any],
    pkl_path: Path,
    splits_path: Path,
    *,
    pkl_digest: str | None = None,
    splits_digest: str | None = None,
) -> None:
    """Require both input paths and contents to match the recorded run."""

    expected = (
        ("pkl", "pkl_sha256", pkl_path, pkl_digest),
        ("splits_json", "splits_json_sha256", splits_path, splits_digest),
    )
    for path_key, digest_key, current_path, current_digest in expected:
        recorded_path = config.get(path_key)
        if not isinstance(recorded_path, str) or (
            Path(recorded_path).expanduser().resolve() != current_path.resolve()
        ):
            raise ValueError(f"Resume {path_key} path does not match config.json.")
        recorded_digest = config.get(digest_key)
        if not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
            raise ValueError(
                f"config.json lacks valid {digest_key} provenance; start a fresh run."
            )
        actual_digest = current_digest or file_sha256(current_path)
        if actual_digest != recorded_digest:
            raise ValueError(f"Resume {path_key} contents changed after run creation.")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _calibration_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "nonfinite_count": 0,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def validate_bands_calibration_report(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    report_snapshot: FileSnapshot,
    pkl_digest: str,
    splits_digest: str,
    source_provenance: dict[str, Any],
    runtime_provenance: dict[str, Any],
    execution_provenance: dict[str, Any],
    splits_path: Path | None = None,
) -> None:
    """Bind a bands run to the exact completed training-only calibration report."""

    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ValueError("Bands calibration report is not complete.")
    if report.get("constraint_set") != "bands":
        raise ValueError("Bands calibration report has the wrong constraint set.")
    if report.get("dataset") != args.dataset or report.get("fold") != args.fold:
        raise ValueError("Bands calibration dataset/fold does not match this run.")
    augmentation = getattr(args, "training_augmentation", "none")
    if report.get("training_augmentation", "none") != augmentation:
        raise ValueError("Bands calibration training_augmentation does not match this run.")
    if augmentation == "mild_v1" and (
        report.get("augmentation_policy") != MILD_V1
        or report.get("augmentation_seed") != report.get("seed", -1) + 1
    ):
        raise ValueError("Bands calibration augmentation policy/RNG does not match.")
    if report.get("band_steps") != args.band_steps:
        raise ValueError("Bands calibration step count does not match this run.")
    if report.get("bands_loss_type", "focal_bce") != getattr(
        args, "bands_loss_type", "focal_bce"
    ):
        raise ValueError("Bands calibration loss type does not match this run.")
    if report.get("tversky_false_positive_weight", 0.60) != float(
        getattr(args, "tversky_fp_weight", 0.60)
    ) or report.get("tversky_false_negative_weight", 0.40) != float(
        getattr(args, "tversky_fn_weight", 0.40)
    ):
        raise ValueError("Bands calibration Tversky weights do not match this run.")
    if report.get("bands_focal_gamma", 0.0) != float(
        getattr(args, "bands_focal_gamma", 0.0)
    ):
        raise ValueError("Bands calibration focal gamma does not match this run.")
    shared_gamma = float(getattr(args, "bands_focal_gamma", 0.0))
    expected_inner_gamma = getattr(args, "bands_inner_focal_gamma", None)
    expected_outer_gamma = getattr(args, "bands_outer_focal_gamma", None)
    expected_inner_gamma = (
        shared_gamma if expected_inner_gamma is None else float(expected_inner_gamma)
    )
    expected_outer_gamma = (
        shared_gamma if expected_outer_gamma is None else float(expected_outer_gamma)
    )
    if report.get("bands_inner_focal_gamma", report.get("bands_focal_gamma", 0.0)) != expected_inner_gamma:
        raise ValueError("Bands calibration inner focal gamma does not match this run.")
    if report.get("bands_outer_focal_gamma", report.get("bands_focal_gamma", 0.0)) != expected_outer_gamma:
        raise ValueError("Bands calibration outer focal gamma does not match this run.")
    if report.get("amp") is not bool(args.amp):
        raise ValueError("Bands calibration AMP mode does not match this run.")
    if report.get("foreground_class_ids") != list(args.foreground_class_ids) or report.get(
        "complement_class_ids"
    ) != list(args.complement_class_ids):
        raise ValueError("Bands calibration class grouping does not match this run.")
    calibrated_weight = report.get("recommended_bands_weight")
    if (
        isinstance(calibrated_weight, bool)
        or not isinstance(calibrated_weight, (int, float))
        or not math.isfinite(float(calibrated_weight))
        or float(calibrated_weight) <= 0
        or float(calibrated_weight) != float(args.bands_weight)
    ):
        raise ValueError("--bands-weight must exactly match the calibration report.")
    if report.get("target_ratio") != 0.10 or report.get("safety_ratio") != 0.50:
        raise ValueError("Bands calibration uses a noncanonical gradient-ratio policy.")
    cases = report.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("Bands calibration must contain 1..32 training-case records.")
    case_names = [
        row.get("case_name") if isinstance(row, dict) else None for row in cases
    ]
    if (
        any(not isinstance(name, str) or not name for name in case_names)
        or len(case_names) != len(set(case_names))
        or report.get("training_case_ids") != case_names
    ):
        raise ValueError("Bands calibration case identity records are malformed.")
    splits = _load_splits_json(splits_path or args.splits_json)
    if args.fold < 0 or args.fold >= len(splits):
        raise ValueError("Bands calibration fold is outside the split file.")
    training_names = set(splits[args.fold]["train"])
    if not set(case_names).issubset(training_names):
        raise ValueError("Bands calibration contains held-out or unavailable cases.")
    seed = _exact_int(report.get("seed"), "calibration seed")
    max_cases = _exact_int(report.get("max_cases"), "calibration max_cases")
    if not 1 <= max_cases <= 32:
        raise ValueError("Bands calibration max_cases is outside 1..32.")
    expected_case_names = sorted(
        random.Random(seed).sample(
            sorted(training_names), min(max_cases, len(training_names))
        )
    )
    if case_names != expected_case_names:
        raise ValueError("Bands calibration cases are not the deterministic training sample.")
    expected_patient_ids = [
        patient_id_from_case(name, args.dataset) for name in case_names
    ]
    if report.get("training_cases") != [
        {"case_name": name, "patient_id": patient_id}
        for name, patient_id in zip(case_names, expected_patient_ids)
    ] or report.get("training_patient_ids") != sorted(set(expected_patient_ids)):
        raise ValueError("Bands calibration patient identity records are malformed.")
    dice_rms: list[float] = []
    band_rms: list[float] = []
    conditional_band_rms: list[float] = []
    for row, expected_patient_id in zip(cases, expected_patient_ids):
        if not isinstance(row, dict):
            raise ValueError("Bands calibration contains a malformed case record.")
        inner_voxels = _exact_int(row.get("inner_voxels"), "calibration inner_voxels")
        outer_voxels = _exact_int(row.get("outer_voxels"), "calibration outer_voxels")
        if inner_voxels < 0 or outer_voxels < 0:
            raise ValueError("Bands calibration voxel counts must be non-negative.")
        if row.get("patient_id") != expected_patient_id or not isinstance(
            row.get("edge_touching"), bool
        ):
            raise ValueError("Bands calibration case diagnostics are malformed.")
        expected_valid = inner_voxels > 0 and outer_voxels > 0
        if row.get("valid") is not expected_valid:
            raise ValueError("Bands calibration validity disagrees with band voxel counts.")
        if row.get("valid") is True:
            dice_rms.append(_finite_float(row.get("dice_gradient_rms"), "calibration Dice RMS"))
            band_rms.append(
                _finite_float(
                    row.get("band_gradient_rms_unconditional"),
                    "calibration band RMS",
                )
            )
            conditional_band_rms.append(
                _finite_float(
                    row.get("band_gradient_rms_conditional"),
                    "calibration conditional band RMS",
                )
            )
    if not dice_rms or len(dice_rms) != len(band_rms):
        raise ValueError("Bands calibration contains no valid gradient pairs.")
    valid_count = len(dice_rms)
    if (
        report.get("valid_case_count") != valid_count
        or report.get("skipped_case_count") != len(cases) - valid_count
        or report.get("valid_case_fraction") != valid_count / len(cases)
        or report.get("dice_gradient_rms") != _calibration_summary(dice_rms)
        or report.get("band_gradient_rms_unconditional")
        != _calibration_summary(band_rms)
        or report.get("band_gradient_rms_conditional")
        != _calibration_summary(conditional_band_rms)
    ):
        raise ValueError("Bands calibration summary diagnostics disagree with case data.")
    dice_median = float(np.median(np.asarray(dice_rms, dtype=np.float64)))
    band_array = np.asarray(band_rms, dtype=np.float64)
    band_median = float(np.median(band_array))
    band_q95 = float(np.quantile(band_array, 0.95))
    if dice_median <= 0 or band_median <= 0 or band_q95 <= 0:
        raise ValueError("Bands calibration gradient magnitudes must be positive.")
    recomputed_target = 0.10 * dice_median / band_median
    recomputed_cap = 0.50 * dice_median / band_q95
    recomputed_weight = min(recomputed_target, recomputed_cap)
    reported_values = (
        ("target_weight", recomputed_target),
        ("cap_weight", recomputed_cap),
        ("recommended_bands_weight", recomputed_weight),
    )
    for key, expected_value in reported_values:
        actual_value = _finite_float(report.get(key), f"calibration {key}")
        if not math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"Bands calibration {key} does not match its case data.")
    report_source = report.get("source_provenance")
    if not isinstance(report_source, dict) or report_source.get("sha256") != source_provenance.get(
        "sha256"
    ):
        raise ValueError("Bands calibration source code differs from this run.")
    report_runtime = report.get("runtime_provenance")
    if not isinstance(report_runtime, dict) or canonical_sha256(
        report_runtime
    ) != canonical_sha256(runtime_provenance):
        raise ValueError("Bands calibration runtime differs from this run.")
    report_execution = report.get("execution_provenance")
    if not isinstance(report_execution, dict) or canonical_sha256(
        report_execution
    ) != canonical_sha256(execution_provenance):
        raise ValueError("Bands calibration compute device differs from this run.")
    expected_inputs = (
        ("pkl", args.pkl, "pkl_sha256", pkl_digest),
        ("splits_json", args.splits_json, "splits_json_sha256", splits_digest),
    )
    for path_key, current_path, digest_key, current_digest in expected_inputs:
        recorded_path = report.get(path_key)
        if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != current_path.resolve():
            raise ValueError(f"Bands calibration {path_key} path does not match this run.")
        if report.get(digest_key) != current_digest:
            raise ValueError(f"Bands calibration {path_key} contents do not match this run.")
    checkpoint_value = report.get("checkpoint")
    config_value = report.get("checkpoint_config")
    checkpoint_source = report.get("checkpoint_source_provenance")
    checkpoint_source_sha256 = report.get("checkpoint_source_sha256")
    if (
        not isinstance(checkpoint_source, dict)
        or checkpoint_source.get("sha256") != checkpoint_source_sha256
    ):
        raise ValueError("Bands calibration lacks valid checkpoint source provenance.")
    if not isinstance(checkpoint_value, str) or not _valid_sha256(
        report.get("checkpoint_sha256")
    ):
        raise ValueError("Bands calibration lacks valid checkpoint provenance.")
    if not isinstance(config_value, str) or not _valid_sha256(
        report.get("checkpoint_config_sha256")
    ):
        raise ValueError("Bands calibration lacks valid checkpoint_config provenance.")
    checkpoint_path = Path(checkpoint_value).expanduser()
    checkpoint_config_path = Path(config_value).expanduser()
    if not checkpoint_path.is_file() or not checkpoint_config_path.is_file():
        raise ValueError("Bands calibration source checkpoint/config is unavailable.")
    with snapshot_file(checkpoint_path) as checkpoint_snapshot, snapshot_file(
        checkpoint_config_path
    ) as config_snapshot:
        if checkpoint_snapshot.sha256 != report["checkpoint_sha256"]:
            raise ValueError("Bands calibration checkpoint contents changed.")
        if config_snapshot.sha256 != report["checkpoint_config_sha256"]:
            raise ValueError("Bands calibration checkpoint_config contents changed.")
        source_config = load_bands_calibration_report(config_snapshot)
        source_run = source_config.get("run")
        if (
            not isinstance(source_run, dict)
            or source_run.get("constraint_set") != "none"
            or source_run.get("supervised_loss", "dice") != "dice"
            or source_run.get("ce_weight", 0.0) != 0.0
            or source_run.get("dataset") != args.dataset
            or source_run.get("fold") != args.fold
            or source_run.get("epochs") != 5
            or source_run.get("amp") is not bool(args.amp)
            or source_run.get("training_augmentation", "none") != augmentation
            or source_run.get("pkl_sha256") != pkl_digest
            or source_run.get("splits_json_sha256") != splits_digest
            or source_run.get("source_sha256") != checkpoint_source_sha256
            or source_run.get("runtime_sha256") != canonical_sha256(runtime_provenance)
            or source_run.get("execution_sha256")
            != canonical_sha256(execution_provenance)
        ):
            raise ValueError("Bands calibration source run configuration is inconsistent.")
        # Legacy reports use their verified source configuration for preprocessing.
        for field, expected in (
            ("spatial_size", list(args.spatial_size)), ("resize", bool(args.resize)),
        ):
            recorded = source_run.get(field, False if field == "resize" else None)
            if field == "spatial_size" and recorded is not None:
                recorded = list(recorded)
            if recorded != expected or report.get(field, recorded) != expected:
                raise ValueError(f"Bands calibration {field} does not match this run.")
        if source_config.get("source_provenance") != checkpoint_source:
            raise ValueError("Bands calibration checkpoint source provenance changed.")
        source_checkpoint = torch.load(
            checkpoint_snapshot.path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(source_checkpoint, dict)
            or source_checkpoint.get("epoch") != 5
            or json.loads(json.dumps(source_checkpoint.get("run"))) != source_run
            or report.get("checkpoint_epoch") != source_checkpoint.get("epoch")
            or report.get("checkpoint_constraint_set")
            != source_run.get("constraint_set")
        ):
            raise ValueError("Bands calibration source checkpoint is inconsistent.")
    if not _valid_sha256(report_snapshot.sha256):
        raise ValueError("Bands calibration report snapshot digest is malformed.")


def load_bands_calibration_report(snapshot: FileSnapshot) -> dict[str, Any]:
    try:
        report = json.loads(snapshot.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read bands calibration report: {error}") from error
    if not isinstance(report, dict):
        raise ValueError("Bands calibration report must contain a JSON object.")
    return report


def validate_onecut_calibration_report(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    report_snapshot: FileSnapshot,
    pkl_digest: str,
    splits_digest: str,
    source_provenance: dict[str, Any],
    runtime_provenance: dict[str, Any],
    execution_provenance: dict[str, Any],
    splits_path: Path | None = None,
) -> None:
    """Bind a one-cut run to its exact training-only gradient calibration."""

    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ValueError("One-cut calibration report is not complete.")
    if report.get("constraint_set") != "onecut":
        raise ValueError("One-cut calibration report has the wrong constraint set.")
    if report.get("dataset") != args.dataset or report.get("fold") != args.fold:
        raise ValueError("One-cut calibration dataset/fold does not match this run.")
    expected_geometry = {
        "spacing": list(args.onecut_spacing),
        "radius_mm": float(args.onecut_radius_mm),
        "ray_step_mm": float(args.onecut_ray_step_mm),
        "tolerance_mm": float(args.onecut_tolerance_mm),
        "margin": float(args.onecut_margin),
        "temperature": float(args.onecut_temperature),
        "max_surface_points": int(args.onecut_max_surface_points),
        "geometry_seed": int(args.seed),
    }
    if report.get("onecut_geometry") != expected_geometry:
        raise ValueError("One-cut calibration geometry does not match this run.")
    if report.get("amp") is not bool(args.amp):
        raise ValueError("One-cut calibration AMP mode does not match this run.")
    if report.get("foreground_class_ids") != list(args.foreground_class_ids) or report.get(
        "complement_class_ids"
    ) != list(args.complement_class_ids):
        raise ValueError("One-cut calibration class grouping does not match this run.")
    weight = report.get("recommended_onecut_weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) <= 0
        or float(weight) != float(args.onecut_weight)
    ):
        raise ValueError("--onecut-weight must exactly match the calibration report.")
    if report.get("target_ratio") != 0.10 or report.get("safety_ratio") != 0.50:
        raise ValueError("One-cut calibration uses a noncanonical gradient-ratio policy.")
    cases = report.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("One-cut calibration must contain 1..32 case records.")
    case_names = [row.get("case_name") if isinstance(row, dict) else None for row in cases]
    if any(not isinstance(name, str) or not name for name in case_names):
        raise ValueError("One-cut calibration case identities are malformed.")
    splits = _load_splits_json(splits_path or args.splits_json)
    training_names = set(splits[args.fold]["train"])
    seed = _exact_int(report.get("seed"), "onecut calibration seed")
    max_cases = _exact_int(report.get("max_cases"), "onecut calibration max_cases")
    expected_names = sorted(
        random.Random(seed).sample(
            sorted(training_names), min(max_cases, len(training_names))
        )
    )
    if case_names != expected_names or report.get("training_case_ids") != expected_names:
        raise ValueError("One-cut calibration is not the deterministic training sample.")
    dice_rms: list[float] = []
    onecut_rms: list[float] = []
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("One-cut calibration contains a malformed case record.")
        ray_count = _exact_int(row.get("ray_count"), "onecut calibration ray_count")
        expected_valid = ray_count > 0
        if row.get("valid") is not expected_valid or not isinstance(
            row.get("edge_touching"), bool
        ):
            raise ValueError("One-cut calibration validity diagnostics are inconsistent.")
        if expected_valid:
            dice_rms.append(
                _finite_float(row.get("dice_gradient_rms"), "onecut calibration Dice RMS")
            )
            onecut_rms.append(
                _finite_float(
                    row.get("onecut_gradient_rms_unconditional"),
                    "onecut calibration gradient RMS",
                )
            )
    if not dice_rms or len(dice_rms) != len(onecut_rms):
        raise ValueError("One-cut calibration has no valid gradient pairs.")
    valid_count = len(dice_rms)
    if (
        report.get("valid_case_count") != valid_count
        or report.get("skipped_case_count") != len(cases) - valid_count
        or report.get("dice_gradient_rms") != _calibration_summary(dice_rms)
        or report.get("onecut_gradient_rms_unconditional")
        != _calibration_summary(onecut_rms)
    ):
        raise ValueError("One-cut calibration summaries disagree with case data.")
    dice_median = float(np.median(np.asarray(dice_rms, dtype=np.float64)))
    onecut_array = np.asarray(onecut_rms, dtype=np.float64)
    target = 0.10 * dice_median / float(np.median(onecut_array))
    cap = 0.50 * dice_median / float(np.quantile(onecut_array, 0.95))
    recommended = min(target, cap)
    for key, expected in (
        ("target_weight", target),
        ("cap_weight", cap),
        ("recommended_onecut_weight", recommended),
    ):
        actual = _finite_float(report.get(key), f"onecut calibration {key}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"One-cut calibration {key} is inconsistent.")
    if report.get("source_provenance") != source_provenance:
        raise ValueError("One-cut calibration source differs from this run.")
    if canonical_sha256(report.get("runtime_provenance")) != canonical_sha256(
        runtime_provenance
    ):
        raise ValueError("One-cut calibration runtime differs from this run.")
    if canonical_sha256(report.get("execution_provenance")) != canonical_sha256(
        execution_provenance
    ):
        raise ValueError("One-cut calibration device differs from this run.")
    for path_key, current_path, digest_key, current_digest in (
        ("pkl", args.pkl, "pkl_sha256", pkl_digest),
        ("splits_json", args.splits_json, "splits_json_sha256", splits_digest),
    ):
        recorded = report.get(path_key)
        if not isinstance(recorded, str) or Path(recorded).resolve() != current_path.resolve():
            raise ValueError(f"One-cut calibration {path_key} path differs from this run.")
        if report.get(digest_key) != current_digest:
            raise ValueError(f"One-cut calibration {path_key} contents differ from this run.")
    checkpoint_path = Path(str(report.get("checkpoint", ""))).expanduser()
    checkpoint_config_path = Path(str(report.get("checkpoint_config", ""))).expanduser()
    if not checkpoint_path.is_file() or not checkpoint_config_path.is_file():
        raise ValueError("One-cut calibration source checkpoint/config is unavailable.")
    with snapshot_file(checkpoint_path) as checkpoint_snapshot, snapshot_file(
        checkpoint_config_path
    ) as config_snapshot:
        if checkpoint_snapshot.sha256 != report.get("checkpoint_sha256"):
            raise ValueError("One-cut calibration checkpoint changed.")
        if config_snapshot.sha256 != report.get("checkpoint_config_sha256"):
            raise ValueError("One-cut calibration checkpoint config changed.")
        source_config = load_bands_calibration_report(config_snapshot)
        source_run = source_config.get("run")
        if (
            not isinstance(source_run, dict)
            or source_run.get("constraint_set") != "none"
            or source_run.get("supervised_loss", "dice") != "dice"
            or source_run.get("ce_weight", 0.0) != 0.0
            or source_run.get("epochs") != 5
            or source_run.get("dataset") != args.dataset
            or source_run.get("fold") != args.fold
            or source_run.get("amp") is not bool(args.amp)
            or source_run.get("source_sha256") != source_provenance.get("sha256")
            or source_run.get("pkl_sha256") != pkl_digest
            or source_run.get("splits_json_sha256") != splits_digest
        ):
            raise ValueError("One-cut calibration source run is inconsistent.")
        checkpoint = torch.load(checkpoint_snapshot.path, map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("epoch") != 5
            or report.get("checkpoint_epoch") != 5
            or json.loads(json.dumps(checkpoint.get("run"))) != source_run
        ):
            raise ValueError("One-cut calibration source checkpoint is inconsistent.")
    if not _valid_sha256(report_snapshot.sha256):
        raise ValueError("One-cut calibration report digest is malformed.")


def validate_ap_plane_calibration_report(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    report_snapshot: FileSnapshot,
    pkl_digest: str,
    splits_digest: str,
    source_provenance: dict[str, Any],
    runtime_provenance: dict[str, Any],
    execution_provenance: dict[str, Any],
    splits_path: Path | None = None,
) -> None:
    """Bind an A/P-plane run to its training-only gradient calibration."""

    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ValueError("A/P-plane calibration report is not complete.")
    expected_constraint_set = args.constraint_set
    if report.get("constraint_set") != expected_constraint_set:
        raise ValueError("A/P-plane calibration report has the wrong constraint set.")
    if report.get("dataset") != args.dataset or report.get("fold") != args.fold:
        raise ValueError("A/P-plane calibration dataset/fold does not match this run.")
    augmentation = getattr(args, "training_augmentation", "none")
    if report.get("training_augmentation", "none") != augmentation:
        raise ValueError("A/P-plane calibration training_augmentation does not match this run.")
    if augmentation == "mild_v1" and (
        report.get("augmentation_policy") != MILD_V1
        or report.get("augmentation_seed") != report.get("seed", -1) + 1
    ):
        raise ValueError("A/P-plane calibration augmentation policy/RNG does not match.")
    expected_geometry = {
        "axis": int(args.ap_plane_axis),
        "anterior_side": str(args.ap_plane_anterior_side),
        "margin": float(args.ap_plane_margin),
        "require_both": True,
    }
    if expected_constraint_set == "ap_plane_location":
        expected_geometry["objective"] = "location"
    elif expected_constraint_set == "ap_plane_ce_control":
        expected_geometry["objective"] = "conditional_ce"
    if report.get("ap_plane") != expected_geometry:
        raise ValueError("A/P-plane calibration geometry does not match this run.")
    if report.get("amp") is not bool(args.amp):
        raise ValueError("A/P-plane calibration AMP mode does not match this run.")
    weight = report.get("recommended_ap_plane_weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) <= 0
        or float(weight) != float(args.ap_plane_weight)
    ):
        raise ValueError("--ap-plane-weight must exactly match the calibration report.")
    if report.get("target_ratio") != 0.10 or report.get("safety_ratio") != 0.50:
        raise ValueError("A/P-plane calibration uses a noncanonical gradient policy.")

    cases = report.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("A/P-plane calibration must contain 1..32 case records.")
    case_names = [row.get("case_name") if isinstance(row, dict) else None for row in cases]
    if any(not isinstance(name, str) or not name for name in case_names):
        raise ValueError("A/P-plane calibration case identities are malformed.")
    splits = _load_splits_json(splits_path or args.splits_json)
    training_names = set(splits[args.fold]["train"])
    seed = _exact_int(report.get("seed"), "A/P-plane calibration seed")
    max_cases = _exact_int(
        report.get("max_cases"), "A/P-plane calibration max_cases"
    )
    expected_names = sorted(
        random.Random(seed).sample(
            sorted(training_names), min(max_cases, len(training_names))
        )
    )
    if case_names != expected_names or report.get("training_case_ids") != expected_names:
        raise ValueError("A/P-plane calibration is not the deterministic training sample.")

    dice_rms: list[float] = []
    plane_rms: list[float] = []
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("A/P-plane calibration contains a malformed case record.")
        valid = row.get("valid") is True and row.get("finite_positive_gradients") is True
        if valid:
            dice_rms.append(
                _finite_float(row.get("dice_gradient_rms"), "plane calibration Dice RMS")
            )
            plane_rms.append(
                _finite_float(row.get("plane_gradient_rms"), "plane calibration RMS")
            )
    if not dice_rms or len(dice_rms) != len(plane_rms):
        raise ValueError("A/P-plane calibration has no valid gradient pairs.")
    if (
        report.get("valid_case_count") != len(dice_rms)
        or report.get("skipped_case_count") != len(cases) - len(dice_rms)
        or report.get("dice_gradient_rms") != _calibration_summary(dice_rms)
        or report.get("plane_gradient_rms") != _calibration_summary(plane_rms)
    ):
        raise ValueError("A/P-plane calibration summaries disagree with case data.")
    dice_median = float(np.median(np.asarray(dice_rms, dtype=np.float64)))
    plane_array = np.asarray(plane_rms, dtype=np.float64)
    target = 0.10 * dice_median / float(np.median(plane_array))
    cap = 0.50 * dice_median / float(np.quantile(plane_array, 0.95))
    recommended = min(target, cap)
    for key, expected in (
        ("target_weight", target),
        ("cap_weight", cap),
        ("recommended_ap_plane_weight", recommended),
    ):
        actual = _finite_float(report.get(key), f"plane calibration {key}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"A/P-plane calibration {key} is inconsistent.")
    if report.get("source_provenance") != source_provenance:
        raise ValueError("A/P-plane calibration source differs from this run.")
    if canonical_sha256(report.get("runtime_provenance")) != canonical_sha256(
        runtime_provenance
    ):
        raise ValueError("A/P-plane calibration runtime differs from this run.")
    report_execution = report.get("execution_provenance")
    compatible_execution_fields = (
        "device_type",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "float32_matmul_precision",
        "cuda_compute_capability",
    )
    if not isinstance(report_execution, dict) or any(
        report_execution.get(field) != execution_provenance.get(field)
        for field in compatible_execution_fields
    ):
        raise ValueError(
            "A/P-plane calibration numerical backend differs from this run."
        )
    for path_key, current_path, digest_key, current_digest in (
        ("pkl", args.pkl, "pkl_sha256", pkl_digest),
        ("splits_json", args.splits_json, "splits_json_sha256", splits_digest),
    ):
        recorded = report.get(path_key)
        if not isinstance(recorded, str) or Path(recorded).resolve() != current_path.resolve():
            raise ValueError(f"A/P-plane calibration {path_key} differs from this run.")
        if report.get(digest_key) != current_digest:
            raise ValueError(
                f"A/P-plane calibration {path_key} contents differ from this run."
            )
    checkpoint_path = Path(str(report.get("checkpoint", ""))).expanduser()
    if not checkpoint_path.is_file():
        raise ValueError("A/P-plane calibration source checkpoint is unavailable.")
    with snapshot_file(checkpoint_path) as checkpoint_snapshot:
        if checkpoint_snapshot.sha256 != report.get("checkpoint_sha256"):
            raise ValueError("A/P-plane calibration checkpoint changed.")
        checkpoint = torch.load(checkpoint_snapshot.path, map_location="cpu", weights_only=True)
        source_run = checkpoint.get("run") if isinstance(checkpoint, dict) else None
        if not isinstance(source_run, dict):
            raise ValueError("A/P-plane calibration checkpoint lacks preprocessing provenance.")
        # Older reports omit preprocessing; recover it from the verified source
        # checkpoint, whose geometry the calibration command already checked.
        for field, expected in (
            ("spatial_size", list(args.spatial_size)),
            ("resize", bool(args.resize)),
        ):
            recorded = source_run.get(field)
            if field == "spatial_size" and isinstance(recorded, (list, tuple)):
                recorded = list(recorded)
            if recorded != expected or report.get(field, recorded) != recorded:
                raise ValueError(f"A/P-plane calibration {field} does not match this run.")
    if not _valid_sha256(report_snapshot.sha256):
        raise ValueError("A/P-plane calibration report digest is malformed.")


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Resume checkpoint {field} must be an exact integer.")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Resume checkpoint {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Resume checkpoint {field} must be finite.")
    return result


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    run_spec: RunSpec,
) -> None:
    """Reject checkpoints that do not belong to the exact resumed run."""

    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint payload is not a dictionary.")
    required = {
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "best_hard_dice",
        "best_epoch",
        "rng",
        "run",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Resume checkpoint is missing fields: {', '.join(missing)}")
    expected_run = json.loads(json.dumps(asdict(run_spec)))
    checkpoint_run = json.loads(json.dumps(checkpoint["run"]))
    if checkpoint_run != expected_run:
        raise ValueError("Resume checkpoint run provenance does not match config.json.")
    epoch = _exact_int(checkpoint["epoch"], "epoch")
    best_epoch = _exact_int(checkpoint["best_epoch"], "best_epoch")
    best_hard_dice = _finite_float(checkpoint["best_hard_dice"], "best_hard_dice")
    if epoch < 1 or epoch > run_spec.epochs:
        raise ValueError("Resume checkpoint epoch is outside the configured run.")
    if best_epoch < 0 or best_epoch > epoch:
        raise ValueError("Resume checkpoint best_epoch is inconsistent with epoch.")
    if not 0.0 <= best_hard_dice <= 1.0:
        raise ValueError("Resume checkpoint best_hard_dice must be finite in [0, 1].")
    for field in ("model", "optimizer", "scaler", "rng"):
        if not isinstance(checkpoint[field], dict):
            raise ValueError(f"Resume checkpoint {field} state is malformed.")
    scheduler_state = checkpoint["scheduler"]
    if run_spec.optimizer_mode == "adamw_0.01":
        if not isinstance(scheduler_state, dict) or not scheduler_state:
            raise ValueError("AdamW resume requires a saved StepLR scheduler state.")
        required_scheduler_fields = {
            "step_size",
            "gamma",
            "base_lrs",
            "last_epoch",
            "_step_count",
            "_last_lr",
        }
        if required_scheduler_fields - set(scheduler_state):
            raise ValueError("AdamW resume has an incomplete StepLR scheduler state.")
        scheduler_step_size = _exact_int(scheduler_state["step_size"], "scheduler.step_size")
        scheduler_epoch = _exact_int(scheduler_state["last_epoch"], "scheduler.last_epoch")
        scheduler_step_count = _exact_int(
            scheduler_state["_step_count"], "scheduler._step_count"
        )
        if scheduler_step_size != run_spec.step_size or scheduler_epoch != epoch:
            raise ValueError("AdamW scheduler chronology/configuration is inconsistent.")
        if scheduler_step_count != epoch + 1:
            raise ValueError("AdamW scheduler step count is inconsistent with epoch.")
        gamma = _finite_float(scheduler_state["gamma"], "scheduler.gamma")
        if gamma != run_spec.adamw_gamma:
            raise ValueError("AdamW scheduler gamma does not match the run specification.")
        base_lrs = scheduler_state["base_lrs"]
        last_lrs = scheduler_state["_last_lr"]
        optimizer_groups = checkpoint["optimizer"].get("param_groups")
        if (
            not isinstance(base_lrs, list)
            or not base_lrs
            or not isinstance(last_lrs, list)
            or not isinstance(optimizer_groups, list)
            or len(base_lrs) != len(last_lrs)
            or len(base_lrs) != len(optimizer_groups)
        ):
            raise ValueError("AdamW scheduler/optimizer parameter groups are malformed.")
        expected_lr = float(run_spec.learning_rate) * (
            float(run_spec.adamw_gamma) ** (epoch // run_spec.step_size)
        )
        for index, (base_lr, last_lr, optimizer_group) in enumerate(
            zip(base_lrs, last_lrs, optimizer_groups)
        ):
            if not isinstance(optimizer_group, dict):
                raise ValueError("AdamW optimizer parameter group is malformed.")
            checked_base = _finite_float(base_lr, f"scheduler.base_lrs[{index}]")
            checked_last = _finite_float(last_lr, f"scheduler._last_lr[{index}]")
            optimizer_lr = _finite_float(
                optimizer_group.get("lr"), f"optimizer.param_groups[{index}].lr"
            )
            if checked_base != float(run_spec.learning_rate):
                raise ValueError("AdamW scheduler base learning rate changed.")
            if not math.isclose(checked_last, expected_lr, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError("AdamW scheduler learning-rate chronology is inconsistent.")
            if checked_last != optimizer_lr:
                raise ValueError("AdamW optimizer and scheduler learning rates disagree.")
    elif scheduler_state is not None:
        raise ValueError("nnU-Net resume expects no serialized scheduler state.")


def reconcile_best_checkpoint(
    output_dir: Path,
    latest_checkpoint: dict[str, Any],
    run_spec: RunSpec,
) -> None:
    """Repair a best checkpoint interrupted after the durable latest save."""

    latest_epoch = int(latest_checkpoint["epoch"])
    expected_best_epoch = int(latest_checkpoint["best_epoch"])
    best_path = output_dir / "checkpoint_best.pt"
    if expected_best_epoch == latest_epoch:
        save_checkpoint_payload(best_path, latest_checkpoint)
        return
    if not best_path.is_file():
        raise FileNotFoundError("Resume run is missing checkpoint_best.pt.")
    with snapshot_file(best_path) as best_snapshot:
        best_checkpoint = torch.load(
            best_snapshot.path,
            map_location="cpu",
            weights_only=True,
        )
    validate_resume_checkpoint(best_checkpoint, run_spec)
    if (
        best_checkpoint["epoch"] != expected_best_epoch
        or best_checkpoint["best_epoch"] != expected_best_epoch
        or best_checkpoint["best_hard_dice"] != latest_checkpoint["best_hard_dice"]
    ):
        raise ValueError("checkpoint_best.pt does not match the latest best_epoch.")


def restore_resume_checkpoint(
    checkpoint: dict[str, Any],
    run_spec: RunSpec,
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    data_generator: torch.Generator,
    translation_generator: torch.Generator,
) -> None:
    """Restore every state successfully before repairing any durable artifact."""

    validate_resume_checkpoint(checkpoint, run_spec)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng"], data_generator, translation_generator)
    reconcile_best_checkpoint(output_dir, checkpoint, run_spec)


def reconcile_metrics_for_resume(
    path: Path,
    checkpoint_epoch: int,
    fieldnames: list[str],
) -> None:
    """Remove rows written after the last durable checkpoint before appending."""

    retained = validate_metrics_for_resume(path, checkpoint_epoch, fieldnames)
    save_csv(path, fieldnames, retained)


def validate_metrics_for_resume(
    path: Path,
    checkpoint_epoch: int,
    fieldnames: list[str],
) -> list[dict[str, str]]:
    """Validate durable metric history without changing any run artifact."""

    required_fields = {
        "epoch",
        "train_loss",
        "train_supervised_loss",
        "train_constraint_loss",
        "constraint_scale",
        "learning_rate",
        "val_dice_soft",
        "val_dice_hard",
    }
    if not path.is_file():
        raise FileNotFoundError("--resume requires metrics.csv.")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError("metrics.csv header does not match the current pipeline.")
        retained: list[dict[str, str]] = []
        seen_epochs: set[int] = set()
        for row in reader:
            if None in row:
                raise ValueError("metrics.csv contains columns outside its header.")
            try:
                epoch = int(row["epoch"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("metrics.csv contains an invalid epoch.") from error
            if epoch > checkpoint_epoch:
                continue
            expected_epoch = len(retained) + 1
            if epoch != expected_epoch:
                raise ValueError("metrics.csv epochs must be strictly ordered from 1.")
            if epoch in seen_epochs:
                raise ValueError(f"metrics.csv contains duplicate epoch {epoch}.")
            for field in fieldnames:
                value = row.get(field)
                if value in (None, ""):
                    if field in required_fields:
                        raise ValueError(
                            f"metrics.csv contains a blank required {field} value."
                        )
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"metrics.csv contains a non-numeric {field} value."
                    ) from error
                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"metrics.csv contains a nonfinite {field} value."
                    )
            seen_epochs.add(epoch)
            retained.append(row)
    expected_epochs = set(range(1, checkpoint_epoch + 1))
    if seen_epochs != expected_epochs:
        raise ValueError("metrics.csv does not contain every durable checkpoint epoch.")
    return retained


def _wandb_metric_row(row: dict[str, str]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for name, value in row.items():
        if value == "":
            continue
        result[name] = int(value) if name == "epoch" else float(value)
    return result


def backfill_wandb_history(
    wandb_run: Any,
    metrics_path: Path,
    checkpoint_epoch: int,
    fieldnames: list[str],
) -> None:
    """Replay all durable epochs; W&B drops already-present monotonic steps."""

    rows = validate_metrics_for_resume(metrics_path, checkpoint_epoch, fieldnames)
    for row in rows:
        epoch = int(row["epoch"])
        wandb_run.log(_wandb_metric_row(row), step=epoch)


def validate_optimizer_hyperparameters(
    *,
    step_size: int,
    learning_rate: float,
    weight_decay: float,
    adamw_gamma: float,
) -> None:
    """Reject optimizer values that would crash or poison a run after it starts."""

    if step_size <= 0:
        raise ValueError("--step-size must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative.")
    if not math.isfinite(adamw_gamma) or not 0 < adamw_gamma <= 1:
        raise ValueError("--adamw-gamma must be finite and in (0, 1].")


def validate_epoch_metrics(row: dict[str, Any]) -> None:
    """Stop before logging or checkpointing a numerically poisoned epoch."""

    for name, value in row.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise FloatingPointError(f"Epoch metric {name} is not finite.")


def _main(snapshot_stack: ExitStack) -> None:
    args = parse_args()
    if args.training_augmentation != "none" and (
        args.translation_augmentation
        or args.constraint_set not in {"none", "ap_plane_ce_control", "bands"}
        or args.batch_size != 1
    ):
        raise ValueError("mild_v1 requires constraint-set none, ap_plane_ce_control or bands, batch size 1, and no translation augmentation.")
    if not math.isfinite(args.drop_rate) or not 0 <= args.drop_rate < 1:
        raise ValueError("--drop-rate must be finite and in [0, 1).")
    stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        min_epochs=args.early_stopping_min_epochs,
    )
    if stopping.patience and stopping.min_epochs > args.epochs:
        raise ValueError("Early-stopping min epochs cannot exceed the maximum epochs.")
    loss_spec = supervised_loss_config(args.supervised_loss, args.ce_weight)
    constraint_scale_knots = parse_constraint_scale_knots(args.constraint_scale_knots)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.constraint_warmup_epochs < 0 or args.constraint_eval_every < 0:
        raise ValueError("Constraint warmup/evaluation intervals must be non-negative.")
    if constraint_scale_knots and args.constraint_warmup_epochs != 0:
        raise ValueError(
            "--constraint-scale-knots requires --constraint-warmup-epochs 0 "
            "so only one scheduling rule is active."
        )
    if args.equivariance_max_samples < 0:
        raise ValueError("--equivariance-max-samples must be non-negative.")
    if args.band_steps < 1:
        raise ValueError("--band-steps must be positive.")
    if args.telemetry:
        if args.constraint_set != "bands":
            raise ValueError("--telemetry currently requires --constraint-set bands.")
        if (
            args.telemetry_probe_cases < 1
            or args.telemetry_spatial_cases < 0
            or args.telemetry_spatial_cases > args.telemetry_probe_cases
        ):
            raise ValueError("Telemetry case counts are inconsistent.")
        probe_epochs = tuple(args.telemetry_probe_epochs)
        if (
            not probe_epochs
            or any(epoch < 1 or epoch > args.epochs for epoch in probe_epochs)
            or len(set(probe_epochs)) != len(probe_epochs)
            or tuple(sorted(probe_epochs)) != probe_epochs
            or args.epochs not in probe_epochs
        ):
            raise ValueError(
                "Telemetry probe epochs must be sorted, unique, within the run, and include the final epoch."
            )
    if args.resume and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive.")
    if not args.pkl.is_file() or not args.splits_json.is_file():
        raise FileNotFoundError("The dataset pickle and split JSON must both exist.")
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    if args.bands_calibration_json is not None and not args.bands_calibration_json.is_file():
        raise FileNotFoundError(args.bands_calibration_json)
    if args.onecut_calibration_json is not None and not args.onecut_calibration_json.is_file():
        raise FileNotFoundError(args.onecut_calibration_json)
    if args.ap_plane_calibration_json is not None and not args.ap_plane_calibration_json.is_file():
        raise FileNotFoundError(args.ap_plane_calibration_json)
    validate_optimizer_hyperparameters(
        step_size=args.step_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adamw_gamma=args.adamw_gamma,
    )

    source_provenance = collect_source_provenance()
    runtime_provenance = collect_runtime_provenance()
    pkl_snapshot = snapshot_stack.enter_context(snapshot_file(args.pkl))
    splits_snapshot = snapshot_stack.enter_context(snapshot_file(args.splits_json))
    initial_snapshot = (
        snapshot_stack.enter_context(snapshot_file(args.init_checkpoint))
        if args.init_checkpoint is not None
        else None
    )
    bands_calibration_snapshot = (
        snapshot_stack.enter_context(snapshot_file(args.bands_calibration_json))
        if args.bands_calibration_json is not None
        else None
    )
    onecut_calibration_snapshot = (
        snapshot_stack.enter_context(snapshot_file(args.onecut_calibration_json))
        if args.onecut_calibration_json is not None
        else None
    )
    ap_plane_calibration_snapshot = (
        snapshot_stack.enter_context(snapshot_file(args.ap_plane_calibration_json))
        if args.ap_plane_calibration_json is not None
        else None
    )
    pkl_digest = pkl_snapshot.sha256
    splits_digest = splits_snapshot.sha256

    seed_everything(args.seed)
    data_generator, translation_generator, translation_seed = (
        build_experiment_generators(args.seed)
    )
    device = resolve_device(args.device)
    execution_provenance = collect_execution_provenance(device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA.")
    config = resolve_constraint_config(args)
    calibration_report: dict[str, Any] | None = None
    if bands_calibration_snapshot is not None:
        calibration_report = load_bands_calibration_report(bands_calibration_snapshot)
        validate_bands_calibration_report(
            calibration_report,
            args,
            report_snapshot=bands_calibration_snapshot,
            pkl_digest=pkl_digest,
            splits_digest=splits_digest,
            source_provenance=source_provenance,
            runtime_provenance=runtime_provenance,
            execution_provenance=execution_provenance,
            splits_path=splits_snapshot.path,
        )
    elif onecut_calibration_snapshot is not None:
        calibration_report = load_bands_calibration_report(onecut_calibration_snapshot)
        validate_onecut_calibration_report(
            calibration_report,
            args,
            report_snapshot=onecut_calibration_snapshot,
            pkl_digest=pkl_digest,
            splits_digest=splits_digest,
            source_provenance=source_provenance,
            runtime_provenance=runtime_provenance,
            execution_provenance=execution_provenance,
            splits_path=splits_snapshot.path,
        )
    elif ap_plane_calibration_snapshot is not None:
        calibration_report = load_bands_calibration_report(
            ap_plane_calibration_snapshot
        )
        validate_ap_plane_calibration_report(
            calibration_report,
            args,
            report_snapshot=ap_plane_calibration_snapshot,
            pkl_digest=pkl_digest,
            splits_digest=splits_digest,
            source_provenance=source_provenance,
            runtime_provenance=runtime_provenance,
            execution_provenance=execution_provenance,
            splits_path=splits_snapshot.path,
        )
    # Constructing the objective also validates all constraint hyperparameters.
    objective = NewConstraintObjective(config).to(device)
    evaluation_objective = NewConstraintObjective(
        replace(config, equivariance_max_samples=None)
    ).to(device)

    run_spec = RunSpec(
        dataset=args.dataset,
        fold=args.fold,
        seed=args.seed,
        translation_seed=translation_seed,
        translation_augmentation=getattr(args, "translation_augmentation", False),
        epochs=args.epochs,
        batch_size=args.batch_size,
        spatial_size=tuple(args.spatial_size),
        resize=args.resize,
        optimizer_mode=args.optim_mode,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        adamw_gamma=args.adamw_gamma,
        constraint_set=(
            "equivariance" if args.constraint_set == "translation" else args.constraint_set
        ),
        constraint_config=asdict(config),
        constraint_warmup_epochs=args.constraint_warmup_epochs,
        constraint_scale_knots=constraint_scale_knots,
        constraint_eval_every=args.constraint_eval_every,
        amp=args.amp,
        initial_checkpoint=(
            str(initial_snapshot.original_path) if initial_snapshot is not None else None
        ),
        initial_checkpoint_sha256=(
            initial_snapshot.sha256 if initial_snapshot is not None else None
        ),
        pkl_sha256=pkl_digest,
        splits_json_sha256=splits_digest,
        source_sha256=source_provenance["sha256"],
        runtime_sha256=canonical_sha256(runtime_provenance),
        execution_sha256=canonical_sha256(execution_provenance),
        bands_calibration=(
            str(bands_calibration_snapshot.original_path)
            if bands_calibration_snapshot is not None
            else None
        ),
        bands_calibration_sha256=(
            bands_calibration_snapshot.sha256
            if bands_calibration_snapshot is not None
            else None
        ),
        onecut_calibration=(
            str(onecut_calibration_snapshot.original_path)
            if onecut_calibration_snapshot is not None
            else None
        ),
        onecut_calibration_sha256=(
            onecut_calibration_snapshot.sha256
            if onecut_calibration_snapshot is not None
            else None
        ),
        ap_plane_calibration=(
            str(ap_plane_calibration_snapshot.original_path)
            if ap_plane_calibration_snapshot is not None
            else None
        ),
        ap_plane_calibration_sha256=(
            ap_plane_calibration_snapshot.sha256
            if ap_plane_calibration_snapshot is not None
            else None
        ),
        telemetry=bool(args.telemetry),
        telemetry_probe_epochs=(
            tuple(args.telemetry_probe_epochs) if args.telemetry else ()
        ),
        telemetry_probe_cases=(args.telemetry_probe_cases if args.telemetry else 0),
        telemetry_spatial_cases=(args.telemetry_spatial_cases if args.telemetry else 0),
        supervised_loss=args.supervised_loss,
        ce_weight=float(loss_spec["ce_weight"]),
        calibration_diagnostics=bool(args.calibration_diagnostics),
        early_stopping_patience=stopping.patience,
        early_stopping_min_delta=stopping.min_delta,
        early_stopping_min_epochs=stopping.min_epochs,
        drop_rate=args.drop_rate,
        activation_checkpointing=args.activation_checkpointing,
        plain_tensors=args.plain_tensors,
        training_augmentation=args.training_augmentation,
    )

    output_dir = args.output_dir.resolve()
    previous: dict[str, Any] | None = None
    recover_incomplete_run = False
    if args.resume:
        if not (output_dir / "checkpoint_latest.pt").is_file():
            raise FileNotFoundError("--resume requires checkpoint_latest.pt.")
        if not (output_dir / "config.json").is_file():
            raise FileNotFoundError("--resume requires config.json.")
        previous = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        validate_input_file_provenance(
            previous,
            args.pkl,
            args.splits_json,
            pkl_digest=pkl_digest,
            splits_digest=splits_digest,
        )
        run_spec = resolve_resume_run_spec(run_spec, previous["run"])
    elif output_dir.exists():
        config_path = output_dir / "config.json"
        if (
            not config_path.is_file()
            or (output_dir / "checkpoint_latest.pt").exists()
            or (output_dir / "checkpoint_best.pt").exists()
        ):
            raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        validate_input_file_provenance(
            previous,
            args.pkl,
            args.splits_json,
            pkl_digest=pkl_digest,
            splits_digest=splits_digest,
        )
        if previous.get("run") != json.loads(json.dumps(asdict(run_spec))):
            raise ValueError("Incomplete run configuration does not match this restart.")
        recover_incomplete_run = True

    if previous is not None:
        wandb_config = previous.get("wandb")
        if not isinstance(wandb_config, dict):
            raise ValueError("Existing run lacks persisted W&B provenance.")
        if bool(wandb_config.get("enabled")) != bool(args.wandb):
            raise ValueError("W&B enablement must match the existing run.")
        if args.wandb and (
            wandb_config.get("project") != args.wandb_project
            or wandb_config.get("entity") != args.wandb_entity
            or not isinstance(wandb_config.get("run_id"), str)
            or not wandb_config["run_id"]
        ):
            raise ValueError("Existing run has incompatible W&B provenance.")
    else:
        wandb_config = {
            "enabled": bool(args.wandb),
            "project": args.wandb_project if args.wandb else None,
            "entity": args.wandb_entity if args.wandb else None,
            "run_id": uuid.uuid4().hex if args.wandb else None,
        }

    train_loader, validation_loader, num_classes, train_count, validation_count = (
        build_data(
            args,
            data_generator,
            pkl_path=pkl_snapshot.path,
            splits_path=splits_snapshot.path,
        )
    )
    pkl_snapshot.cleanup()
    splits_snapshot.cleanup()

    model = build_swinunetr(
        run_spec.spatial_size, num_classes, device, drop_rate=run_spec.drop_rate,
        activation_checkpointing=run_spec.activation_checkpointing,
    )
    if initial_snapshot is not None:
        load_initial_weights(model, initial_snapshot.path)
        initial_snapshot.cleanup()
    if bands_calibration_snapshot is not None:
        bands_calibration_snapshot.cleanup()
    if onecut_calibration_snapshot is not None:
        onecut_calibration_snapshot.cleanup()
    if ap_plane_calibration_snapshot is not None:
        ap_plane_calibration_snapshot.cleanup()
    optimizer, scheduler = build_optimizer_and_scheduler(
        args.optim_mode,
        model,
        args.epochs,
        adamw_gamma=args.adamw_gamma,
        step_size=args.step_size if args.optim_mode == "adamw_0.01" else None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    supervised_loss_function = build_supervised_loss(args.supervised_loss, args.ce_weight)
    fields = metric_fieldnames(args.calibration_diagnostics, num_classes)

    config_payload = {
        "run": asdict(run_spec),
        "supervised_loss": loss_spec,
        "training_augmentation": dict(MILD_V1) if args.training_augmentation == "mild_v1" else None,
        "calibration_diagnostics": (
            calibration_diagnostics_config() if args.calibration_diagnostics else None
        ),
        "train_samples": train_count,
        "validation_samples": validation_count,
        "pkl": str(args.pkl.resolve()),
        "pkl_sha256": pkl_digest,
        "splits_json": str(args.splits_json.resolve()),
        "splits_json_sha256": splits_digest,
        "source_provenance": source_provenance,
        "runtime_provenance": runtime_provenance,
        "execution_provenance": execution_provenance,
        "bands_calibration": (
            str(args.bands_calibration_json.resolve())
            if args.bands_calibration_json is not None
            else None
        ),
        "bands_calibration_sha256": run_spec.bands_calibration_sha256,
        "bands_calibration_checkpoint_sha256": (
            calibration_report.get("checkpoint_sha256")
            if calibration_report is not None and args.bands_calibration_json is not None
            else None
        ),
        "onecut_calibration": (
            str(args.onecut_calibration_json.resolve())
            if args.onecut_calibration_json is not None
            else None
        ),
        "onecut_calibration_sha256": run_spec.onecut_calibration_sha256,
        "onecut_calibration_checkpoint_sha256": (
            calibration_report.get("checkpoint_sha256")
            if calibration_report is not None and args.onecut_calibration_json is not None
            else None
        ),
        "ap_plane_calibration": (
            str(args.ap_plane_calibration_json.resolve())
            if args.ap_plane_calibration_json is not None
            else None
        ),
        "ap_plane_calibration_sha256": run_spec.ap_plane_calibration_sha256,
        "ap_plane_calibration_checkpoint_sha256": (
            calibration_report.get("checkpoint_sha256")
            if calibration_report is not None
            and args.ap_plane_calibration_json is not None
            else None
        ),
        "wandb": wandb_config,
        "device": str(device),
    }
    if not args.resume:
        validate_source_provenance_unchanged(source_provenance)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        canonical_config_payload = json.loads(json.dumps(config_payload))
        if recover_incomplete_run and previous != canonical_config_payload:
            raise ValueError("Incomplete config.json changed before restart.")
        save_json(output_dir / "config.json", config_payload)

    start_epoch = 1
    checkpoint_epoch = 0
    best_hard_dice = -float("inf")
    best_epoch = 0
    if args.resume:
        with snapshot_file(output_dir / "checkpoint_latest.pt") as checkpoint_snapshot:
            checkpoint = torch.load(
                checkpoint_snapshot.path,
                # RNG states are CPU byte tensors; loading the full payload onto
                # CUDA makes torch.set_rng_state fail during resume.
                map_location="cpu",
                weights_only=True,
            )
        checkpoint_epoch = _exact_int(checkpoint.get("epoch"), "epoch")
        retained_metrics = validate_metrics_for_resume(
            output_dir / "metrics.csv",
            checkpoint_epoch,
            fields,
        )
        restore_resume_checkpoint(
            checkpoint,
            run_spec,
            output_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            data_generator,
            translation_generator,
        )
        start_epoch = checkpoint_epoch + 1
        best_hard_dice = float(checkpoint["best_hard_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        save_csv(output_dir / "metrics.csv", fields, retained_metrics)
        for metric_row in retained_metrics:
            stopping.update(int(metric_row["epoch"]), float(metric_row["val_dice_hard"]))

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or output_dir.name,
            id=wandb_config["run_id"],
            config=asdict(run_spec),
            resume=(
                "must"
                if args.resume
                else "allow" if recover_incomplete_run else "never"
            ),
        )
        if args.resume:
            backfill_wandb_history(
                wandb_run,
                output_dir / "metrics.csv",
                checkpoint_epoch,
                fields,
            )

    telemetry_epoch_path = output_dir / "telemetry_epoch.csv"
    telemetry_gradient_path = output_dir / "telemetry_gradients.csv"
    telemetry_epoch_rows: list[dict[str, Any]] = []
    telemetry_gradient_rows: list[dict[str, Any]] = []
    if args.telemetry and args.resume:
        telemetry_epoch_rows = telemetry_rows_for_resume(
            telemetry_epoch_path,
            TELEMETRY_EPOCH_FIELDS,
            checkpoint_epoch,
            one_row_per_epoch=True,
        )
        telemetry_gradient_rows = telemetry_rows_for_resume(
            telemetry_gradient_path,
            TELEMETRY_GRADIENT_FIELDS,
            checkpoint_epoch,
            one_row_per_epoch=False,
        )
        save_csv(telemetry_epoch_path, TELEMETRY_EPOCH_FIELDS, telemetry_epoch_rows)
        save_csv(
            telemetry_gradient_path,
            TELEMETRY_GRADIENT_FIELDS,
            telemetry_gradient_rows,
        )

    metrics_path = output_dir / "metrics.csv"
    metrics_mode = "a" if args.resume else "w"
    with metrics_path.open(metrics_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not args.resume:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            if stopping.should_stop:
                break
            epoch_started = time.perf_counter()
            model.train()
            loss_totals = {"total": 0.0, "supervised": 0.0, "constraint": 0.0}
            constraint_totals: dict[str, dict[str, Any]] = {}
            augmentation_totals: dict[str, int] = {}
            constraint_scale = constraint_scale_for_epoch(
                epoch,
                args.constraint_warmup_epochs,
                run_spec.constraint_scale_knots,
            )

            for batch_index, batch in enumerate(train_loader):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                if run_spec.training_augmentation == "mild_v1":
                    images, labels, augmentation_stats = augment_mild(
                        images, labels, generator=translation_generator,
                    )
                    for key, value in augmentation_stats.items():
                        augmentation_totals[key] = augmentation_totals.get(key, 0) + value
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(images)
                    shifted_logits = None
                    selected_shift = None
                    identity_supervised_loss = supervised_loss_function(logits, labels)
                    if run_spec.translation_augmentation:
                        shifted_images, shifted_labels, selected_shift = augment_translation(
                            images, labels, generator=translation_generator,
                        )
                        shifted_logits = model(shifted_images)
                        shifted_supervised_loss = supervised_loss_function(
                            shifted_logits, shifted_labels
                        )
                        supervised_loss = 0.5 * (
                            identity_supervised_loss + shifted_supervised_loss
                        )
                    else:
                        supervised_loss = identity_supervised_loss
                    constraint_output = objective(
                        model,
                        images,
                        logits,
                        labels,
                        shift=(
                            selected_shift
                            if objective.config.equivariance_weight > 0
                            else None
                        ),
                        generator=translation_generator,
                        transformed_logits=(
                            shifted_logits
                            if objective.config.equivariance_weight > 0
                            else None
                        ),
                    )
                    constraint_loss = constraint_output["loss"]
                    loss = supervised_loss + constraint_scale * constraint_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_totals["total"] += float(loss.detach().cpu())
                loss_totals["supervised"] += float(supervised_loss.detach().cpu())
                loss_totals["constraint"] += float(constraint_loss.detach().cpu())
                update_constraint_totals(
                    constraint_totals,
                    constraint_output["results"],
                )

            training_seconds = time.perf_counter() - epoch_started
            validation_started = time.perf_counter()
            batches = max(1, len(train_loader))
            learning_rate = optimizer.param_groups[0]["lr"]
            should_evaluate_constraints = (
                args.constraint_eval_every > 0
                and (
                    epoch % args.constraint_eval_every == 0
                    or epoch == 1
                    or epoch == args.epochs
                )
            )
            telemetry_accumulator = None
            if args.telemetry:
                inner_gamma = (
                    config.bands_focal_gamma
                    if config.bands_inner_focal_gamma is None
                    else config.bands_inner_focal_gamma
                )
                outer_gamma = (
                    config.bands_focal_gamma
                    if config.bands_outer_focal_gamma is None
                    else config.bands_outer_focal_gamma
                )
                telemetry_accumulator = BoundaryTelemetryAccumulator(
                    foreground_class_ids=config.foreground_class_ids,
                    complement_class_ids=config.complement_class_ids,
                    band_steps=config.band_steps,
                    inner_gamma=inner_gamma,
                    outer_gamma=outer_gamma,
                    loss_type=config.bands_loss_type,
                    tversky_false_positive_weight=(
                        config.tversky_false_positive_weight
                    ),
                    tversky_false_negative_weight=(
                        config.tversky_false_negative_weight
                    ),
                    num_classes=num_classes,
                )
            validation_metrics = evaluate_validation_metrics(
                model,
                validation_loader,
                evaluation_objective,
                device,
                num_classes=num_classes,
                amp=args.amp,
                evaluate_constraints=should_evaluate_constraints,
                all_translation_shifts=False,
                telemetry=telemetry_accumulator,
                calibration_diagnostics=args.calibration_diagnostics,
            )
            telemetry_epoch_row: dict[str, Any] | None = None
            if telemetry_accumulator is not None:
                telemetry_epoch_row = telemetry_accumulator.finalize(epoch)
                validate_epoch_metrics(telemetry_epoch_row)
                telemetry_epoch_rows.append(telemetry_epoch_row)
                save_csv(
                    telemetry_epoch_path,
                    TELEMETRY_EPOCH_FIELDS,
                    telemetry_epoch_rows,
                )
            gradient_rows_for_epoch: list[dict[str, Any]] = []
            if args.telemetry and epoch in args.telemetry_probe_epochs:
                gradient_rows_for_epoch, spatial_payloads = probe_component_gradients(
                    model,
                    validation_loader,
                    evaluation_objective,
                    supervised_loss_function,
                    device,
                    epoch=epoch,
                    constraint_scale=constraint_scale,
                    max_cases=args.telemetry_probe_cases,
                    spatial_cases=args.telemetry_spatial_cases,
                )
                for gradient_row in gradient_rows_for_epoch:
                    validate_epoch_metrics(gradient_row)
                telemetry_gradient_rows.extend(gradient_rows_for_epoch)
                save_csv(
                    telemetry_gradient_path,
                    TELEMETRY_GRADIENT_FIELDS,
                    telemetry_gradient_rows,
                )
                telemetry_dir = output_dir / "telemetry"
                for case_index, payload in enumerate(spatial_payloads):
                    save_torch_payload(
                        telemetry_dir
                        / f"focus_epoch_{epoch:03d}_case_{case_index:02d}.pt",
                        payload,
                    )
            validation_seconds = time.perf_counter() - validation_started
            soft_dice = validation_metrics["val_dice_soft"]
            hard_dice = validation_metrics["val_dice_hard"]
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": loss_totals["total"] / batches,
                "train_supervised_loss": loss_totals["supervised"] / batches,
                "train_constraint_loss": loss_totals["constraint"] / batches,
                "constraint_scale": constraint_scale,
                "learning_rate": learning_rate,
                "val_dice_soft": soft_dice,
                "val_dice_hard": hard_dice,
                **averaged_constraint_metrics(constraint_totals, "train"),
                **{
                    key: value
                    for key, value in validation_metrics.items()
                    if key not in {"val_dice_soft", "val_dice_hard"}
                },
            }
            validate_epoch_metrics(row)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            printable = {key: value for key, value in row.items() if value != ""}
            print(json.dumps(printable), flush=True)

            if args.optim_mode == "nnunetv2":
                scheduler.step(epoch)
            else:
                scheduler.step()
            improved = hard_dice > best_hard_dice
            if improved:
                best_hard_dice = hard_dice
                best_epoch = epoch
            checkpoint_started = time.perf_counter()
            validate_source_provenance_unchanged(source_provenance)
            save_checkpoint(
                output_dir / "checkpoint_latest.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_hard_dice=best_hard_dice,
                best_epoch=best_epoch,
                data_generator=data_generator,
                translation_generator=translation_generator,
                run_spec=run_spec,
            )
            if improved:
                save_checkpoint(
                    output_dir / "checkpoint_best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_hard_dice=best_hard_dice,
                    best_epoch=best_epoch,
                    data_generator=data_generator,
                    translation_generator=translation_generator,
                    run_spec=run_spec,
                )
            if augmentation_totals:
                print(json.dumps({"augmentation_epoch": epoch, **augmentation_totals}), flush=True)
            print(json.dumps({"timing_epoch": epoch, "training_seconds": training_seconds,
                              "validation_seconds": validation_seconds,
                              "checkpoint_seconds": time.perf_counter() - checkpoint_started,
                              "epoch_seconds": time.perf_counter() - epoch_started}), flush=True)
            if wandb_run is not None:
                wandb_run.log(printable, step=epoch)
                if telemetry_epoch_row is not None:
                    wandb_run.log(
                        {
                            f"telemetry/{name}": value
                            for name, value in telemetry_epoch_row.items()
                            if name != "epoch"
                        },
                        step=epoch,
                    )
                all_gradient_row = next(
                    (
                        gradient_row
                        for gradient_row in gradient_rows_for_epoch
                        if gradient_row["parameter_group"] == "all"
                    ),
                    None,
                )
                if all_gradient_row is not None:
                    wandb_run.log(
                        {
                            f"telemetry_gradient/{name}": value
                            for name, value in all_gradient_row.items()
                            if name not in {"epoch", "parameter_group"}
                        },
                        step=epoch,
                    )

            stopping.update(epoch, hard_dice)
            if stopping.should_stop:
                print(json.dumps({"status": "early_stopping", **stopping.summary()}), flush=True)
                break

    selected_epoch = stopping.epoch
    if stopping.patience:
        # Latest remains a complete resume checkpoint at the stopping epoch.
        # The exported model and final evaluation use the best raw Dice epoch.
        selected = torch.load(output_dir / "checkpoint_best.pt", map_location="cpu", weights_only=True)
        validate_resume_checkpoint(selected, run_spec)
        if int(selected["epoch"]) != best_epoch:
            raise ValueError("Best checkpoint epoch disagrees with the training history.")
        model.load_state_dict(selected["model"])
        selected_epoch = best_epoch
        del selected

    model_dir = output_dir / f"{args.dataset}_fold{args.fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"
    save_torch_payload(model_path, model.state_dict())
    final_constraint_details: list[dict[str, Any]] = []
    final_validation_metrics = evaluate_validation_metrics(
        model,
        validation_loader,
        evaluation_objective,
        device,
        num_classes=num_classes,
        amp=args.amp,
        evaluate_constraints=True,
        all_translation_shifts=True,
        detail_rows=final_constraint_details,
        calibration_diagnostics=args.calibration_diagnostics,
    )
    final_metrics = {
        "epoch": selected_epoch,
        "checkpoint": "best" if stopping.patience else "final",
        "best_epoch": best_epoch,
        "best_val_dice_hard": best_hard_dice,
        **final_validation_metrics,
    }
    if stopping.patience:
        final_metrics["early_stopping"] = stopping.summary()
    validate_source_provenance_unchanged(source_provenance)
    final_metrics_path = output_dir / "final_metrics.json"
    save_json(final_metrics_path, final_metrics)
    detail_path = output_dir / "validation_constraint_details.csv"
    detail_fieldnames = [
        "constraint_name",
        "case_name",
        "inner_component_label",
        "outer_component_label",
        "shift_dx",
        "shift_dy",
        "shift_dz",
        "class_id",
        "optimization_truth",
        "legacy_linear_value",
        "class_confidence_adherent",
        "case_direction_truth",
        "case_direction_confidence_weighted_agreement",
        "case_direction_confidence_adherent",
        "band_valid",
        "band_loss",
        "inner_loss",
        "outer_loss",
        "inner_voxels",
        "outer_voxels",
        "onecut_valid",
        "onecut_loss",
        "onecut_truth",
        "onecut_allowed_cut_mass",
        "onecut_ray_count",
        "edge_touching",
        "raw_loss",
        "axis_0_loss",
        "axis_1_loss",
        "axis_2_loss",
        "foreground_loss",
        "anterior_loss",
        "posterior_loss",
        "teacher_views",
        "ap_cut_abs_error",
        "ap_gt_cut_probability",
        "ap_plane_valid",
        "ap_plane_loss",
        "ap_plane_selected_cut",
        "ap_plane_candidate_count",
        "ap_plane_target_cut",
        "ap_plane_cut_abs_error",
        "ap_plane_gt_disagreement_fraction",
        "ap_plane_target_cost",
        "ap_plane_target_cost_gap",
        "ap_plane_target_cost_gap_fraction",
        "ap_plane_raw_selected_cut",
        "ap_plane_raw_cut_abs_error",
        "ap_plane_raw_exact_cut",
        "ap_plane_raw_within_one_cut",
        "ap_plane_raw_disagreement_fraction",
        "ap_plane_raw_has_both_ap_classes",
        "ap_plane_raw_cut_valid",
        "foreground_union_dice",
        "ap_swap_voxels",
        "ap_swap_fraction",
        "displaced_slab_swap_voxels",
        "displaced_slab_swap_fraction",
    ]
    save_csv(detail_path, detail_fieldnames, final_constraint_details)
    completion_artifacts = {
        str(model_path.relative_to(output_dir)): file_sha256(model_path),
        final_metrics_path.name: file_sha256(final_metrics_path),
        detail_path.name: file_sha256(detail_path),
    }
    if args.telemetry:
        completion_artifacts[telemetry_epoch_path.name] = file_sha256(
            telemetry_epoch_path
        )
        completion_artifacts[telemetry_gradient_path.name] = file_sha256(
            telemetry_gradient_path
        )
        for focus_path in sorted((output_dir / "telemetry").glob("*.pt")):
            completion_artifacts[str(focus_path.relative_to(output_dir))] = file_sha256(
                focus_path
            )
    save_json(
        output_dir / "completion_manifest.json",
        {
            "status": "complete",
            "epoch": stopping.epoch,
            "selected_epoch": selected_epoch,
            "run": asdict(run_spec),
            "artifacts": completion_artifacts,
        },
    )
    if wandb_run is not None:
        wandb_run.log(
            {f"final/{name}": value for name, value in final_metrics.items()},
            step=stopping.epoch + 1,
        )
        artifact = wandb.Artifact(
            name=f"swinunetr-new-constraints-{args.constraint_set}-{args.dataset}-fold{args.fold}",
            type="model",
        )
        artifact.add_file(str(model_path), name="model.pt")
        artifact.add_file(str(output_dir / "config.json"), name="config.json")
        artifact.add_file(
            str(output_dir / "final_metrics.json"),
            name="final_metrics.json",
        )
        artifact.add_file(
            str(detail_path),
            name="validation_constraint_details.csv",
        )
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(
        json.dumps(
            {"status": "complete", "output_dir": str(output_dir), **final_metrics}
        ),
        flush=True,
    )


def main() -> None:
    with ExitStack() as snapshot_stack:
        _main(snapshot_stack)


if __name__ == "__main__":
    main()
