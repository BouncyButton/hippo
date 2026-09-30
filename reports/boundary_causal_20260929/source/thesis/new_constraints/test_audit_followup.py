"""Analytical gradients, repair accounting, and safe frozen-audit reports."""

import json

import numpy as np
import pytest
import torch

from thesis.new_constraints import audit_followup as audit
from thesis.new_constraints.teacher import TranslationTeacherKLLoss


def test_gradient_statistics_matches_analytical_teacher_gradient():
    logits = torch.tensor([0.8, -0.4, 0.2]).reshape(1, 3, 1, 1, 1).requires_grad_()
    target = torch.tensor([0.1, 0.7, 0.2]).reshape_as(logits)
    constraint = TranslationTeacherKLLoss()
    teacher = constraint.build_from_aligned([target], [torch.ones(1, 1, 1, 1, 1)])
    gradient, = torch.autograd.grad(constraint.loss_from_teacher(logits, teacher).loss, logits)
    expected = logits.detach().softmax(dim=1) - target
    assert torch.allclose(gradient, expected)
    stats = audit.gradient_statistics(expected * 2, gradient)
    assert stats["auxiliary_over_supervised_rms"] == pytest.approx(0.5)
    assert stats["cosine"] == pytest.approx(1.0)
    assert stats["auxiliary_rms"] == pytest.approx(float(expected.double().square().mean().sqrt()))


def test_normalized_repair_fixes_an_ap_swap_and_preserves_union():
    labels = torch.tensor([1, 2]).reshape(2, 1, 1)
    logits = torch.tensor([[-4.0, -4.0], [1.0, 1.0], [0.5, 0.0]]).reshape(3, 2, 1, 1)
    gradient = torch.zeros_like(logits)
    gradient[1, 1] = 1
    gradient[2, 1] = -1
    before = audit.segmentation_metrics(logits, labels)
    after_logits = audit.normalized_logit_step(logits, gradient, 1.0)
    after = audit.segmentation_metrics(after_logits, labels)
    assert before["swaps"] == 1
    assert after["swaps"] == 0
    assert before["union_dice"] == after["union_dice"] == 1
    assert after["macro_dice"] == 1
    assert after["total_errors"] == 0
    assert torch.max(torch.abs(after_logits - logits)) == 1


def test_normalized_repair_preserves_tiny_nonzero_gradient_direction():
    logits = torch.zeros(3, 1, 1, 1)
    gradient = torch.full_like(logits, torch.finfo(torch.float32).tiny)
    assert torch.allclose(audit.normalized_logit_step(logits, gradient, 0.25), torch.full_like(logits, -0.25))
    assert torch.equal(audit.normalized_logit_step(logits, gradient * 0, 0.25), logits)


def test_diagnostic_weight_respects_both_median_and_p95_limits():
    inputs = [{"auxiliary_over_supervised_rms": ratio} for ratio in (0.0, 1.0, 1.0, 1.0, 100.0, None)]
    result = audit.diagnostic_weight(inputs)
    assert result["valid_cases"] == 4
    assert result["value"] * result["p95_auxiliary_over_supervised_rms"] <= 0.5 + 1e-12
    assert result["value"] * result["median_auxiliary_over_supervised_rms"] <= 0.1 + 1e-12
    assert result["calibration_valid"] is False


def _write_case(path, invalid_ap=False):
    labels = np.ones((4, 3, 3), dtype=np.int64)
    labels[2:] = 2
    if invalid_ap:
        labels[1, 0, 0] = 2
    logits = np.zeros((3, 4, 3, 3), dtype=np.float32)
    logits[0] = -3
    logits[1] = 0.7
    logits[2] = 0.2
    teacher_logits = np.zeros((2, 3, 4, 3, 3), dtype=np.float32)
    teacher_logits[:, 0] = -3
    teacher_logits[:, 1] = 0.1
    teacher_logits[:, 2] = 1
    np.savez(path, logits=logits, labels=labels, teacher_logits=teacher_logits, shifts=np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.int64))


def _args(tmp_path, constraint="teacher"):
    arguments = [
        "--input-dir", str(tmp_path), "--output", str(tmp_path / "report.json"),
        "--purpose", "diagnostic", "--constraint", constraint,
        "--supervised-loss", "dice_ce", "--ce-weight", "1",
    ]
    if constraint == "ap_cut":
        arguments += ["--ap-axis", "0", "--ap-anterior-side", "low"]
    return audit.build_parser().parse_args(arguments)


def test_teacher_report_records_provenance_and_repairs_and_refuses_overwrite(tmp_path, monkeypatch):
    # Fix the source snapshot so other agents' concurrent source edits do not
    # perturb this numerical/reporting test.
    monkeypatch.setattr(audit, "source_digest", lambda: "test-source")
    _write_case(tmp_path / "case_01.npz")
    args = _args(tmp_path)
    report = audit.run_audit(args)
    loaded = json.loads((tmp_path / "report.json").read_text(), parse_constant=lambda value: pytest.fail(value))
    assert report == loaded
    assert report["valid_cases"] == 1
    assert report["source_sha256"] == "test-source"
    assert report["cases"][0]["sha256"] == audit.file_sha256(tmp_path / "case_01.npz")
    assert report["training_calibration_valid"] is False
    assert report["quality_claim_supported"] is False
    assert report["cases"][0]["teacher_same_support"]["teacher"]["valid_voxels"] == 36
    assert set(report["cases"][0]["repairs"]["auxiliary"]) == {"0.25", "1.0", "4.0"}
    before = (tmp_path / "report.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit.run_audit(args)
    assert (tmp_path / "report.json").read_bytes() == before


def test_ap_invalid_geometry_is_reported_and_excluded_from_weight(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "source_digest", lambda: "test-source")
    _write_case(tmp_path / "case_01.npz")
    _write_case(tmp_path / "case_02.npz", invalid_ap=True)
    report = audit.run_audit(_args(tmp_path, "ap_cut"))
    assert report["valid_cases"] == 1
    assert report["skipped_cases"] == 1
    assert "nonplanar" in report["cases"][1]["reason"]
    assert report["baseline_all_readable_cases"]["case_count"] == 2
    assert report["baseline_valid_cases"]["case_count"] == 1
    assert report["diagnostic_auxiliary_weight"]["valid_cases"] == 1


def test_source_changes_prevent_publication(tmp_path, monkeypatch):
    _write_case(tmp_path / "case_01.npz")
    snapshots = iter(("before", "after"))
    monkeypatch.setattr(audit, "source_digest", lambda: next(snapshots))
    with pytest.raises(RuntimeError, match="Source files changed"):
        audit.run_audit(_args(tmp_path))
    assert not (tmp_path / "report.json").exists()


def test_calibration_is_rejected_and_ap_geometry_must_be_explicit(tmp_path):
    args = _args(tmp_path)
    args.purpose = "calibration"
    with pytest.raises(ValueError, match="diagnostic-only"):
        audit.run_audit(args)
    args.purpose = "diagnostic"
    args.constraint = "ap_cut"
    with pytest.raises(ValueError, match="explicit"):
        audit.run_audit(args)


def test_cuda_audit_never_falls_back_to_cpu(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.device = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        audit.run_audit(args)
    assert not (tmp_path / "report.json").exists()
