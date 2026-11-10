"""Bind constraint arms without changing the frozen per-model base recipes."""
import sys
from pathlib import Path
from functools import partial
from copy import deepcopy
import math

bundle = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(bundle))
import worker
import torch
import numpy as np

original = worker.configure


def configure(p, output):
    original(p, output)
    exp = worker.exp
    if p["model_identity"] == "unetr_vit_b16":
        import constraint_objective as objective
        exp.trainer.training_loss = partial(objective.training_loss, policy=p["sdf_policy"][p["objective"]])
        exp.trainer.calibration_case = objective.calibration_case
    else:
        exp.trainer.training_loss = partial(worker.binary.training_loss, policy=p["sdf_policy"][p["objective"]])
        exp.trainer.calibration_case = worker.binary.calibration_case
    builder = exp.base.build_model

    def build(*args, **kwargs):
        model = builder(*args, **kwargs)
        if worker.io.model_digest(model) != p["reference_initializations"][str(p["seeds"][0])]:
            raise ValueError("Initialization differs from the existing matching-seed baseline")
        return model

    exp.base.build_model = build
    recipe = worker.io.read(output / "runtime_recipe.json")
    recipe.update(objective=p["objective"], constraints=p["objective"].split("_")[1:],
                  calibration_target_ratio=p["calibration_target_ratio"],
                  calibration_p95_cap=p["calibration_p95_cap"],
                  sdf_policy=p["sdf_policy"][p["objective"]],
                  baseline_reference=p["baseline_reference"],
                  bands_frame_width=2 if p["model_identity"] == "unetr_vit_b16" else 0,
                  geometry="Source array voxels; XY D4 affine restored for UNETR DEPTH",
                  cap_includes_calibration=True)
    worker.io.write(output / "runtime_recipe.json", recipe)


def preflight(p, folder, output):
    """Exercise each actual loss and exact optimizer/RNG checkpoint replay."""
    exp, io = worker.exp, worker.io
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU required")
    seed = p["seeds"][0]
    exp.base.seed_everything(seed)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    model = exp.base.build_model(p["trainer_alias"], io.read(folder / "nnunet_architecture.json")).to(device)
    criterion, optimizer = exp.base.build_loss_and_optimizer(p["trainer_alias"], model, p)
    native = criterion if p["model_identity"] == "nnunet" else None
    split = io.read(folder / "splits_10patients.json")[p["folds"][0]]
    data = exp.SeismicDataset(io.read(folder / "dataset.json"), split["train"][:1])
    x, y = data[0]["image"][None], data[0]["label"][None]
    affine = p["native_geometry"][data[0]["case_name"]]["model_affine"]
    arm = p["objective"]
    sdf = exp.trainer.signed_distance_maps(y).to(device) if "sdf" in arm else None
    depth = exp.trainer.geometry_weights(y, affine, p["depth_geometry"]).to(device) if "depth" in arm else None
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model(x.to(device))
    calibration = exp.trainer.calibration_case(raw, y.to(device), arm, sdf, native, depth)
    weights, stats = exp.trainer.select_weights([calibration], arm)
    assert set(weights) == set(arm.split("_")[1:])
    assert all(math.isfinite(w) and w > 0 for w in weights.values())
    del raw

    def step(update):
        xx, yy, current = x, y, affine
        if p["augmentation"] == "xy_d4_v1":
            from xy_augmentation import augment
            from constraint_objective import xy_affine
            xx, yy = augment(x, y, seed, update)
            current = xy_affine(affine, seed, update, 96)
        sd = exp.trainer.signed_distance_maps(yy).to(device) if "sdf" in arm else None
        dep = exp.trainer.geometry_weights(yy, current, p["depth_geometry"]).to(device) if "depth" in arm else None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xx.to(device))
            loss = exp.trainer.training_loss(logits, yy.to(device), arm, weights, 1., sd, native, dep, 1.)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite preflight loss")
        loss.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
            raise ValueError("Nonfinite preflight gradient")
        if native is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())

    first = step(0)
    # A temporary diagnostic checkpoint is removed only after exact replay succeeds.
    path = output / "preflight_resume.pt"
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), path)
    expected_loss = step(1)
    expected = io.model_digest(model)
    expected_opt = deepcopy(optimizer.state_dict())
    expected_cpu, expected_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    cp = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model"])
    optimizer.load_state_dict(cp["optimizer"])
    torch.set_rng_state(cp["cpu_rng"])
    torch.cuda.set_rng_state_all(cp["cuda_rng"])
    actual_loss = step(1)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return a.device == b.device and torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    assert math.isclose(expected_loss, actual_loss, rel_tol=1e-6, abs_tol=5e-7)
    assert io.model_digest(model) == expected
    assert equal(expected_opt, optimizer.state_dict())
    assert equal(expected_cpu, torch.get_rng_state()) and equal(expected_cuda, torch.cuda.get_rng_state_all())
    path.unlink()
    io.write(output / "gpu_preflight.json", dict(status="passed", model=p["model_identity"],
             arm=arm, first_loss=first, resume_loss=actual_loss, weights=weights,
             statistics=stats, exact_model_optimizer_rng_replay=True, diagnostic_optimizer_updates=3,
             calibration_cases=1, scientific_result=False))


worker.configure = configure
worker.preflight = preflight
worker.main()
