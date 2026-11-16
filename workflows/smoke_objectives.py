"""Exercise the four spatial terms together on a synthetic 64-cube, without training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main(dataset):
    import torch
    if dataset == "MSD":
        from scripts.capped_depth_delay.objectives import training_loss
        from thesis.new_constraints.signed_distance import signed_distance_maps
    elif dataset == "MNI":
        from functools import partial
        from scripts.mni_constraints import objectives
        signed_distance_maps = objectives.signed_distance_maps
        training_loss = partial(objectives.training_loss, identity="basic_unet16")
    else:
        from scripts.adni_constraints.objectives import signed_distance_maps, training_loss

    classes = {"MSD": 3, "MNI": 4, "binary": 2}[dataset]
    torch.set_num_threads(2)
    torch.manual_seed(0)
    labels = torch.zeros((1, 1, 64, 64, 64), dtype=torch.long)
    for k in range(1, classes):
        labels[:, :, 10*k:10*k+10, 20:40, 18:42] = k
    logits = torch.randn((1, classes, 64, 64, 64), requires_grad=True)
    depth = torch.full((64, 64, 64), 1 / 64**3)
    loss = training_loss(logits, labels, "dice_bands_edge_sdf_depth",
                         dict(bands=.1, edge=.1, sdf=.1, depth=.1), 1.,
                         target_sdf=signed_distance_maps(labels), depth_weights=depth, sdf_scale=1.)
    loss.backward()
    if not (torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0):
        raise ValueError("Nonfinite loss or invalid gradient")
    print(f"{dataset}: finite forward and backward")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("MSD", "MNI", "binary"))
    main(parser.parse_args().dataset)
