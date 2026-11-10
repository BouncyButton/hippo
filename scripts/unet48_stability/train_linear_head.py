"""Isolate normalization of the three class channels in the final residual head."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(Path(__file__).resolve().parent)]


def build_linear_head_model():
    import torch
    from monai.networks.nets import UNet
    from train_unet48_budget_pilot import KWARGS
    model = UNet(**KWARGS)
    top = model.model[2][0]
    assert isinstance(top.adn.N, torch.nn.InstanceNorm3d)
    assert not top.adn.N.affine and isinstance(top.adn.A, torch.nn.PReLU)
    assert top.conv.out_channels == 3
    # Hidden feature normalization remains unchanged. The final upsampling
    # convolution and following linear residual unit now form a linear head.
    top.adn = torch.nn.Identity()
    return model


def main():
    from train_stability import install
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir/'protocol.json').read_text())
    assert protocol['stability_adapter_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert protocol['decoder'] == 'final_linear_transpose'
    trainer = install(protocol['stability_recipe'])

    def build_model(name, architecture):
        assert name == 'unet'
        return build_linear_head_model()

    trainer.base.build_model = build_model
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = {'recipe': protocol['stability_recipe'], 'decoder': protocol['decoder'],
              'adapter_sha256': protocol['stability_adapter_sha256'],
              'schedule_adapter_sha256': hashlib.sha256(Path(__file__).with_name('train_stability.py').read_bytes()).hexdigest(),
              'protocol_sha256': trainer.base.digest(args.protocol_dir/'protocol.json')}
    target = args.output/'runtime_recipe.json'
    if target.exists():
        assert json.loads(target.read_text()) == recipe
    else:
        trainer.base.atomic_json(target, recipe)
    trainer.main()


if __name__ == '__main__':
    main()
