"""Test a resize-convolution output decoder, retaining width-48 residual blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(Path(__file__).resolve().parent)]


def build_resize_model():
    import torch
    from monai.networks.nets import UNet
    from train_unet48_budget_pilot import KWARGS
    model = UNet(**KWARGS)
    convolution = model.model[2][0]
    old = convolution.conv
    assert isinstance(old, torch.nn.ConvTranspose3d)
    assert old.out_channels == 3 and old.kernel_size == (3, 3, 3) and old.stride == (2, 2, 2)
    rng = torch.get_rng_state()
    replacement = torch.nn.Conv3d(old.in_channels, old.out_channels, kernel_size=3,
                                  padding=1, bias=old.bias is not None)
    # Reuse the seeded weights with the channel axes transposed. This retains
    # every initial scalar and RNG state, isolating the upsampling operation.
    with torch.no_grad():
        replacement.weight.copy_(old.weight.transpose(0, 1))
        if old.bias is not None:
            replacement.bias.copy_(old.bias)
    torch.set_rng_state(rng)
    convolution.conv = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode='nearest'), replacement)
    return model


def main():
    from train_stability import install
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir/'protocol.json').read_text())
    assert protocol['stability_adapter_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert protocol['decoder'] == 'final_nearest_resize_conv3'
    trainer = install(protocol['stability_recipe'])

    def build_model(name, architecture):
        assert name == 'unet'
        return build_resize_model()

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
