"""Run the tested Dice trainer with the frozen width-48 residual U-Net."""
import argparse
import hashlib
import json
from pathlib import Path

import train_compute_matched_constraints as trainer

KWARGS = dict(spatial_dims=3, in_channels=1, out_channels=3,
              channels=[48, 96, 192, 384, 768], strides=[2, 2, 2, 2], num_res_units=2)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir / "protocol.json").read_text())
    assert protocol["architecture_adapter_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert protocol["models"]["unet"]["model_kwargs"] == KWARGS
    head = protocol.get("unet_output_head", "legacy_normalized")
    assert head in ("legacy_normalized", "linear")
    assert protocol["models"]["unet"]["parameters"] == (43234201 if head == "linear" else 43234202)
    if head == "legacy_normalized":
        assert list(protocol["models"]["unet"]["objectives"]) == ["dice"]
    else:
        from baselines.residual_unet import build_unet48, absolute_learning_rate
        factory = Path(__file__).resolve().parents[1] / "baselines/residual_unet.py"
        assert protocol["unet_factory_sha256"] == hashlib.sha256(factory.read_bytes()).hexdigest()
        assert protocol["unet_lr_policy"] == "absolute_warmup_constant"
        assert protocol["unet_warmup_updates"] == 100

        def learning_rate(name, step, max_steps, initial, **kwargs):
            assert name == "unet"
            return absolute_learning_rate(step, initial, protocol["unet_warmup_updates"])

        trainer.base.learning_rate = learning_rate

    def build_model(name, architecture):
        assert name == "unet"
        return build_unet48() if head == "linear" else trainer.base.UNet(**KWARGS)

    trainer.base.build_model = build_model
    trainer.main()


if __name__ == "__main__":
    main()
