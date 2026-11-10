"""Train a native nnU-Net arm at the95-baseline-update compute cap."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objective', default='dice', choices=['dice', 'dice_bands'])
    args, _ = parser.parse_known_args()
    if args.objective == 'dice':
        import train_budget_baseline as adapter
        base = adapter.base
    else:
        import train_budget_bands as adapter
        base = adapter.trainer.base
    from train_nnunet_fixedlr_control import fixed_horizon_learning_rate
    path = args.protocol_dir / 'protocol.json'
    protocol = json.loads(path.read_text())
    expected = 33752920148595
    updates = 95 if args.objective == 'dice' else 91
    if (protocol['schedule_adapter_sha256'] != base.digest(Path(__file__))
            or protocol['lr_prefix_adapter_sha256'] != base.digest(Path(__file__).with_name('train_nnunet_fixedlr_control.py'))
            or protocol['nnunet_lr_horizon_updates'] != 174
            or protocol['planned_updates'] != updates or protocol['budget_flops'] != expected
            or protocol['maximum_common_budget_flops'] != 33752920148595):
        raise ValueError('95-update comparison differs from declared recipe')
    base.learning_rate = fixed_horizon_learning_rate(base.learning_rate, 174)
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {'schedule_adapter_sha256': base.digest(Path(__file__)),
                'lr_prefix_adapter_sha256': protocol['lr_prefix_adapter_sha256'],
                'protocol_sha256': base.digest(path), 'lr_horizon_updates': 174,
                'training_cap_updates': updates, 'fresh_initialization': True,
                'comparison_axis': 'training_flops',
                'maximum_common_budget_flops': 33752920148595}
    target = args.output / 'schedule_control.json'
    if target.exists() and json.loads(target.read_text()) != evidence:
        raise ValueError('Schedule changed on resume')
    base.atomic_json(target, evidence)
    adapter.main()


if __name__ == '__main__':
    main()
