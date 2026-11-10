"""Stop at the compute cap while retaining a separately declared LR horizon."""
import argparse
import json
from pathlib import Path


def fixed_horizon_learning_rate(original, horizon):
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Invalid LR horizon')
    def schedule(model, step, max_steps, initial, nnunet_warmup_fraction=0.0):
        if model != 'nnunet' or not 0 <= step < max_steps <= horizon or nnunet_warmup_fraction != 0:
            raise ValueError('Unexpected control schedule')
        return original(model, step, horizon, initial, nnunet_warmup_fraction)
    return schedule


def main():
    import train_budget_baseline as adapter
    base = adapter.base
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args, _ = parser.parse_known_args()
    protocol_path = args.protocol_dir / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if (protocol['schedule_adapter_sha256'] != base.digest(Path(__file__))
            or protocol['nnunet_lr_horizon_updates'] != 174
            or protocol['budget_flops'] != 100 * protocol['models']['nnunet']['flops_per_update']):
        raise ValueError('Control differs from the declared schedule/budget')
    base.learning_rate = fixed_horizon_learning_rate(base.learning_rate, 174)
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {'schedule_adapter_sha256': base.digest(Path(__file__)),
                'protocol_sha256': base.digest(protocol_path),
                'lr_horizon_updates': 174, 'training_cap_updates': 100,
                'fresh_initialization': True}
    target = args.output / 'schedule_control.json'
    if target.exists() and json.loads(target.read_text()) != evidence:
        raise ValueError('Schedule changed on resume')
    base.atomic_json(target, evidence)
    adapter.main()


if __name__ == '__main__':
    main()
