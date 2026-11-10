"""Bind a classical CNN preset to the existing compute-accounted Dice trainer."""
import argparse
import json
from pathlib import Path
import sys

import train_compute_matched_baselines as base
from baselines.simple_segmentation import build_candidate


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir / 'protocol.json').read_text())
    for name, expected in protocol['training_source_sha256'].items():
        if base.digest(base.REPO / name) != expected:
            raise ValueError('Training source differs from profile/protocol: ' + name)
    candidate = protocol['candidate_name']
    base.build_model = lambda name, architecture: build_candidate(candidate)
    cost = protocol['models']['unet']['flops_per_update']
    cap = protocol['budget_flops'] // cost
    milestones = sorted({max(1, min(cap, target // cost))
                         for target in protocol['validation_target_flops']} | {cap})
    base.validation_steps = lambda max_steps, count=20: sorted(
        {min(max_steps, step) for step in milestones})
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = {'candidate': candidate, 'architecture': protocol['candidate'],
              'protocol_sha256': base.digest(args.protocol_dir / 'protocol.json'),
              'adapter_sha256': base.digest(Path(__file__)),
              'validation_updates': milestones,
              'learning_rate_policy': protocol['learning_rate_policy'],
              'internal_trainer_alias': 'unet'}
    path = args.output / 'runtime_recipe.json'
    if path.exists() and json.loads(path.read_text()) != recipe:
        raise ValueError('Resume recipe changed')
    base.atomic_json(path, recipe)
    base.main()


if __name__ == '__main__':
    main()
