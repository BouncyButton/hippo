"""Restore completed training for evaluation without another optimizer update."""
from functools import wraps
import random

import numpy as np


def install(original):
    reference = original.train

    @wraps(reference)
    def train(model, criterion, parameters, slabs, protocol, calibration, arm, output):
        import torch
        if not (output/'runtime_validation.json').exists():
            from scripts.cross_dataset_campaign import adni_foundation_full_runtime as full_runtime
            # Recovery snapshots contain calibration, but not the preflight
            # receipt. Recreate that zero-update check on the original init.
            full_runtime.native_training_check(model, criterion, parameters, slabs, protocol, output)
        path = output/'checkpoint_latest.pt'
        if not path.exists():
            return reference(model, criterion, parameters, slabs, protocol, calibration, arm, output)
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['step'] != protocol['updates']:
            del state
            return reference(model, criterion, parameters, slabs, protocol, calibration, arm, output)
        if (len(state['history']) != protocol['updates']
                or state['protocol_sha256'] != original.digest(output/'protocol.json')):
            raise ValueError('Completed recovery identity/history differs')
        if (original.parameter_digest(model,False) != state['frozen_before']
                or original.parameter_digest(model,True) != state['adapter_before']):
            raise ValueError('Completed recovery initialization differs')
        checkpoint_sha = original.digest(path)
        model.load_state_dict(state['adapter'], strict=False)
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        model.eval().requires_grad_(False)
        frozen_after = original.parameter_digest(model,False)
        adapter_after = original.parameter_digest(model,True)
        if frozen_after != state['frozen_before'] or adapter_after == state['adapter_before']:
            raise ValueError('Invalid completed adapter recovery')
        original.write(output/'slab_schedule.json', slabs.schedule(protocol))
        original.write(output/'training_history.json', state['history'])
        original.write(output/'training_summary.json',dict(
            updates=state['step'], frozen_before=state['frozen_before'], frozen_after=frozen_after,
            adapter_before=state['adapter_before'], adapter_after=adapter_after,
            schedule_sha256=original.digest(output/'slab_schedule.json'),
            elapsed_seconds=state['elapsed_seconds'], augmentation='none', weights=calibration['weights'],
            learning_rate_horizon_updates=protocol.get('learning_rate_horizon_updates',protocol['updates'])))
        original.write(output/'recovery_evaluation_only.json',dict(
            restored_updates=state['step'], optimizer_updates_performed=0,
            checkpoint_sha256=checkpoint_sha, adapter_after=adapter_after))
        if original.digest(path) != checkpoint_sha:
            raise ValueError('Evaluation-only restore changed the checkpoint')
        print('RESTORED COMPLETED TRAINING: evaluation only; zero optimizer updates.', flush=True)

    original.train = train
