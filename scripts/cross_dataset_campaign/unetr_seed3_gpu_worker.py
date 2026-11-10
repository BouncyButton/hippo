"""Audit Slurm's isolated GPU binding, then invoke the unchanged trainer."""
import argparse
import os
from pathlib import Path
import socket
import sys
import time

from scripts.cross_dataset_campaign.common import write


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--probe-directory', type=Path)
    args, remaining = parser.parse_known_args()
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each experiment must see exactly one Slurm-assigned GPU')
    properties = torch.cuda.get_device_properties(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Original BF16 precision requires a supported GPU')
    if properties.name != 'NVIDIA A100 80GB PCIe MIG 4g.40gb':
        raise RuntimeError('GPU differs from the validated original cluster runtime')
    identity = str(properties.uuid)
    if not identity:
        raise RuntimeError('Missing physical CUDA device identity')
    # Exercise the assigned device before acknowledging its binding.
    torch.ones(1, device='cuda').sum().item()
    receipt = dict(status='passed', device_uuid=identity, visible_devices=1,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_step_gpus=os.environ.get('SLURM_STEP_GPUS'),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), slurm_step_id=os.environ.get('SLURM_STEP_ID'),
        rank=os.environ.get('SLURM_PROCID'), worker=os.environ.get('HIPPO_GPU_WORKER'),
        gpu_name=properties.name, total_memory=properties.total_memory,
        cpu_affinity=sorted(os.sched_getaffinity(0)), host=socket.gethostname(), time=time.time())
    if args.probe_directory:
        write(args.probe_directory/(os.environ['SLURM_PROCID']+'.json'), receipt)
        return
    if args.record is None:
        raise ValueError('GPU worker receipt is required')
    write(args.record, receipt)
    # No changes to batches, gradients, precision, calibration, or optimizer.
    sys.argv = [sys.argv[0], *remaining]
    from scripts.cross_dataset_campaign.msd_mixed_losses import main as train
    train()


if __name__ == '__main__':
    main()
