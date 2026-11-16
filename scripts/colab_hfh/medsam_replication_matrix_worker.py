"""Run one fold/seed in a dedicated process, using the seed-0 objective implementations."""
import os
from pathlib import Path
import signal
import sys

from scripts.colab_hfh import medsam_seed0_matrix_worker as matrix
from scripts.colab_hfh import medsam_gain_replication as gain
from scripts.colab_hfh import medsam_gain_replication_worker as native_worker
from scripts.colab_hfh.medsam_matrix_acceleration import install
from scripts.cross_dataset_campaign.common import digest


def check_protocol(p):
    gain.check_protocol(p)
    if p['seed'] not in (1, 2) or p['requested_arms'] not in [[arm] for arm in matrix.ARMS]:
        raise ValueError('Expected one replication objective')
    if (p['adaptive_band'] != matrix.band.POLICY or p['constraint_warmup_updates'] != 20
            or p['learning_rate_horizon_updates'] != 200):
        raise ValueError('Frozen recipe changed')
    for name, expected in p['matrix_source_hashes'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Source changed: '+name)


class OnFilesystemCheckpoint:
    """The cluster trainer already atomically commits every checkpoint to persistent storage."""
    def __init__(self, output, durable, interval):
        self.output = output

    def capture(self, force=False):
        pass

    def close(self, force=False):
        pass


def main():
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    arm = sys.argv[sys.argv.index('--arm')+1]
    matrix.check_protocol = check_protocol
    install()
    if arm == 'dice':
        source = native_worker.worker_source()
    else:
        source = matrix.worker_source().replace('from scripts.colab_hfh.medsam_adaptive_band import install',
                                                'from scripts.colab_hfh.medsam_seed0_matrix_worker import install')
    if os.environ.get('SLURM_JOB_ID'):
        import torch
        from scripts.colab_hfh import medsam_recovery, medsam_adaptive_band_recovery
        from scripts.cross_dataset_campaign.common import write
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Worker must see its one assigned BF16 GPU')
        output = Path(sys.argv[sys.argv.index('--output')+1])
        device = torch.cuda.get_device_properties(0)
        write(output/'gpu_binding.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=device.name,
              uuid=str(device.uuid), memory=device.total_memory, step=os.environ.get('SLURM_STEP_ID')))
        medsam_adaptive_band_recovery.RecoveryWriter = OnFilesystemCheckpoint
        medsam_recovery.RecoveryWriter = OnFilesystemCheckpoint
    exec(compile(source, __file__, 'exec'), {'__name__':'__main__','__file__':__file__})


if __name__ == '__main__':
    main()
