"""Repeat native no-augmentation control without replacing the saved original."""
import signal
from scripts.colab_hfh.medsam_augmentation_components_worker import worker_source as reference


def worker_source():
    source=reference()
    for old,new in (
        ('from scripts.colab_hfh.medsam_augmentation_components import install as install_aug',
         'from scripts.colab_hfh.medsam_native125_logged import install as install_aug'),
        ('from scripts.colab_hfh.medsam_augmentation_components import install_calibration',
         'from scripts.colab_hfh.medsam_native125_logged import install_calibration'),
        ('Native-only component augmentation pilot; 125 updates; individual native loss logging',
         'Native no-augmentation replication;125 updates;individual native loss logging'),
    ):
        if source.count(old)!=1:raise ValueError('Audited component worker changed')
        source=source.replace(old,new)
    return source


if __name__=='__main__':
    signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    exec(compile(worker_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
