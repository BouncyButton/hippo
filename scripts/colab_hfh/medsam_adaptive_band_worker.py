"""Adaptive BAND worker retaining the established gain/evaluation/recovery path."""
import signal
from scripts.colab_hfh.medsam_gain_replication_worker import worker_source as reference


def worker_source():
    source = reference()
    for old,new in (
        ('from scripts.colab_hfh.medsam_gain_replication import install as install_aug',
         'from scripts.colab_hfh.medsam_adaptive_band import install as install_aug'),
        ('from scripts.colab_hfh.medsam_gain_replication import install_calibration',
         'from scripts.colab_hfh.medsam_adaptive_band import install_calibration'),
        ('from scripts.colab_hfh.medsam_recovery import RecoveryWriter, restore',
         'from scripts.colab_hfh.medsam_adaptive_band_recovery import RecoveryWriter\nfrom scripts.colab_hfh.medsam_recovery import restore'),
    ):
        if source.count(old) != 1:
            raise ValueError('Audited worker hook changed: '+old)
        source = source.replace(old,new)
    return source


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    exec(compile(worker_source(),__file__,'exec'), {'__name__':'__main__','__file__':__file__})
