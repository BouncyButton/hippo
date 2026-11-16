"""Existing deterministic125 worker with only the calibration hook changed."""
import signal
from scripts.colab_hfh.medsam_gain_replication_worker import worker_source as reference


def worker_source():
    source = reference()
    for name in ('install as install_aug', 'install_calibration'):
        old = 'from scripts.colab_hfh.medsam_gain_replication import ' + name
        new = 'from scripts.colab_hfh.medsam_band_recalibration import ' + name
        if source.count(old) != 1:
            raise ValueError('Audited worker hook changed')
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    exec(compile(worker_source(), __file__, 'exec'), {'__name__':'__main__', '__file__':__file__})
