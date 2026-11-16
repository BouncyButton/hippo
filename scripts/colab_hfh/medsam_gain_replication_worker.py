"""Original component worker with nine-pair scope and fresh fold-local calibration."""
import signal
from scripts.colab_hfh.medsam_augmentation_components_worker import worker_source as reference


def worker_source():
    source=reference()
    for old,new in (
        ('from scripts.colab_hfh.medsam_augmentation_components import install as install_aug',
         'from scripts.colab_hfh.medsam_gain_replication import install as install_aug'),
        ('from scripts.colab_hfh.medsam_augmentation_components import install_calibration',
         'from scripts.colab_hfh.medsam_gain_replication import install_calibration'),
        ('Native-only component augmentation pilot; 125 updates; individual native loss logging',
         'Gain-only three-arm replication;125 updates;fresh training-only calibration;individual losses logged')):
        if source.count(old)!=1:raise ValueError('Audited worker changed: '+old)
        source=source.replace(old,new)
    return source

if __name__=='__main__':
    signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    exec(compile(worker_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
