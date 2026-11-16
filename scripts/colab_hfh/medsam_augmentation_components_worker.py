"""Single-component native125 worker; same model, samples and LR prefix."""
import signal
from scripts.colab_hfh.medsam_augmentation125_worker import worker_source as reference


def worker_source():
    source = reference()
    for old, new in (
        ('from scripts.colab_hfh.medsam_intensity_augmentation import install as install_aug',
         'from scripts.colab_hfh.medsam_augmentation_components import install as install_aug'),
        ('from scripts.colab_hfh.medsam_intensity_augmentation import install_calibration',
         'from scripts.colab_hfh.medsam_augmentation_components import install_calibration'),
        ('Native plus calibrated constraints; exact paired samples/LR; 200 updates',
         'Native-only component augmentation pilot; 125 updates; individual native loss logging'),
    ):
        if source.count(old) != 1:
            raise ValueError('Audited worker changed: '+old)
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    exec(compile(worker_source(), __file__, 'exec'), {'__name__':'__main__', '__file__':__file__})
