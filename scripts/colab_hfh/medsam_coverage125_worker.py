"""Coverage-first pair using the audited full native graph and clean evaluation."""
import signal
from scripts.colab_hfh.medsam_native125_logged_worker import worker_source as reference


def worker_source():
    source = reference()
    replacements = (
        ('from scripts.colab_hfh.medsam_constraints_support import PairedSlabs as SupportSlabs, bind_learning_rates',
         'from scripts.colab_hfh.medsam_constraints_support import bind_learning_rates\n'
         'from scripts.colab_hfh.medsam_coverage125 import coverage_slabs\nSupportSlabs = coverage_slabs()'),
        ('from scripts.colab_hfh.medsam_native125_logged import install_calibration',
         'from scripts.colab_hfh.medsam_coverage125 import install_calibration'),
        ('Native no-augmentation replication;125 updates;individual native loss logging',
         'Coverage-first native/EDGE+DEPTH pair;125 updates;fresh calibration;no augmentation'),
    )
    for old,new in replacements:
        if source.count(old) != 1:
            raise ValueError('Audited worker hook changed: '+old)
        source = source.replace(old,new)
    return source


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    exec(compile(worker_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
