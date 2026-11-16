"""Opt-in experimental worker; preparing protocols never launches this module."""
import signal
from scripts.colab_hfh.medsam_gain_replication_worker import worker_source as reference


def worker_source():
    source=reference()
    replacements=[('from scripts.colab_hfh.medsam_constraints_policy import ARMS',
                   'from scripts.colab_hfh.medsam_joint_constraints import ARMS'),
                  ('Gain-only three-arm replication;125 updates;fresh training-only calibration;individual losses logged',
                   'Experimental four-constraint gradient routing and joint block cap;125 matched updates;effectiveness unvalidated')]
    replacements.extend(('from scripts.colab_hfh.medsam_gain_replication import '+name,
                         'from scripts.colab_hfh.medsam_joint_constraints import '+name)
                        for name in ('install as install_aug','install_calibration'))
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Audited worker hook changed: '+old)
        source=source.replace(old,new)
    return source


if __name__=='__main__':
    signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    exec(compile(worker_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
