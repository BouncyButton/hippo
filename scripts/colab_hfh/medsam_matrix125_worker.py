"""Original semantic constraints, full16-arm support, spatial final evaluation."""
import signal
from scripts.colab_hfh.medsam_constraints_worker import worker_source as original_source


def worker_source():
    source=original_source()
    needle='from scripts.colab_hfh.medsam_constraints_policy import ARMS'
    if source.count(needle)!=1:raise ValueError('Worker arm guard changed')
    source=source.replace(needle,needle+"\n    ARMS = (*ARMS, 'dice_edge', 'dice_edge_sdf')")
    needle='    install_evaluation()'
    if source.count(needle)!=1:raise ValueError('Evaluation setup changed')
    return source.replace(needle,
        '    from scripts.colab_hfh.medsam_constraints_mask_only import install_spatial_evaluation\n'
        '    install_spatial_evaluation(matched)\n'+needle)


if __name__=='__main__':
    signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    exec(compile(worker_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
