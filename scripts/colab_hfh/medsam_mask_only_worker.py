"""Audited training runner with one explicit mask-only BAND intervention."""
import signal
from scripts.colab_hfh.medsam_constraints_worker import worker_source as reference_source


def worker_source():
    source = reference_source()
    needle = '    install_evaluation()'
    if source.count(needle) != 1:
        raise ValueError('Worker evaluation setup changed')
    return source.replace(needle,
        '    from scripts.colab_hfh.medsam_constraints_mask_only import install, install_spatial_evaluation\n'
        "    install(original, {**read(args.protocol), 'arm': args.arm})\n"
        '    install_spatial_evaluation(matched)\n' + needle)


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    exec(compile(worker_source(), __file__, 'exec'), {'__name__':'__main__','__file__':__file__})
