"""Three-arm augmentation pilot, retaining original semantic constraints."""
import signal
from scripts.colab_hfh.medsam_matrix125_worker import worker_source as reference_source


def worker_source():
    source = reference_source()
    # Install before the resume wrapper so partial checkpoints use the augmented
    # loop. Completed-checkpoint summaries are corrected by an outer wrapper.
    needle = '    install_resume(original)'
    if source.count(needle) != 1:
        raise ValueError('Worker recovery hook changed')
    source = source.replace(needle,
        '    from scripts.colab_hfh.medsam_intensity_augmentation import install as install_aug\n'
        '    install_aug(original, read(args.protocol), args.output)\n' + needle)
    needle = '    original.calibrate = full_runtime.calibrate'
    if source.count(needle) != 1:
        raise ValueError('Worker calibration hook changed')
    source = source.replace(needle, needle + '\n'
        '    from scripts.colab_hfh.medsam_intensity_augmentation import install_calibration\n'
        '    install_calibration(original, read(args.protocol), output)')
    # Recovery's completed-step branch normally writes augmentation=none.
    needle = '            original.main()'
    if source.count(needle) != 1:
        raise ValueError('Worker completion hook changed')
    return source.replace(needle, needle + '\n'
        "            summary=read(output/'training_summary.json')\n"
        "            summary['augmentation']=read(args.protocol)['augmentation']\n"
        "            write(output/'training_summary.json',summary)")


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    exec(compile(worker_source(), __file__, 'exec'), {'__name__':'__main__', '__file__':__file__})
