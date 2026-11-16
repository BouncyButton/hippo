"""Binary HFH adapter for the previously audited MedSAM3 optimized runner."""
from scripts.colab_hfh.data import Volumes
from scripts.colab_mni_medsam3 import optimized
from scripts.cross_dataset_campaign import foundation


def main():
    from scripts.colab_hfh.medsam_recovery import install
    install(optimized)
    optimized.Volumes = Volumes
    optimized.SupportSlabs = foundation.SupportSlabs
    optimized.main()


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    main()
