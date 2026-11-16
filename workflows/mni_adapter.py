"""Bind the standalone MNI worker to the paired, observed training trajectory."""
import sys
from pathlib import Path
from scripts.colab_mni_medsam3 import optimized
from scripts.cross_dataset_campaign import adni_foundation_full_runtime as full
from scripts.mni_medsam3_cluster.constraint_support import install


def main():
    p = optimized.read(Path(sys.argv[sys.argv.index("--protocol") + 1]))
    engine, matched = optimized.original, optimized.matched
    engine.Volumes = optimized.Volumes
    engine.SupportSlabs = optimized.SupportSlabs
    engine.load_foundation = matched.load_foundation
    # This also binds the recorded native rates across milestone extensions.
    # The geometry wrapper is only called for arms with a depth term.
    install(p)
    engine.calibrate = matched.calibrate if sys.argv[sys.argv.index("--arm") + 1] == "dice" else full.calibrate
    matched.configure_retained_backward()
    optimized.install_evaluation()
    engine.main()


if __name__ == "__main__":
    main()
