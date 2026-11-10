"""Milder LR reductions after half-rate nnUNet failed within the short cap."""
from scripts.mni_constraints import nnunet_seed19_lr as study

study.NAME = 'mni_nnunet_seed19_lr_refinement_shared43t_20261108'
study.MODULE = 'scripts.mni_constraints.nnunet_seed19_lr_refinement'
study.RATES = (.009, .008, .007)


if __name__ == '__main__':
    study.main()
