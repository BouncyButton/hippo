"""Spatial dimensions and class definitions for the MNI worker."""
DATASETS = {
    'MNI': dict(folder='Dataset102_MNI', native_shape=(40, 56, 48),
                spatial_size=(64, 64, 64), train_subjects=5,
                classes=['background', 'CA1_3', 'subiculum', 'CA4_DG']),
    'COBRA': dict(folder='Dataset105_COBRA', native_shape=(48, 64, 80),
                  spatial_size=(64, 64, 96), train_subjects=4,
                  classes=['background', 'CA1', 'subiculum', 'CA4_DG', 'CA2_3', 'SR_SL_SM']),
}
