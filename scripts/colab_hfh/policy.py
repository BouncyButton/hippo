"""Pure scheduling rules: final-budget decisions, never mid-training rejection."""
import math

MODELS = ('basic_unet16', 'nnunet', 'unetr_vit_b16', 'swinunetr')
FOLDS = (0, 1, 2)
THRESHOLD = .5


def cells(kind):
    models = MODELS if kind == 'baselines' else ('medsam3',)
    return [(model, fold) for fold in FOLDS for model in models]


def decision(kind, completed):
    """completed maps (model, fold) to final Dice; exceptions are not failures."""
    expected = cells(kind)
    if any(key not in expected for key in completed):
        raise ValueError('Unexpected experiment identity')
    for key, score in completed.items():
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid final Dice; retain for inspection')
        if score < THRESHOLD:
            return 'rejected', None
    for key in expected:
        if key not in completed:
            return 'pending', key
    return 'accepted', None
