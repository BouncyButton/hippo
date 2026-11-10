"""Read frozen native MNI crops without relabeling the three subfields."""
from pathlib import Path
import numpy as np


class Volumes:
    def __init__(self, protocol):
        self.protocol = protocol

    def load(self, case, labels=True):
        from evaluation.medsam3_zero_shot import normalize_image
        with np.load(Path(self.protocol['data']) / (case + '.npz'), allow_pickle=False) as item:
            image = normalize_image(item['image'])
            truth = item['label'].copy() if labels else None
        if truth is not None:
            if image.shape != truth.shape or not np.isin(truth, (0, 1, 2, 3)).all():
                raise ValueError('Invalid MNI crop: ' + case)
        return image, truth


def select_support(folds, count, fold=0):
    """Nested subject subsets, fixed independently of optimization seed/outcomes."""
    split = folds[fold]
    subject = lambda case: case.rsplit('_', 1)[0]
    people = sorted({subject(c) for c in split['train']})
    order = np.random.RandomState(0).permutation(people).tolist()
    if not 1 <= count <= len(order):
        raise ValueError('Invalid number of training subjects')
    chosen = order[:count]
    train = sorted(c for c in split['train'] if subject(c) in chosen)
    val = sorted(split['val'])
    if set(chosen) & {subject(c) for c in val}:
        raise ValueError('Subject leakage')
    if len(train) != 2 * count or any(sum(subject(c) == s for c in train) != 2 for s in chosen):
        raise ValueError('Both hippocampi must be included per subject')
    return dict(train_cases=train, evaluation_cases=val, support_subjects=chosen,
                subject_order=order, subset_seed=0)
