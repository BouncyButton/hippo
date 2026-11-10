"""Small, dependency-free contracts shared by trainers and the controller."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

CAP = 31976450667090
SEISMIC_CAP = 182178126543160  # 152 Dice-only nnUNet updates; 19 volumes × 8 passes.
ARMS = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge',
        'dice_bands_edge_sdf', 'dice_depth', 'dice_bands_depth', 'dice_edge_depth',
        'dice_sdf_depth', 'dice_bands_edge_depth', 'dice_bands_sdf_depth',
        'dice_edge_sdf_depth', 'dice_bands_edge_sdf_depth')
MODELS = ('nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16')
REMOTE = Path('/mnt/beegfsstudents/home/3160552')
NAME = 'cross_dataset_exploration_20261014'
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def signal_stop(signum, frame):
    global STOP
    STOP = True


def install_signals():
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal_stop)


def stopping():
    control = os.environ.get('HIPPO_CONTROL')
    deadline = float(os.environ.get('HIPPO_DEADLINE', 'inf'))
    return STOP or time.time() >= deadline or (control and read(control).get('pause', False))


def check_no_augmentation(protocol):
    if protocol.get('augmentation') != 'none':
        raise ValueError('Only explicitly unaugmented runs are allowed')


def confusion_metrics(matrix, classes):
    """Integer confusion counts are sufficient to audit every reported error."""
    n = len(classes)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError('Confusion shape differs from class definitions')
    if any(type(v) is not int or v < 0 for row in matrix for v in row):
        raise ValueError('Counts must be nonnegative integers')
    metrics = {}
    for c, name in enumerate(classes):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(n)) - tp
        fn = sum(matrix[c]) - tp
        metrics[name] = dict(tp=tp, fp=fp, fn=fn,
                             dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)
    return dict(confusion=matrix, per_class=metrics,
                macro_hard_dice=sum(metrics[c]['dice'] for c in classes[1:])/(n-1),
                foreground_fp=sum(matrix[0][1:]),
                foreground_fn=sum(matrix[c][0] for c in range(1, n)),
                ap_swaps=(dict(anterior_to_posterior=matrix[1][2],
                               posterior_to_anterior=matrix[2][1]) if n == 3 else None))
