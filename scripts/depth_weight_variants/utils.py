"""Small durable I/O and reproducibility helpers for this experiment."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_history(path, history):
    if not history:
        return
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def stopping_state(history, patience=8, min_epochs=25, min_delta=.0005):
    reference, reference_epoch = -float('inf'), 0
    for expected, row in enumerate(history, 1):
        if row['epoch'] != expected:
            raise ValueError('Early stopping needs consecutive epoch history')
        score = row['validation_dice']
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid validation score')
        if score > reference + min_delta:
            reference, reference_epoch = score, expected
    epoch = len(history)
    return dict(epoch=epoch, reference_score=reference if history else None,
                reference_epoch=reference_epoch, epochs_without_improvement=epoch-reference_epoch,
                patience=patience, min_epochs=min_epochs, min_delta=min_delta,
                triggered=bool(patience and epoch >= min_epochs and epoch-reference_epoch >= patience))
