"""Bind shared training-only calibration to its support data and initialization."""
import hashlib
import json
import math
from pathlib import Path


def calibration_context(protocol, inventory, adapter_before, frozen_before):
    keys = ('train_cases', 'base_sha256', 'calibration_seed',
            'calibration_slabs_per_volume', 'gradient_ratio_target', 'gradient_ratio_p95_cap')
    return dict(protocol={key: protocol[key] for key in keys},
                inventory_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
                adapter_before=adapter_before, frozen_before=frozen_before)


def load_calibration(path, expected_sha256, context, source_run):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Shared calibration file changed.')
    report = json.loads(raw)
    if report.get('context') != context or report.get('source_run') != source_run:
        raise ValueError('Shared calibration support, settings or initialization mismatch.')
    if report.get('evaluation_labels_used') is not False:
        raise ValueError('Shared calibration is not training-only.')
    for kind in ('bands', 'edge'):
        weight = report[kind]['weight']
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Invalid shared calibration coefficient.')
    return report
