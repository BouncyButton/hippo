"""Bounded offline checkpoint benchmark; no campaign or GPU allocation access."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import torch

from scripts.colab_medsam3.performance_candidates import packed_cpu_snapshot

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT/'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015/bundle'


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def benchmark(output, trials=5):
    torch.set_num_threads(4)
    adapter = torch.load(BUNDLE/'weights/best_lora_weights.pt', map_location='cpu', weights_only=False)
    parameters = [torch.nn.Parameter(value.clone()) for value in adapter.values()]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=.01)
    # Synthetic moments with real adapter dimensions; no model is trained.
    for parameter in parameters:
        optimizer.state[parameter] = dict(step=torch.tensor(84.),
            exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.ones_like(parameter))
    state = dict(step=84, adapter=adapter, optimizer=optimizer.state_dict(),
                 rng=dict(torch=torch.get_rng_state()), history=[dict(update=i+1) for i in range(84)],
                 protocol_sha256='offline_fixture_not_a_scientific_checkpoint', elapsed_seconds=0.)
    methods = ('original_zip', 'legacy_serialization', 'packed_zip')
    measurements = {key: [] for key in methods}
    sizes = {}
    with tempfile.TemporaryDirectory(prefix='medsam3_io_benchmark_', dir='/private/tmp') as scratch:
        destination = Path(scratch)/'checkpoint_latest.pt'
        for trial in range(trials+1):
            # Alternate order; first round is warm-up and excluded.
            for method in methods if trial % 2 == 0 else reversed(methods):
                gc.collect()
                temporary = destination.with_suffix('.tmp')
                start = time.perf_counter()
                value = packed_cpu_snapshot(state) if method == 'packed_zip' else state
                packed_at = time.perf_counter()
                torch.save(value, temporary, _use_new_zipfile_serialization=method != 'legacy_serialization')
                temporary.replace(destination)
                finished = time.perf_counter()
                if trial:
                    measurements[method].append(dict(snapshot_seconds=packed_at-start,
                        save_seconds=finished-packed_at, total_seconds=finished-start))
                sizes[method] = destination.stat().st_size
                if trial == 0 or trial == trials:
                    recovered = torch.load(destination, map_location='cpu', weights_only=False)
                    if not same(state, recovered): raise AssertionError(method+' changed checkpoint values')
                    del recovered
                del value
        results = {method: dict(median_seconds=statistics.median(x['total_seconds'] for x in values),
                    minimum_seconds=min(x['total_seconds'] for x in values), maximum_seconds=max(x['total_seconds'] for x in values),
                    file_bytes=sizes[method], trials=values) for method, values in measurements.items()}
    base = results['original_zip']['median_seconds']
    for value in results.values():
        value['component_speedup_vs_original'] = base/value['median_seconds']
        value['seconds_saved_per_200_saves_on_this_cpu'] = 200*(base-value['median_seconds'])
    report = dict(status='offline_only_not_deployable', environment=dict(python=platform.python_version(),
        torch=torch.__version__, platform=platform.platform(), device='cpu', threads=torch.get_num_threads(),
        cuda_available=torch.cuda.is_available()),
        adapter_tensors=len(adapter), adapter_elements=sum(t.numel() for t in adapter.values()),
        adapter_bytes=sum(t.numel()*t.element_size() for t in adapter.values()),
        fixture='Real medical adapter shapes/values; synthetic AdamW moments. No model forward or scientific optimizer updates.',
        original_save_policy='Atomic temporary file + rename, every update; no added fsync, matching frozen implementation.',
        trials_per_method=trials, optimizer_updates=0, campaign_files_modified=False,
        payload_equal_after_load=True, results=results,
        limits=['Mac CPU/local SSD; not the Colab A100 or its local disk.',
                'No GPU-to-CPU transfer timing and no Google Drive latency measurement.',
                'Serialization component timings cannot establish end-to-end training improvement.',
                'No live workers, checkpoints or published notebooks were altered.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a != 'trials'} for k,v in results.items()}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=5)
    args=parser.parse_args()
    if not 3 <= args.trials <= 7: parser.error('Use 3–7 bounded repetitions.')
    benchmark(args.output, args.trials)
