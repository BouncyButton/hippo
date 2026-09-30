"""Remove an explicit, outcome-verified set of unsuccessful experiment weights.

Run --apply only after inspecting the inventory. Keep all reports, source,
metrics, selected control weights, successful low-data runs, and datasets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path('/mnt/beegfsstudents/home/3160552')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    active = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i'], text=True).strip()
    if active:
        raise RuntimeError('Active jobs exist; recheck dependencies before cleanup.')
    aux = BASE/'boundary_auxiliary_20260923_34d3_v4/results_666967'
    summary = json.loads((aux/'aggregate_summary.json').read_text())
    assert len(summary['results']) == 12
    assert all(r['selected_epoch'] == 0 for r in summary['results'])
    paths, protected = [], []
    for fold in range(1, 5):
        root = aux/f'fold{fold}'
        control = root/'control/best.pt'
        control_hash = digest(control)
        protected.append(dict(path=str(control), sha256=control_hash))
        for arm in ['endpoint', 'endpoint_completion']:
            target = root/arm/'best.pt'
            assert digest(target) == control_hash, 'Selected auxiliary weight differs from epoch-zero control.'
            paths.append((target, 'No selected improvement: epoch-zero duplicate of retained control.'))
        paths.append((root/'common.pt', 'Continuation/optimizer initialization; all arms selected epoch zero; selected control retained.'))
    direct = BASE/'direct_contour_20260923_34d3_v2/results_667067'
    treatment = json.loads((direct/'direct_contour_done.json').read_text())['fixed_inner']
    control = json.loads((direct/'control_done.json').read_text())['fixed_inner']
    assert treatment['union_dice'] < control['union_dice']
    assert treatment['assd_mm'] > control['assd_mm']
    protected.append(dict(path=str(direct/'control_final.pt'), sha256=digest(direct/'control_final.pt')))
    paths.append((direct/'direct_contour_final.pt', 'Worse inner Dice and ASSD than matched control.'))
    paths.append((direct.parent/'results_667065/roundtrip.pt', 'Smoke-test serialization of the same unsuccessful experiment.'))
    files = []
    for path, reason in paths:
        assert not path.is_symlink() and path.resolve().is_relative_to(BASE)
        stat = path.stat()
        files.append(dict(path=str(path), bytes=stat.st_size, sha256=digest(path), reason=reason))
    report = dict(status='inventory', files=files, protected=protected,
                  logical_bytes=sum(f['bytes'] for f in files),
                  metadata_retained=True, active_jobs=active)
    out = BASE/'negative_cleanup_20260928'
    out.mkdir(exist_ok=True)
    (out/'inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.apply:
        # Exact files only; no directory recursion or broad globs.
        for f in files:
            path = Path(f['path'])
            assert path.stat().st_size == f['bytes'] and digest(path) == f['sha256']
            path.unlink()
        for f in protected:
            assert digest(Path(f['path'])) == f['sha256']
        assert all(not Path(f['path']).exists() for f in files)
        report['status'] = 'complete'
        (out/'deleted.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
