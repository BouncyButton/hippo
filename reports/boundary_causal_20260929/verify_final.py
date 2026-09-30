"""Check completion, artifact bindings and stopping policies before synthesis."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result = dict(status='passed', trained_models={}, inference_audits={}, artifacts={})
    groups = [(ROOT / 'results', 0, ['dice', 'sum', 'dice_aug', 'sum_aug']),
              (ROOT.parent / 'boundary_replication_20260929/results', 1, ['dice', 'dice_aug']),
              (ROOT.parent / 'boundary_diversity_20260929/results', 0, ['sum10_repeated'])]
    bindings = {}
    validation_names = None
    for root, seed, arms in groups:
        for arm in arms:
            f = root / f'seed{seed}' / arm
            done, config = [read(f / name) for name in ('completion.json', 'config.json')]
            assert done['status'] == 'complete' and done['run'] == config['run']
            assert sha(f / 'selected_cases.json') == done['selected_cases_sha256']
            run = done['run']
            assert run['early_stopping_patience'] > 0
            assert run['early_stopping_min_epochs'] <= done['stopped_epoch'] <= run['epochs']
            cases = read(f / 'selected_cases.json')
            names = {r['case_name'] for r in cases if r['split'] == 'validation'}
            assert len(names) == 52
            if validation_names is None:
                validation_names = names
            assert validation_names == names
            curves = [read(p) for p in sorted((f / 'epochs').glob('*.json'))]
            assert [r['epoch'] for r in curves] == list(range(1, done['stopped_epoch']+1))
            assert all(r['skipped_updates'] == 0 for r in curves)
            key = f'seed{seed}/{arm}'
            bindings[key] = done['checkpoint_sha256']
            result['trained_models'][key] = dict(selected_epoch=done['selected_epoch'], stopped_epoch=done['stopped_epoch'],
                optimizer_updates=curves[-1]['optimizer_updates_seen'], early_stopping_patience=run['early_stopping_patience'],
                checkpoint_sha256=done['checkpoint_sha256'], selected_cases_sha256=done['selected_cases_sha256'], skipped_updates=0)
    preflights = [read(g[0] / 'PREFLIGHT.json') for g in groups[:2]]
    assert all(p['status'] == 'passed' and p['all_prior_source_files_unchanged'] for p in preflights)
    assert preflights[0]['payload_sha256'] == preflights[1]['payload_sha256']
    assert preflights[0]['runtime'] == preflights[1]['runtime']
    result['runtime'] = preflights[0]['runtime']
    result['source_manifest_sha256'] = preflights[0]['payload_sha256']
    reference_labels = {}
    for folder in ('boundary_translation_20260929', 'boundary_translation_replication_20260929'):
        f = ROOT.parent / folder
        cp = read(f / 'results/completion.json')
        assert cp['status'] == 'complete' and cp['training'] is False
        assert cp['script_sha256'] == sha(f / 'audit.py')
        assert cp['source_manifest_sha256'] == result['source_manifest_sha256']
        for b in cp['model_bindings']:
            key = f"seed{b['seed']}/{b['arm']}"
            assert b['checkpoint_sha256'] == bindings[key]
            rows = read(f / 'results' / f"{b['arm']}.json")['cases']
            assert len(rows) == 102
            assert {r['case_name'] for r in rows if r['split'] == 'validation'} == validation_names
            for row in rows:
                name = (row['split'], row['case_name'])
                h = row['gt_geometry']['label_sha256']
                if name in reference_labels:
                    assert reference_labels[name] == h
                reference_labels[name] = h
                assert len(row['views']) == 12
            result['inference_audits'][key] = dict(cases=102, views=13,
                input_clipped_cases=sum(any(v['lost_input_nonzero_voxels'] for v in r['views']) for r in rows),
                gt_clipped_cases=sum(any(v['lost_gt_foreground_voxels'] for v in r['views']) for r in rows),
                max_shell_identity_drift=max(abs(r['identity_drift']['boundary_union_errors']) for r in rows))
    for folder in ('boundary_causal_20260929', 'boundary_diversity_20260929', 'boundary_replication_20260929',
                   'boundary_translation_20260929', 'boundary_translation_replication_20260929'):
        f = ROOT.parent / folder / 'RESULTS.json'
        result['artifacts'][str(f.relative_to(ROOT.parent))] = sha(f)
    result['total_optimizer_updates'] = sum(r['optimizer_updates'] for r in result['trained_models'].values())
    archived = []
    for allowlist in ROOT.glob('cleanup*/ALLOWLIST.json'):
        for row in read(allowlist):
            relative = row['relative_path']
            assert 'medsam' not in relative.lower()
            f = allowlist.parent / 'checkpoints' / relative
            assert sha(f) == row['sha256']
            archived.append(dict(relative_path=relative, sha256=row['sha256'], local_archive_verified=True))
    assert len(archived) == 6
    result['preserved_archives'] = archived
    result['medsam3_in_cleanup_allowlists'] = False
    (ROOT / 'VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('trained_models', 'artifacts', 'preserved_archives')}, indent=2))


if __name__ == '__main__':
    main()
