"""Reuse complete measurements and unchanged audits, never old coefficients.

This is a one-way, read-only import from the known 99-run preflight. It creates
no campaign approval or calibration receipt: the new preflight still verifies
the model, data context and geometry and recalculates every coefficient.
"""
import ast
import numpy as np
import torch

from runtime import FOLDS, binding, copy_verified, digest, read, write

LEGACY_MANIFEST = '26f4f23962fb7dd1931f70c2e672e8f76d8c5f4cd983ce980b2dc041c994f20a'
CHECKS = ('original_repeatability', 'original_loss_gradient_equivalence',
          'optimizer_checkpoint_resume_exact', 'label_free_inference_exact')


def compatible_source(root):
    """Prove unchanged forward, calibration measurement and audit dependencies."""
    previous = root/'compatibility/bundle_manifest.json'
    if digest(previous) != LEGACY_MANIFEST: raise ValueError('Unknown predecessor manifest')
    old, new = read(previous), read(root/'bundle_manifest.json')
    changed = {'engine.py', 'runtime.py', 'plan.json'}
    for name, sha in old.items():
        if name not in changed and new.get(name) != sha:
            raise ValueError('Preflight dependency changed: '+name)
    excluded = {'engine.py': {'weights_for', 'preflight'},
                'runtime.py': {'verify_bundle', 'report', 'main'}}
    checked = {}
    for name, exceptions in excluded.items():
        source = root/'compatibility'/name
        if digest(source) != old[name]: raise ValueError('Predecessor source changed: '+name)
        before, after = ast.parse(source.read_text()), ast.parse((root/name).read_text())
        def definitions(tree):
            return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name not in exceptions}
        if definitions(before) != definitions(after):
            raise ValueError('Audited or scientific code changed: '+name)
        # Globals/imports can affect unchanged functions; compare them as well.
        def globals_only(tree):
            return [ast.dump(n, include_attributes=False) for n in tree.body
                    if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Expr))]
        if globals_only(before) != globals_only(after):
            raise ValueError('Audited module globals changed: '+name)
        checked[name] = sorted(definitions(before))
    # Calibration measurement/resume statements remain unchanged inside preflight.
    old_tree = ast.parse((root/'compatibility/engine.py').read_text())
    new_tree = ast.parse((root/'engine.py').read_text())
    def measurement_loop(tree):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
        return [ast.dump(n, include_attributes=False) for n in function.body if isinstance(n, ast.For)]
    if measurement_loop(old_tree) != measurement_loop(new_tree):
        raise ValueError('Calibration measurement loop changed')
    return dict(previous_manifest_sha256=LEGACY_MANIFEST, unchanged_definitions=checked,
                measurement_loop_unchanged=True, dependencies_unchanged=True)


def validate_rows(rows, schedule):
    expected = [(str(c), int(a)) for c, a in schedule]
    if len(rows) != 20 or [(r['case'], r['anchor']) for r in rows] != expected:
        raise ValueError('Complete original 20-slab schedule required')
    for row in rows:
        gram = np.asarray(row['gram'], dtype=np.float64)
        if (gram.shape != (5, 5) or not np.isfinite(gram).all()
                or gram[0, 0] <= 0 or (np.diag(gram) < 0).any()
                or not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-12)):
            raise ValueError('Invalid saved gradient Gram matrix')
        scale = max(float(np.max(np.abs(gram))), 1.)
        if np.linalg.eigvalsh(gram).min() < -1e-10*scale:
            raise ValueError('Saved gradient Gram matrix is not positive semidefinite')
        if not np.isfinite([row['losses'][n] for n in ('native', 'bands', 'edge', 'sdf', 'depth')]).all():
            raise ValueError('Invalid saved calibration losses')


def verify_reused_evidence(folder, value):
    receipt = folder/'reuse_receipt.json'
    if not receipt.exists():
        if (folder/'measurements.json').exists(): raise ValueError('Measurements lack a reuse receipt')
        return
    record = read(receipt)
    if record['binding'] != value: raise ValueError('Reused evidence environment changed')
    for name, sha in record['files'].items():
        from runtime import relative
        if digest(folder/relative(name)) != sha: raise ValueError('Reused evidence changed: '+name)


def reuse_previous(root, durable, previous_results, previous_bundle):
    if durable == previous_results or root == previous_bundle:
        raise ValueError('The matched study requires separate bundle and results folders')
    marker = durable/'preflight_reuse.json'
    value = binding(root, durable)
    if marker.exists():
        if read(marker)['binding'] != value: raise ValueError('Reuse environment changed')
        for fold in FOLDS: verify_reused_evidence(durable/'calibration'/f'fold{fold}', value)
        print('Preflight evidence import already complete; original files remain intact.', flush=True)
        return read(marker)
    for folder in (root/'work', durable/'runs', durable/'recovery',
                   previous_bundle/'work', previous_results/'runs', previous_results/'recovery'):
        # An empty sweep lock is not scientific work.
        if any(p.is_file() and p.name != 'sweep.lock' for p in folder.rglob('*')):
            raise ValueError('Scientific work or recovery exists; preflight-only import refused')
    # Honor a deliberate pause, including when the environment prevents reuse.
    if (previous_results/'STOP').exists() and not (durable/'STOP').exists():
        copy_verified(previous_results/'STOP', durable/'STOP')
    old_binding_path = previous_results/'environment_binding.json'
    if not old_binding_path.exists():
        write(marker, dict(binding=value, status='skipped_no_previous_evidence', folds={},
                           campaign_approved=False, coefficients_imported=False))
        print('No previous preflight to import. The new preflight will run normally.', flush=True)
        return
    old_value = read(old_binding_path)
    if old_value.get('bundle_sha256') != LEGACY_MANIFEST:
        raise ValueError('Previous preflight belongs to an unknown package; original files retained')
    proof = compatible_source(root)
    if old_value['environment'] != value['environment']:
        write(marker, dict(binding=value, status='skipped_different_environment', folds={},
                           previous_binding=old_value, campaign_approved=False, coefficients_imported=False))
        print('GPU/software differs: retained previous evidence; all preflight checks will run anew.', flush=True)
        return
    source_binding = previous_results/'source_binding.json'
    if source_binding.exists() and read(source_binding).get('manifest') != LEGACY_MANIFEST:
        raise ValueError('Previous source binding disagrees')
    if (previous_bundle/'bundle_manifest.json').exists():
        if digest(previous_bundle/'bundle_manifest.json') != LEGACY_MANIFEST:
            raise ValueError('Previous staged bundle identity changed')
        from runtime import relative
        for name, sha in read(previous_bundle/'bundle_manifest.json').items():
            if digest(previous_bundle/relative(name)) != sha:
                raise ValueError('Previous staged source changed: '+name)
    result = dict(binding=value, previous_binding=old_value, compatibility=proof, folds={},
                  scientific_updates_imported=0, coefficients_imported=False,
                  campaign_approved=False, historical_results_role='reference_only')
    for fold in FOLDS:
        source = previous_results/'calibration'/f'fold{fold}'
        target = durable/'calibration'/f'fold{fold}'
        audit_path = source/'equivalence.json'
        if not audit_path.exists(): continue
        audit = read(audit_path)
        if (audit.get('binding') != old_value or audit.get('status') != 'passed'
                or not all(audit.get('checks', {}).get(key) is True for key in CHECKS)):
            raise ValueError(f'Previous fold {fold} audit is not verified')
        # Do not overwrite an independently completed new calibration/audit.
        if (target/'receipt.json').exists():
            raise ValueError('New calibration already exists; reuse must precede preflight')
        evidence = target/'previous_evidence'
        copy_verified(audit_path, evidence/'equivalence.json')
        files = {'previous_evidence/equivalence.json': digest(audit_path)}
        rows = None
        receipt_path = source/'receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            calibration = source/'calibration.json'
            if (receipt.get('status') != 'passed' or receipt.get('binding') != old_value
                    or receipt.get('calibration_sha256') != digest(calibration)):
                raise ValueError('Previous calibration receipt/hash disagrees')
            for path in (receipt_path, calibration):
                copy_verified(path, evidence/path.name)
                files['previous_evidence/'+path.name] = digest(path)
            rows = read(evidence/'calibration.json')['rows']
        elif (source/'partial.pt').exists():
            copy_verified(source/'partial.pt', evidence/'partial.pt')
            files['previous_evidence/partial.pt'] = digest(evidence/'partial.pt')
            state = torch.load(evidence/'partial.pt', map_location='cpu', weights_only=False)
            if state['binding'] != old_value: raise ValueError('Previous measurement binding differs')
            # An incomplete stream needs its RNG state. Leave it intact and rerun
            # calibration; only complete 20-sample streams are imported here.
            if len(state['rows']) == 20: rows = state['rows']
        if rows is not None:
            schedule = [(r['case'], r['anchor']) for r in read(root/'calibration'/f'fold{fold}.json')['slabs']]
            validate_rows(rows, schedule)
            write(target/'measurements.json', dict(binding=value, rows=rows,
                  origin='Complete Colab measurements from the verified predecessor',
                  previous_binding=old_value, coefficients_imported=False))
            files['measurements.json'] = digest(target/'measurements.json')
        write(target/'equivalence.json', dict(status='passed', binding=value, checks=audit['checks'],
              reused_from=dict(binding=old_value, receipt_sha256=digest(audit_path)), compatibility=proof))
        files['equivalence.json'] = digest(target/'equivalence.json')
        write(target/'reuse_receipt.json', dict(binding=value, files=files, compatibility=proof))
        verify_reused_evidence(target, value)
        result['folds'][str(fold)] = dict(audit_reused=True, measurements_reused=0 if rows is None else len(rows))
        print(f'Fold {fold}: reused verified audit and {0 if rows is None else len(rows)} measurements; no old weights imported.', flush=True)
    write(marker, result)
    return result
