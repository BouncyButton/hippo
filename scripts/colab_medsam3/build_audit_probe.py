"""Export the candidate audit as one standalone file without changing a bundle."""
import ast
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPORT = HERE.parents[1]/'reports/medsam3_msd_colab_3fold_3seed_noaug_20261015'

RUNNER = '''
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root/'source'))
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Run the supplied cell with its isolated audit environment.')
    import torch._functorch.config as aot_config
    aot_config.donated_buffer = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    print('AUDIT SETTINGS:', torch.are_deterministic_algorithms_enabled(),
          aot_config.donated_buffer, flush=True)
    import engine
    import runtime
    from scripts.train_medsam3_few_shot import parameter_digest
    runtime.verify_bundle(root)
    globals().update(forward=engine.forward, rng_state=engine.rng_state, set_rng=engine.set_rng)
    report = dict(status='diagnostic_only', preflight_approved=False, scientific_updates=0,
        environment=runtime.environment(), bundle_sha256=runtime.digest(root/'bundle_manifest.json'),
        probe_sha256=runtime.digest(Path(__file__)), candidate_sources=CANDIDATE_SOURCES,
        gradient_rtol=1e-5, gradient_atol=1e-7,
        training_recipe_changed=False, augmentation='none')
    try:
        p = runtime.read(root/'protocols/fold0_seed17.json')
        original, model, processor, criterion, parameters, slabs, old = engine.load(root, p)
        print('AUDIT: strict original repeat, extension, optimizer resume and inference.', flush=True)
        print('Any optimizer steps are temporary diagnostic steps; no training receipts are written.', flush=True)
        with audit_empty_point_grid() as grid_audit:
            report['empty_point_grid'] = grid_audit
            checks = equivalence(original, model, processor, criterion, parameters, slabs, p)
        report['checks'] = checks
        report['adapter_restored'] = parameter_digest(model, True) == old['context']['adapter_before']
        report['frozen_unchanged'] = parameter_digest(model, False) == old['context']['frozen_before']
        if not report['adapter_restored'] or not report['frozen_unchanged']:
            raise RuntimeError('Audit did not restore the original model exactly')
        report['candidate_audit_passed'] = True
        print('CANDIDATE AUDIT PASSED; campaign preflight is still required.', flush=True)
    except Exception as exc:
        report['candidate_audit_passed'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        runtime.write(args.output, report)
        print('AUDIT_JSON:', args.output, flush=True)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
'''


def main():
    engine = (HERE/'engine.py').read_text()
    node = next(n for n in ast.parse(engine).body if isinstance(n, ast.FunctionDef) and n.name == 'equivalence')
    comparison = ast.get_source_segment(engine, node)
    helper = (HERE/'audit_numerics.py').read_text()
    hashes = {name: hashlib.sha256(source.encode()).hexdigest()
              for name, source in [('equivalence', comparison), ('audit_numerics.py', helper)]}
    content = ('"""Isolated candidate GPU audit. Does not modify or approve the campaign."""\n'
               'import argparse, json, os, sys\nfrom pathlib import Path\nfrom copy import deepcopy\n'
               'import numpy as np\n' + helper + '\n\n' + comparison + '\n\n'
               + 'CANDIDATE_SOURCES = ' + repr(hashes) + '\n' + RUNNER)
    compile(content, 'medsam3_deterministic_audit.py', 'exec')
    path = EXPORT/'medsam3_deterministic_audit.py'
    path.write_text(content)
    print(path)
    print('SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
