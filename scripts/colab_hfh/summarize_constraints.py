"""Independently verify downloaded constraint archives against native controls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import zipfile

from scripts.colab_hfh.constraint_archive import paired_result


def verify(path, references):
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as directory:
        manifest=json.loads(archive.read('results/archive_manifest.json'))
        for name, expected in manifest.items():
            data=archive.read('results/'+name)
            if Path(name).name != name or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Archive member mismatch: '+name)
            Path(directory,name).write_bytes(data)
        protocol_bytes=archive.read('protocol/protocol.json')
        protocol=json.loads(protocol_bytes)
        config=json.loads(archive.read('results/config.json'))
        if config['protocol_sha256'] != hashlib.sha256(protocol_bytes).hexdigest():
            raise ValueError('Protocol binding mismatch')
        key=f'{config["model_identity"]}_f{config["fold"]}_s{config["seed"]}'
        result=paired_result(Path(directory),references[key],protocol)
        if result != json.loads(archive.read('results/paired_comparison.json')):
            raise ValueError('Reported comparison differs from archived evidence')
        result.update(recipe=path.parent.name,archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return result


def audit_recipe(directory, references):
    """Audit one candidate, retaining incomplete status and every paired result."""
    from scripts.colab_hfh.constraint_mean_search import summary
    paths = sorted(directory.glob('*.zip'))
    rows = [verify(path, references) for path in paths]
    if not rows:
        raise ValueError('No experiment archives in recipe directory')
    if len({(r['model'], r['objective']) for r in rows}) != 1:
        raise ValueError('Recipe directory mixes models or objectives')
    identities = [(r['fold'], r['seed']) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError('Duplicate seed/fold evidence')
    gradients = []
    protocols = set()
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            protocols.add(hashlib.sha256(archive.read('protocol/protocol.json')).hexdigest())
            if 'results/gradient_strength_final.json' in archive.namelist():
                probe = json.loads(archive.read('results/gradient_strength_final.json'))
                if probe.get('status') == 'complete':
                    gradients.append(probe)
    if len(protocols) != 1:
        raise ValueError('Recipe directory mixes scientific protocols')
    terms = sorted(set.intersection(*(set(p['anchor']) & set(p['final']) for p in gradients))) if gradients else []
    return dict(recipe=directory.name, protocol_sha256=next(iter(protocols)),
        summary=summary(rows), results=rows,
        pooled_error_counts={key:sum(r[key] for r in rows) for key in (
            'constraint_foreground_fp','baseline_foreground_fp',
            'constraint_foreground_fn','baseline_foreground_fn')},
        fold_mean_deltas={str(f):statistics.mean(r['delta_dice_pp'] for r in rows if r['fold']==f)
            for f in range(3) if sum(r['fold']==f for r in rows)==3},
        gradient_diagnostics=dict(available_pairs=len(gradients),
            terms={term:dict(anchor_median=statistics.median(p['anchor'][term]['median'] for p in gradients),
                final_median=statistics.median(p['final'][term]['median'] for p in gradients)) for term in terms}),
        interpretation='Independent archived-evidence audit. Partial folds have no reported mean. Validation-selected exploratory recipe; no independent generalization claim. Logit-gradient diagnostics do not measure parameter updates.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--recipe',action='store_true',help='Audit one recipe directory rather than all child directories')
    parser.add_argument('--payload',type=Path,default=Path('reports/hfh_colab_exploration_20261111/bands_v1_payload.json'))
    args=parser.parse_args()
    refs=json.loads(args.payload.read_text())['references']
    if args.recipe:
        result=audit_recipe(args.directory,refs)
        target=args.directory/'candidate_audit.json'
        target.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(path=str(target),**result['summary']),indent=2))
        raise SystemExit(0)
    rows=[verify(p,refs) for p in sorted(args.directory.glob('*/*.zip'))]
    target=args.directory/'verified_results.json'
    target.write_text(json.dumps(dict(verified_pairs=len(rows),results=rows),indent=2)+'\n')
    print('Verified',len(rows),'comparisons:',target)
