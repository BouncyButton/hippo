"""Select once after both sweeps, then repeat the frozen recipe in nine cells."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import torch
from scripts.depth_supervision.utils import atomic_json as write, digest as sha
from . import combo_train as train
from . import combo_objectives, ap82_objectives, winner_objectives, objectives


def select_candidate(candidates):
    eligible = [r for r in candidates if r['max_epochs'] == 75 and r['step_size'] == 20]
    assert eligible
    return min(eligible, key=lambda r: (-r['macro_dice'], r['objective']))


def validate_inputs(inputs, p):
    assert inputs['pkl_sha256'] == p['pkl_sha256']
    assert set(inputs['splits']) == {'0','1','2'}
    assert inputs['splits']['0'] == p['splits']['5pct']
    held_out = []
    for fold in ('0','1','2'):
        split = inputs['splits'][fold]
        assert len(split['train']) == len(set(split['train'])) == 10
        assert len(split['val']) == len(set(split['val'])) == 52
        assert not set(split['train']) & set(split['val'])
        assert set(split['train']+split['val']) <= set(inputs['native_geometry'])
        assert all(not set(split['val']) & other for other in held_out)
        held_out.append(set(split['val']))
    for name, geometry in p['native_geometry'].items():
        if name in inputs['native_geometry']:
            assert geometry == inputs['native_geometry'][name]


def components(logits, labels, entry, target, active=True):
    version = entry.get('objective_version')
    if version == 'outer_combo_v1':
        return combo_objectives.components(logits,labels,entry,target,active)
    if version == 'ap_boundary_v1':
        return ap82_objectives.components(logits,labels,entry,target,active)
    return winner_objectives.components(logits,labels,entry,target,active)


def targets(root, p, entry):
    if entry.get('objective_version') == 'outer_combo_v1':
        return combo_objectives.Targets(root,p,entry)
    if entry.get('objective_version') == 'ap_boundary_v1':
        from .ap82_geometry import Targets
    else:
        from .geometry import Targets
    return Targets(root,p,entry)


def select_weights(rows, entry, settings):
    """Same calibration rule, now using only this fold/seed's training prefix."""
    version = entry.get('objective_version')
    if version in ('outer_combo_v1','ap_boundary_v1'):
        base = objectives.select_weights(rows,entry['base_objective'],settings)
        current = dict(entry,base_coefficients=base['coefficients'])
        function = combo_objectives.select_weights if version == 'outer_combo_v1' else ap82_objectives.select_weights
        result = function(rows,current,settings)
        result['replication_base_calibration'] = {k:v for k,v in base.items() if k!='cases'}
        result['basis'] += '; base also recalibrated on this fold/seed training-only epoch-five prefix'
        return result
    return objectives.select_weights(rows,entry,settings)


def choose(root, p):
    from .ap82_targeted_resume import load as parent_load
    from .train import completed_record, identity
    directory = root/'replication'; directory.mkdir(exist_ok=True)
    path = directory/'selection.json'
    if path.exists():
        value = json.loads(path.read_text())
        assert value['combo_protocol_identity'] == p['identity_sha']
        for source,expected in value['evidence_sha256'].items(): assert sha(source)==expected
        return value
    parent = Path(p['predecessor_root'])
    assert json.loads((parent/'status.json').read_text())['state'] == 'complete'
    previous,_ = parent_load(parent,full=False)
    candidates=[]; definitions={}
    for folder,protocol in ((parent,previous),(root,p)):
        for e in protocol['runs']:
            record=completed_record(folder/'runs'/e['id'],identity(protocol,e))
            assert record is not None, 'Selection requires both entire sweeps to finish'
            training=e.get('training') or {}
            candidates.append(dict(objective=e['id'],root=str(folder),
                macro_dice=record['validation']['macro_dice'],
                max_epochs=training.get('max_epochs',protocol['max_epochs']),
                step_size=training.get('step_size',protocol['step_size'])))
            definitions[(str(folder),e['id'])]=e
    winner=select_candidate(candidates)
    value=dict(combo_protocol_identity=p['identity_sha'],winner=winner,
        objective=definitions[(winner['root'],winner['objective'])],candidates=candidates,
        evidence_sha256={str(f/'results.csv'):sha(f/'results.csv') for f in (parent,root)},
        selection='Highest fold0 seed0 validation macro Dice among completed max75/StepLR20 runs; lexical objective ID tie-break.',
        policy='Freeze objective geometry, selection fractions, emphasis, relative budgets and timing. Recalibrate numeric loss coefficients using only each fold/seed training prefix. No validation-based retuning.',
        cells=[dict(fold=f,seed=s,id=f'fold{f}_seed{s}') for f in range(3) for s in range(3)])
    write(path,value);return value


def cell_protocol(root, p, selection, fold, seed):
    inputs=json.loads((root/'replication_inputs.json').read_text());validate_inputs(inputs,p)
    cell=root/'replication'/f'fold{fold}_seed{seed}';cell.mkdir(exist_ok=True)
    q=deepcopy(p)
    for k in ('identity_sha','shared_warmup','source_sha256'):q.pop(k,None)
    split=inputs['splits'][str(fold)]
    entry=deepcopy(selection['objective'])
    entry.update(id=f'fold{fold}_seed{seed}',seed=seed,replication_parent=selection['winner']['objective'])
    entry.setdefault('combo',dict(terms=[]))
    q.update(name=cell.name,fold=fold,seed=seed,runs=[entry],main_runs=1,
        splits={'5pct':split},native_geometry={n:inputs['native_geometry'][n] for n in split['train']+split['val']},
        calibration_policy=selection['policy'],replication_selection_sha256=sha(root/'replication/selection.json'))
    q['identity_sha']=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
    path=cell/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==q
    else:write(path,q)
    return cell,q,entry


def run_cell(root, p, cell_id):
    selection=choose(root,p)
    spec=next(c for c in selection['cells'] if c['id']==cell_id)
    cell,q,e=cell_protocol(root,p,selection,spec['fold'],spec['seed'])
    train.objective_components=components;train.Targets=targets;train.select_weights=select_weights
    train.TERM_ORDER=('bands','pair','surface','ap','ap_ce','ap_band','ap_pair','sdf','edge','depth')
    torch.set_num_threads(4)
    return train.run(q,cell,e)


def report(root, p):
    selection=choose(root,p);rows=[]
    for spec in selection['cells']:
        cell,q,e=cell_protocol(root,p,selection,spec['fold'],spec['seed'])
        record=train.completed_record(cell/'runs'/e['id'],train.identity(q,e))
        if record is None:continue
        rows.append(dict(fold=spec['fold'],seed=spec['seed'],best_epoch=record['best_epoch'],
                         completed_epochs=record['completed_epochs'],**record['validation']))
    def summarize(group):
        values=[r['macro_dice'] for r in group]
        return dict(completed=len(values),mean_dice=statistics.mean(values) if values else None,
                    sd_dice=statistics.stdev(values) if len(values)>1 else None)
    result=dict(winner=selection['winner'],completed=len(rows),planned=9,runs=rows,
        overall=summarize(rows),by_fold={str(f):summarize([r for r in rows if r['fold']==f]) for f in range(3)},
        note='Fold0 informed loss selection. Fold1/2 validation sets overlap training/development cases from other folds; descriptive replication, not a held-out test. Fixed 10-case subset per fold across seeds.')
    write(root/'replication/results.json',result)
    if rows:
        with (root/'replication/results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # User requested completed scratch cleanup after recording results. Keep every
    # run's predictions, config, curves, calibration and completion/retirement receipts.
    for spec in selection['cells']:
        cell,q,e=cell_protocol(root,p,selection,spec['fold'],spec['seed'])
        folder=cell/'runs'/e['id'];record=train.completed_record(folder,train.identity(q,e))
        checkpoint=folder/'best.pt'
        if record is None or not checkpoint.exists():continue
        assert not checkpoint.is_symlink()
        if 'best.pt' in record['artifacts']:
            artifact={'best.pt':record['artifacts'].pop('best.pt')};assert sha(checkpoint)==artifact['best.pt']
            write(folder/'checkpoint_retired.json',dict(identity=record['identity'],artifacts=artifact,
                reason='Replication metrics and predictions recorded; completed scratch checkpoint retired.'))
            record['retired_artifacts']=artifact;write(folder/'completion.json',record)
        checkpoint.unlink()
    return result
