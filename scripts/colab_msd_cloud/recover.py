"""Conservative, CPU-only repair: preserve scientific results, never invent them."""
import csv
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

from store import atomic, digest, member

CAP = 31976450667090
CORE = ('config.json','completion.json','initialization.json','calibration.json',
        'metrics.csv','training_final.json','validation_best.json','validation_latest.json')


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def checked_bytes(name, data, item):
    member(name)
    if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
        raise ValueError('Hash/size mismatch: '+name)
    if name.endswith('.json'):
        json.loads(data)
    elif name.endswith('.jsonl'):
        for line in data.decode().splitlines():
            json.loads(line)
    elif name.endswith('.csv'):
        list(csv.DictReader(io.StringIO(data.decode())))
    elif not name.endswith('.png'):
        data.decode()


def candidates(blobs, all_blobs, expected):
    result = {}
    for name, info in expected.items():
        if name in blobs:
            continue
        for other, content in all_blobs.items():
            if len(content)==info['bytes'] and hashlib.sha256(content).hexdigest()==info['sha256']:
                result[name] = (content, 'Exact duplicate: '+other)
                break
    diagnostic = json.loads(blobs.get('evaluation_diagnostics.json', b'{}'))
    for name, value in diagnostic.get('evaluations', {}).items():
        result.setdefault('validation_'+name+'.json', (json_bytes(value),'Saved evaluation diagnostics'))
    completion = json.loads(blobs.get('completion.json', b'{}'))
    if completion and 'budget_ledger.json' not in blobs:
        keys = {'config_sha256':'config_sha256','charged_attempts':'training_attempts_charged',
                'training_flops':'consumed_training_flops','calibration_flops':'calibration_flops',
                'calibration_attempts':'calibration_attempts','calibration_sha256':'calibration_sha256'}
        if all(v in completion for v in keys.values()):
            result['budget_ledger.json'] = (json_bytes({k:completion[v] for k,v in keys.items()}), 'Completion ledger fields')
    if 'metrics.csv' not in blobs and 'train.log' in blobs:
        rows = []
        for line in blobs['train.log'].decode().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row,dict) and all(k in row for k in ('step','charged_attempts','macro_hard_dice')):
                rows.append({k:v for k,v in row.items() if k not in ('model','objective','fold','seed')})
        if rows:
            f = io.StringIO(newline=''); w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            result['metrics.csv'] = (f.getvalue().encode(), 'Exact CSV reconstructed from saved training log')
    return result


def science_check(blobs, task, row, reference):
    missing = [n for n in CORE if n not in blobs]
    if missing:
        raise ValueError('Unrecoverable required scientific files: '+', '.join(missing))
    cfg, done, init = (json.loads(blobs[n]) for n in ('config.json','completion.json','initialization.json'))
    if (cfg['model_identity'],cfg['objective'],cfg['fold'],cfg['seed']) != (task['model'],task['arm'],task['fold'],task['seed']):
        raise ValueError('Scientific task identity differs')
    if cfg['augmentation'] != 'none' or cfg['budget_flops'] != CAP or done['budget_flops'] != CAP:
        raise ValueError('Augmentation or budget differs')
    if done['status']!='complete' or done['discarded_attempts'] or done['updates']!=cfg['max_steps']:
        raise ValueError('Incomplete optimization trajectory')
    if (done['consumed_training_flops']>CAP or done['updates']!=row['updates']
            or done['consumed_training_flops']!=row['training_flops']
            or done['calibration_flops']!=row['calibration_flops']
            or done['training_attempts_charged']!=done['updates']):
        raise ValueError('Budget/update evidence differs')
    expected_init = reference['models'][task['model']][str(task['seed'])]['model_sha256']
    if init['model_state_sha256'] != expected_init:
        raise ValueError('Initialization differs')
    for name,key in [('config.json','config_sha256'),('metrics.csv','metrics_sha256'),
                     ('calibration.json','calibration_sha256'),('validation_best.json','validation_best_sha256')]:
        if hashlib.sha256(blobs[name]).hexdigest() != done[key]:
            raise ValueError('Completion proof differs: '+name)
    for kind, cohort in [('training_final','train_cases'),('validation_best','validation_cases'),('validation_latest','validation_cases')]:
        value = json.loads(blobs[kind+'.json'])
        if {r['case_name'] for r in value['cases']} != set(cfg[cohort]):
            raise ValueError('Incomplete evaluation cohort: '+kind)
        if kind.startswith('validation') and any('confusion' not in r or 'foreground_fn' not in r for r in value['cases']):
            raise ValueError('Missing FN/FP diagnostics')
    final = json.loads(blobs['validation_latest.json'])
    if abs(100*final['summary']['macro_hard_dice']-row['dice'])>1e-8:
        raise ValueError('Final Dice differs from pre-disconnect snapshot')
    return done


def derive_completion(blobs, row, original_manifest):
    """A NEW receipt from independently saved full-run evidence, never fake bytes.

    Wall time and checkpoint hashes which were not saved are explicitly unknown.
    Every scientific field must be supported by the saved final ledger, curves,
    evaluations and the pre-disconnect results snapshot.
    """
    required=[n for n in CORE if n!='completion.json']+['budget_ledger.json']
    if any(n not in blobs for n in required):return None
    cfg=json.loads(blobs['config.json']);ledger=json.loads(blobs['budget_ledger.json'])
    rows=list(csv.DictReader(io.StringIO(blobs['metrics.csv'].decode())))
    if (not rows or ledger['config_sha256']!=hashlib.sha256(blobs['config.json']).hexdigest()
            or ledger['calibration_sha256']!=hashlib.sha256(blobs['calibration.json']).hexdigest()
            or ledger['charged_attempts']!=cfg['max_steps']
            or int(rows[-1]['step'])!=cfg['max_steps']
            or int(rows[-1]['charged_attempts'])!=ledger['charged_attempts']
            or int(rows[-1]['estimated_training_flops'])!=ledger['training_flops']
            or ledger['training_flops']!=row['training_flops']
            or ledger['calibration_flops']!=row['calibration_flops']):
        raise ValueError('Cannot derive completion from inconsistent evidence')
    best=json.loads(blobs['validation_best.json'])['summary']['macro_hard_dice']
    latest=json.loads(blobs['validation_latest.json'])['summary']['macro_hard_dice']
    if abs(float(rows[-1]['macro_hard_dice'])-latest)>1e-10:
        raise ValueError('Final curve and final evaluation differ')
    chosen=max(rows,key=lambda x:float(x['macro_hard_dice']))
    if abs(float(chosen['macro_hard_dice'])-best)>1e-10:
        raise ValueError('Best curve and best evaluation differ')
    candidates=(original_manifest or {}).get('deletion_candidates',[])
    checkpoint_hashes={x['name']:x['sha256'] for x in candidates}
    value=dict(status='complete',model=cfg['model'],objective=cfg['objective'],fold=cfg['fold'],seed=cfg['seed'],
        updates=ledger['charged_attempts'],training_attempts_charged=ledger['charged_attempts'],discarded_attempts=0,
        calibration_attempts=ledger['calibration_attempts'],calibration_flops=ledger['calibration_flops'],
        consumed_training_flops=ledger['training_flops'],consumed_total_flops=ledger['training_flops']+ledger['calibration_flops'],
        unused_flops=cfg['budget_flops']-ledger['training_flops'],budget_flops=cfg['budget_flops'],budget_scope='optimization_only',
        selected_step=int(chosen['step']),best_validation_hard_dice=best,
        final_training_hard_dice=json.loads(blobs['training_final.json'])['summary']['macro_hard_dice'],
        seconds_this_invocation=None,config_sha256=digest_bytes(blobs['config.json']),
        checkpoint_best_sha256=checkpoint_hashes.get('checkpoint_best.pt'),
        checkpoint_latest_sha256=checkpoint_hashes.get('checkpoint_latest.pt'),
        metrics_sha256=digest_bytes(blobs['metrics.csv']),budget_ledger_sha256=digest_bytes(blobs['budget_ledger.json']),
        validation_best_sha256=digest_bytes(blobs['validation_best.json']),calibration_sha256=digest_bytes(blobs['calibration.json']),
        reconstructed_after_vm_loss=True,
        reconstruction='Derived from independently saved final ledger, full curves, evaluations and pre-disconnect snapshot; not the missing original receipt.')
    return json_bytes(value)


def digest_bytes(data):return hashlib.sha256(data).hexdigest()


def recover_one(source, destination, task, row, reference, binding, all_blobs=None, protocol_sha=None):
    """Write only a new staging directory; never edits the original archive."""
    blobs = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    original_manifest = json.loads(blobs['archive_manifest.json']) if 'archive_manifest.json' in blobs else None
    expected = original_manifest['metadata'] if original_manifest else {}
    if original_manifest and original_manifest['task'] != task:
        raise ValueError('Archive manifest identity changed')
    # Completion provides independent hashes even when its manifest was lost.
    done = json.loads(blobs.get('completion.json', b'{}'))
    expected = deepcopy(expected)
    for name,key in [('config.json','config_sha256'),('metrics.csv','metrics_sha256'),
                     ('calibration.json','calibration_sha256'),('validation_best.json','validation_best_sha256')]:
        if name not in expected and name in blobs and key in done:
            expected[name] = dict(bytes=len(blobs[name]), sha256=done[key])
    repairs = {}
    proposals = candidates(blobs,all_blobs or blobs,expected)
    if 'config.json' not in blobs:
        configs = []
        for key,content in (all_blobs or blobs).items():
            if key.endswith('config.json'):
                configs.append(json.loads(content))
        cohort = next((c for c in configs if c.get('fold')==task['fold']), None)
        for cfg in configs:
            if cfg.get('model_identity')!=task['model'] or cfg.get('objective')!=task['arm'] or not cohort:
                continue
            candidate=deepcopy(cfg)
            candidate.update(fold=task['fold'],seed=task['seed'],train_cases=cohort['train_cases'],validation_cases=cohort['validation_cases'])
            if protocol_sha:
                candidate['protocol_sha256']=protocol_sha
            data=json_bytes(candidate)
            expected_sha=expected.get('config.json',{}).get('sha256') or done.get('config_sha256')
            if hashlib.sha256(data).hexdigest()==expected_sha:
                expected.setdefault('config.json',dict(bytes=len(data),sha256=expected_sha))
                proposals['config.json']=(data,'Frozen config template; exact original checksum matched')
                break
    if 'initialization.json' not in blobs:
        proposals['initialization.json']=(json_bytes({'model_state_sha256':reference['models'][task['model']][str(task['seed'])]['model_sha256']}),'Canonical initialization reference; exact checksum required')
    if 'validation_best.json' not in blobs and 'validation_latest.json' in blobs and done.get('validation_best_sha256'):
        data=blobs['validation_latest.json']
        if hashlib.sha256(data).hexdigest()==done['validation_best_sha256']:
            expected.setdefault('validation_best.json',dict(bytes=len(data),sha256=done['validation_best_sha256']))
            proposals['validation_best.json']=(data,'Best and final evaluation have identical recorded hash')
    for name,(data,why) in proposals.items():
        if name in blobs or name not in expected:
            continue
        try:
            checked_bytes(name,data,expected[name])
        except ValueError:
            continue
        blobs[name] = data; repairs[name] = why
    if 'budget_ledger.json' not in blobs:
        ledgers=[]
        for name,data in blobs.items():
            if name.startswith('concurrency_'):
                obs=json.loads(data)
                if obs.get('completed') and obs.get('ledger'):ledgers.append((obs['ledger'],'Saved completed-worker observation'))
        if 'config.json' in blobs and 'calibration.json' in blobs:
            cfg=json.loads(blobs['config.json'])
            ledgers.append((dict(config_sha256=digest_bytes(blobs['config.json']),charged_attempts=row['updates'],
                training_flops=row['training_flops'],calibration_flops=row['calibration_flops'],
                calibration_attempts=1 if row['calibration_flops'] else 0,
                calibration_sha256=digest_bytes(blobs['calibration.json'])),
                'Saved config, calibrated weights and result ledger fields; exact original hash required'))
        for value,why in ledgers:
            data=json_bytes(value)
            if 'budget_ledger.json' in expected:
                try:checked_bytes('budget_ledger.json',data,expected['budget_ledger.json'])
                except ValueError:continue
            elif why!='Saved completed-worker observation':continue
            blobs['budget_ledger.json']=data;repairs['budget_ledger.json']=why;break
    derived=False
    if 'completion.json' not in blobs:
        data=derive_completion(blobs,row,original_manifest)
        if data:
            blobs['completion.json']=data;derived=True
            repairs['completion.json']='NEW completion receipt derived from matching final ledger, complete curves and evaluations; original receipt bytes are unavailable'
    for name,item in expected.items():
        if name in blobs and not (name=='completion.json' and derived):
            checked_bytes(name, blobs[name], item)
    science_check(blobs, task, row, reference)
    ack = json.loads(blobs.get('archive_acknowledgment.json', b'{}'))
    if ack and (not ack.get('source_verified') or ack.get('source_manifest_sha256')!=binding['bundle_sha256']):
        raise ValueError('Original source acknowledgment differs')
    lost = sorted(set(expected)-set(blobs))
    if not original_manifest:
        lost.append('original archive_manifest.json')
    missing_receipts = [n for n in ('cleanup.json','archive_acknowledgment.json') if n not in blobs]
    provenance = dict(status='scientific_results_recovered', task=task, repairs=repairs,
        unavailable_noncritical_metadata=lost+missing_receipts,
        original_manifest_sha256=hashlib.sha256(blobs['archive_manifest.json']).hexdigest() if original_manifest else None,
        original_acknowledgment=ack or None, original_bytes_preserved=True,
        reconstructed_completion_receipt=derived,
        original_completion_sha256=expected.get('completion.json',{}).get('sha256') if derived else None,
        core_scientific_files_verified=list(CORE), verified_at=time.time(),
        note='New recovery receipt, not a reconstruction of the missing original cleanup/manifest.')
    destination.mkdir(parents=True, exist_ok=False)
    for name,data in blobs.items():
        member(name)
        if name in ('archive_manifest.json','archive_acknowledgment.json','cleanup.json'):
            continue
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    atomic(destination/'recovery_provenance.json', provenance)
    metadata = {str(p.relative_to(destination)):dict(bytes=p.stat().st_size,sha256=digest(p))
                for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest = dict(status='ready',task=task,metadata=metadata,deletion_candidates=[],created_at=time.time(),
                    completion_sha256=digest(destination/'completion.json'), recovery=True)
    atomic(destination/'archive_manifest.json',manifest)
    atomic(destination/'archive_acknowledgment.json',dict(status='verified',source_verified=True,
        source_manifest_sha256=binding['bundle_sha256'],manifest_sha256=digest(destination/'archive_manifest.json'),
        durable_archive='Recovered original Google Drive source; new server commit required before training',
        recovery_provenance='recovery_provenance.json'))
    atomic(destination/'cleanup.json',dict(status='complete',deleted=[],released_file_bytes=0,
        reason='Original VM terminated; metadata-only recovery, no checkpoint deletion performed'))
    return provenance
