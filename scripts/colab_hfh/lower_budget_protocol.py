"""Matched nnU-Net budget truncation; no deliberately weakened control recipe."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.colab_hfh.bands_campaign import read, write

VERSION = 'nnunet_budget75_v1'
MATCHED_FIELDS = ('model_identity','seed','fold','dataset_sha256','train_cases',
                  'validation_cases','lr_horizon_updates','initial_lr','augmentation','precision')


def native_payload(base, study=None):
    version = study['version'] if study else VERSION
    numerator, denominator = study['ratio'] if study else (3, 4)
    if not (isinstance(numerator, int) and isinstance(denominator, int)
            and 0 < numerator < denominator):
        raise ValueError('Expected a positive budget fraction below one')
    payload=deepcopy(base)
    payload.update(version=version, objective='dice', rates={'nnunet':base['rates']['nnunet']},
                   folds=[0,2,1],seeds=[1,0,2],strict_search=False)
    payload['protocols']={'nnunet':payload['protocols']['nnunet']}
    payload['references']={k:v for k,v in payload['references'].items() if k.startswith('nnunet_f')}
    for key in ('pairs','min_pair_delta_pp','reject_nonpositive_fold'):
        payload.pop(key,None)
    p=payload['protocols']['nnunet']['protocol.json']
    full_budget=p['budget_flops'];budget=full_budget*numerator//denominator
    schedule=deepcopy(p['models']['nnunet']['objectives']['dice'])
    cost=schedule['flops_per_update'];steps=budget//cost;used=steps*cost
    schedule.update(max_steps=steps,consumed_training_flops=used,consumed_flops=used,
        consumed_total_flops=used,unused_flops=budget-used,patient_pass_epochs=steps/10)
    p['models']['nnunet']['objectives']={'dice':schedule}
    p.update(name=version+'_native',budget_flops=budget,budget_gflops=budget/1e9,
        constraints=False,requested_arm_order=['dice'],sdf_policy={'dice':'not_applicable'},
        endpoint='Final lower-budget checkpoint; no best-validation selection',
        pairing='Same initialization, split, native loss, optimizer and LR trajectory as full-budget controls; truncate optimization only',
        learning_rate_policy='Unchanged full-budget LR trajectory; horizon remains90 to isolate budget truncation',
        lower_budget_study=dict(version=version,numerator=numerator,denominator=denominator,
            full_budget_flops=full_budget,changed_fields=['budget_flops','native_max_steps'],
            interpretation='Exploratory matched compute comparison; lower absolute performance is not evidence of improvement'))
    return payload


def archive_native(output,durable,model,fold,seed,protocol,reference):
    from scripts.colab_hfh import runtime
    p=read(protocol/'protocol.json');config=read(output/'config.json');done=read(output/'completion.json')
    if model!='nnunet' or config['fold']!=fold or config['seed']!=seed:
        raise ValueError('Lower-budget control identity differs')
    for key in MATCHED_FIELDS:
        if config[key]!=reference['configuration'][key]:
            raise ValueError('Native trajectory differs from full-budget control: '+key)
    if read(output/'initialization.json')!=reference['initialization']:
        raise ValueError('Lower-budget initialization differs')
    schedule=p['models']['nnunet']['objectives']['dice'];cost=schedule['flops_per_update']
    if (done['updates']!=schedule['max_steps'] or done['consumed_training_flops']!=cost*done['updates']
            or not cost*done['updates']<=p['budget_flops']<cost*(done['updates']+1)
            or done['calibration_flops']!=0 or p['budget_flops']!=
            p['lower_budget_study']['full_budget_flops']*p['lower_budget_study']['numerator']//p['lower_budget_study']['denominator']):
        raise ValueError('Lower-budget native accounting differs')
    score=read(output/'validation_latest.json')['summary']['macro_hard_dice']
    paired=dict(model=model,fold=fold,seed=seed,objective='dice',
        baseline_dice=reference['result']['validation_dice'],lower_budget_dice=score,
        delta_dice_pp=100*(score-reference['result']['validation_dice']),
        baseline_archive_sha256=reference['archive_sha256'],
        optimization_flops=done['consumed_training_flops'],updates=done['updates'],
        interpretation='Lower-budget native versus full-budget native; not a constraint gain')
    write(output/'paired_comparison.json',paired)
    runtime.archive(output,durable,'baselines',model,fold,seed,protocol)
    return paired


def references_from_native(work,payload):
    """Read all nine verified lower-budget controls, including low Dice scores."""
    references={}
    for fold in (0,1,2):
        for seed in (0,1,2):
            key=f'nnunet_f{fold}_s{seed}';path=work/(key+'.zip')
            with zipfile.ZipFile(path) as z:
                get=lambda name:json.loads(z.read('results/'+name))
                for name,digest in get('archive_manifest.json').items():
                    if Path(name).name!=name or hashlib.sha256(z.read('results/'+name)).hexdigest()!=digest:
                        raise ValueError('Lower-budget native archive integrity differs')
                protocol_bytes=z.read('protocol/protocol.json');p=json.loads(protocol_bytes)
                expected=payload['protocols']['nnunet']['protocol.json']
                if p!=expected or get('config.json')['protocol_sha256']!=hashlib.sha256(protocol_bytes).hexdigest():
                    raise ValueError('Lower-budget reference belongs to another protocol')
                result=get('result.json');configuration=get('config.json')
                if (result['fold'],result['seed'],result['model'])!=(fold,seed,'nnunet'):
                    raise ValueError('Lower-budget native archive identity differs')
                if result['updates']!=expected['models']['nnunet']['objectives']['dice']['max_steps']:
                    raise ValueError('Lower-budget native endpoint differs')
                references[key]=dict(result=result,initialization=get('initialization.json'),
                    configuration=configuration,validation=get('validation_latest.json'),
                    training=get('training_final.json'),archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    baseline_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),precision=payload['precision'])
    return references
