"""Supplementary historical Dice-only baseline context; existing audits only."""
import ast
import hashlib
import json
import statistics as st
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'reports/edge_coherence_20260928/results'
NEW=ROOT/'reports/separated_edge_20260929/results'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ast_def(source,name):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)

def main():
    for line in (OLD/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split();assert sha(OLD/name)==digest
    completion=read(OLD/'completion.json')
    assert sha(ROOT/'scripts/audit_edge_coherence.py')==completion['script_sha256']
    rows=[r for r in read(OLD/'cases.json') if r['fold']==0 and r['method']=='dice']
    provenance=[r for r in read(OLD/'bindings.json') if r['fold']==0 and r['method']=='dice']
    matched_keys=['fold','seed','epochs','batch_size','spatial_size','resize','optimizer_mode','learning_rate','weight_decay','step_size','adamw_gamma','amp','training_augmentation','drop_rate','activation_checkpointing','early_stopping_min_epochs','early_stopping_patience','early_stopping_min_delta','initial_checkpoint','supervised_loss']
    original_source=(OUT/'historical_baseline_trainer.py').read_text()
    current_source=(ROOT/'reports/separated_edge_20260929/source/thesis/new_constraints/train_swinunetr_constraints.py').read_text()
    equal_functions=['build_data','build_swinunetr','PlainTensorTransform','seed_everything']
    for name in equal_functions:assert ast_def(original_source,name)==ast_def(current_source,name)
    # The only validation evaluator change is an inactive added constraint flag.
    adjusted=current_source.replace('        or objective.config.sagittal_step_weight > 0\n','')
    assert ast_def(original_source,'evaluate_validation_metrics')==ast_def(adjusted,'evaluate_validation_metrics')
    for seed in range(3):
        cfg_path=next((OUT/'historical_baseline_configs').glob(f'low_data_noaug_seed{seed}*/runs/*/config.json'))
        cfg=read(cfg_path);q=read(NEW/f'fold0/seed{seed}/pooled/config.json')
        ref=next(r for r in provenance if r['seed']==seed)
        assert sha(cfg_path)==ref['config_sha256']
        assert sha(OUT/'historical_baseline_trainer.py')==cfg['source_provenance']['files']['thesis/new_constraints/train_swinunetr_constraints.py']
        assert cfg['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py']==q['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py']
        assert all(cfg['run'].get(k)==q['run'].get(k) for k in matched_keys)
        assert cfg['run']['constraint_set']=='none' and cfg['train_samples']==10 and cfg['validation_samples']==52
        assert cfg['pkl_sha256']==q['pkl_sha256'] and cfg['splits_json_sha256']==q['splits_json_sha256']
        assert cfg['runtime_provenance']==q['runtime_provenance']
        selected=read(NEW/f'fold0/seed{seed}/pooled/selected_cases.json')
        for split in ['train','validation']:
            prior={r['case_name']:r['metrics'] for r in rows if r['seed']==seed and r['split']==split}
            now={r['case_name']:r['metrics'] for r in selected if r['split']==split}
            assert set(prior)==set(now)
            for name in prior:
                for metric in ['gt_foreground_voxels','inner_voxel_count','outer_voxel_count','inner_face_count','outer_face_count','cross_face_count']:
                    assert prior[name][metric]==now[name][metric]
    for row in rows:
        row['metrics']['inner_hard_agreement']=1-row['metrics']['inner_disagree']
        row['metrics']['outer_hard_agreement']=1-row['metrics']['outer_disagree']
    summary={}
    keys=[k for k in rows[0]['metrics'] if k!='macro_dice_minus_historical_audit']
    for k in keys:
        values=[]
        for seed in range(3):
            t=st.mean(r['metrics'][k] for r in rows if r['seed']==seed and r['split']=='train');v=st.mean(r['metrics'][k] for r in rows if r['seed']==seed and r['split']=='validation')
            values.append(dict(seed=seed,train=t,validation=v,gap=t-v))
        values.append(dict(seed='Mean',**{f:st.mean(r[f] for r in values) for f in ['train','validation','gap']}))
        summary[k]=values
    save(OUT/'historical_baseline_cases.json',rows)
    save(OUT/'HISTORICAL_BASELINE.json',dict(selected=summary,provenance=provenance,matched_run_fields=matched_keys,identical_source_functions=equal_functions,validation_evaluator_difference='Only an inactive sagittal-step constraint flag was added; hard Dice calculation unchanged.',data_and_split_hashes_match=True,runtime_matches=True,cohorts_and_gt_geometry_match=True,historical_audit_completion=completion,caveat='Historical independently selected checkpoints, not contemporaneously rerun baseline. No baseline epoch-level train/validation trajectory assessed. Saved checkpoint bindings inherited from hash-verified historical audit; no baseline recomputation.'))
    text=['# Supplementary historical Dice-only baseline','This context uses only the existing hash-verified coherence audit; it is not part of the requested two-arm tables. Same fold-0 case IDs (10 train, 52 validation), data/split hashes, runtime, no augmentation, 75-epoch cap, optimizer, learning-rate schedule and early-stopping settings were verified. Original configuration hashes match saved checkpoint bindings. Model construction and preprocessing functions are AST-identical; validation hard-Dice calculation is unchanged (the only evaluator difference is an inactive added constraint flag). Band/face geometry counts also match case by case. Baseline selected epochs are **73, 74, 71** for seeds 0, 1, 2. This remains a historical comparison at independently selected epochs, not an exact contemporaneous baseline rerun or a matched temporal comparison.','Values are percentages and gaps are train minus validation in pp.','| Metric | Seed | Train | Validation | Gap |','| --- | --- | --- | --- | --- |']
    for name,k in [('Inner pair correctness','inner_both_correct'),('Inner equality satisfaction','inner_hard_agreement'),('Outer pair correctness','outer_both_correct'),('Outer equality satisfaction','outer_hard_agreement'),('Crossing correct_transition','cross_correct_transition'),('Macro Dice','macro_dice')]:
        for r in summary[k]:text.append('| '+' | '.join([name,str(r['seed'])]+[f"{r[f]*100:.6f}" for f in ['train','validation','gap']])+' |')
    text.append('The baseline already has an inner-correctness gap of **10.733824 pp** and an inner-equality gap of **5.331831 pp**, alongside a macro-Dice gap of **5.773682 pp**. This directly argues against interpreting the entire inner gap as unique to explicit constraints. The constrained arms enlarge the inner gap while improving validation Dice and boundary accuracy; the observations are consistent with a changed false-positive/false-negative balance and localized generalization difficulty. They do not isolate memorization caused by the constraint formulation. Baseline gaps vary considerably across seeds, especially seed 2. No new baseline inference or training was performed.')
    (OUT/'HISTORICAL_BASELINE.md').write_text('\n\n'.join(text[:4])+'\n'+'\n'.join(text[4:])+'\n')
    print('Verified baseline provenance, identical cohorts/geometry and preprocessing; supplementary outputs saved.')

if __name__=='__main__':main()
