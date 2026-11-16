"""Additional audit of conditional support before the standard verified archive."""
import json
from scripts.colab_hfh import constraint_archive as original
from scripts.colab_hfh.residual_band import POLICY, policy_dict, EXTRA_UPDATE_FLOPS
from scripts.colab_hfh.runtime import read,write


def validate_trace(output,protocol):
    if protocol.get('residual_band_policy')!=policy_dict():
        raise ValueError('Conditional policy changed')
    s=protocol['models']['nnunet']['objectives']['dice_bands']
    if s['residual_band_extra_flops_per_active_update']!=EXTRA_UPDATE_FLOPS:
        raise ValueError('Conditional arithmetic allowance changed')
    rows=[json.loads(x) for x in (output/'residual_band_trace.jsonl').read_text().splitlines()]
    if [r['charged_attempt'] for r in rows]!=list(range(s['prefix_updates']+1,s['max_steps']+1)):
        raise ValueError('Missing or replayed conditional training decisions')
    report=read(output/'residual_band_spatial_final.json')
    config=read(output/'config.json')
    for split,key in [('train','train_cases'),('validation','validation_cases')]:
        if [x['case_name'] for x in report['cases'] if x['split']==split]!=config[key]:
            raise ValueError('Spatial diagnostic case scope changed')
    allowed={x['annotation_sha256'] for x in report['cases'] if x['split']=='train'}
    for row in rows:
        if row['annotation_sha256'] not in allowed:
            raise ValueError('Non-training annotation used for conditional decision')
        if row['ready']!=(row['training_dice']>=POLICY.minimum_training_dice):
            raise ValueError('Readiness rule differs')
        if not row['ready'] and row['selected_errors']:
            raise ValueError('Auxiliary loss active before readiness')
        if row['selected_errors']!=sum(x['selected_errors'] for x in row['sides'].values()):
            raise ValueError('Selected error accounting differs')
        for side,entry in row['sides'].items():
            if entry['selected_width_mm'] is None:continue
            candidates=[x for x in entry['candidates'] if x['error_coverage']>=POLICY.minimum_error_coverage
                and (side!='inner' or x['shell_voxels']/row['foreground_voxels']<=POLICY.maximum_inner_foreground_fraction)]
            if (entry['total_errors']<POLICY.minimum_side_errors or not candidates
                    or entry['selected_width_mm']!=candidates[0]['width_mm']
                    or entry['selected_errors']!=candidates[0]['covered_errors']):
                raise ValueError('Conditional width/support policy differs')
    result=dict(attempts=len(rows),active_attempts=sum(r['selected_errors']>0 for r in rows),
        readiness_skips=sum(not r['ready'] for r in rows),training_only=True,
        extra_flops_already_charged=len(rows)*EXTRA_UPDATE_FLOPS,
        mean_selected_error_coverage=sum(r['total_error_coverage'] for r in rows)/len(rows))
    write(output/'residual_band_trace_audit.json',result)
    return result


def archive(output,durable,model,fold,seed,protocol,reference):
    p=read(protocol/'protocol.json')
    validate_trace(output,p)
    return original.archive(output,durable,model,fold,seed,protocol,reference)
