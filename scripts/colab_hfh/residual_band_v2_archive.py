"""Reject archives with a wrong active ramp or missing actual-gradient probes."""
import json
import math
from scripts.colab_hfh.residual_band_archive import validate_trace
from scripts.colab_hfh.constraint_archive import archive as original
from scripts.colab_hfh.runtime import read,write


def validate(output,protocol):
    result=validate_trace(output,protocol)
    rows=[json.loads(x) for x in (output/'residual_band_trace.jsonl').read_text().splitlines()]
    active=0;probes=[]
    for row in rows:
        active+=row['selected_errors']>0
        expected=min(active/9,1.) if row['selected_errors'] else 0.
        if row['active_update_count']!=active or row['applied_scale']!=expected:
            raise ValueError('Activation-count ramp differs')
        g=row['logit_gradients']
        for name in ('auxiliary_over_native','effective_auxiliary_over_native','fn_over_native','fp_over_native'):
            if not math.isfinite(g[name]) or g[name]<0:raise ValueError('Invalid actual gradient diagnostic')
        if not row['selected_errors'] and g['auxiliary_over_native']!=0:
            raise ValueError('Inactive BAND has a nonzero gradient')
        if 'parameter_gradients' in row:
            probes.append(row['charged_attempt'])
            if 'same_crop_post_update' not in row or 'optimizer_step_directional_loss_change' not in row:
                raise ValueError('Incomplete parameter/update probe')
            s=row['same_crop_post_update']
            for kind in ('fp','fn'):
                if s[kind+'_after']-s[kind+'_before']!=s[kind+'_introduced']['total']-s[kind+'_corrected']['total']:
                    raise ValueError('Post-step transitions do not reconcile')
    first=next((r['charged_attempt'] for r in rows if r['selected_errors']),None)
    expected=sorted(set([60,75,90]+([first] if first else [])))
    if probes!=expected:raise ValueError('Parameter probe schedule differs')
    result.update(active_ramp_verified=True,parameter_probe_updates=probes,
        diagnostic_flops_allowance=protocol['diagnostics']['parameter_probe_max_flops']+protocol['diagnostics']['logit_probe_max_flops'])
    write(output/'residual_band_v2_audit.json',result)
    return result


def archive(output,durable,model,fold,seed,protocol,reference):
    validate(output,read(protocol/'protocol.json'))
    return original(output,durable,model,fold,seed,protocol,reference)
