"""Audit fixed auxiliary-gradient strength after training, without updates."""
import json
import time

import numpy as np

from scripts.colab_hfh.constraint_calibration import components


def fixed_weight_statistics(rows, arm, weights):
    names=components(arm)
    if not rows or set(weights)!=set(names):
        raise ValueError('Incomplete gradient diagnostic')
    if any(r['gradient_names']!=['native',*names] for r in rows):
        raise ValueError('Gradient diagnostic order differs')
    grams=np.asarray([r['gradient_gram'] for r in rows],dtype=np.float64)
    if not np.isfinite(grams).all() or np.any(grams[:,0,0]<=0):
        raise ValueError('Degenerate native diagnostic gradient')
    coefficients=np.asarray([weights[n] for n in names],dtype=np.float64)
    ratios=np.sqrt(np.diagonal(grams,axis1=1,axis2=2)[:,1:]/grams[:,0,0,None])*coefficients
    squared=np.einsum('i,nij,j->n',coefficients,grams[:,1:,1:],coefficients)
    joint=np.sqrt(np.maximum(squared,0)/grams[:,0,0])
    result={n:dict(median=float(np.median(ratios[:,i])),p95=float(np.quantile(ratios[:,i],.95)))
            for i,n in enumerate(names)}
    result['joint']=dict(median=float(np.median(joint)),p95=float(np.quantile(joint,.95)))
    return result


def audit(calibrate, model, patients, arm, device, amp, criterion, protocol, output, schedule):
    """Reuse the RNG-preserving calibration probe only after final evaluation."""
    path=output/'gradient_strength_final.json'
    if path.exists():
        return
    initial=json.loads((output/'calibration.json').read_text())
    started=time.monotonic()
    # A conservative separate allowance covers a second calibration-sized
    # forward/logit-gradient probe plus the fixed-coefficient reductions.
    report=dict(status='started',phase='after_final_training_and_validation_evaluation',
        optimization_updates=0,diagnostic_flops_allowance=2*schedule['calibration_flops'],
        accounting='Separate post-training diagnostic allowance; excluded from optimization and calibration ledgers',
        weights=initial['weights'],gradient_space='detached_fp32_full_resolution_logits')
    if protocol.get('online_sdf_gradient_ratio') is not None:
        report['interpretation']='Counterfactual fixed-anchor strength for comparison; actual online coefficients are recorded per charged update in online_sdf_strength.jsonl'
    try:
        final=calibrate(model,patients,arm,device,amp,criterion,protocol)
        if [r['case_name'] for r in final['cases']]!=[r['case_name'] for r in initial['cases']]:
            raise ValueError('Diagnostic training cases differ')
        report.update(status='complete',anchor_model_sha256=initial['anchor_model_sha256'],
            final_model_sha256=final['anchor_model_sha256'],
            anchor=fixed_weight_statistics(initial['cases'],arm,initial['weights']),
            final=fixed_weight_statistics(final['cases'],arm,initial['weights']),cases=final['cases'])
        report['joint_median_change_factor']=report['final']['joint']['median']/report['anchor']['joint']['median']
    except ValueError as error:
        # A degenerate final probe does not invalidate already-saved model
        # evaluations. Preserve the diagnostic failure explicitly.
        report.update(status='unavailable',error=str(error))
    report['seconds']=time.monotonic()-started
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
