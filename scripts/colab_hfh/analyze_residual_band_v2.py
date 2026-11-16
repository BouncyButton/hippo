"""Verify v2 paired archives and summarize actual gradients and error transitions."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
from tempfile import TemporaryDirectory
import zipfile

ROOT=Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111'
OUT=ROOT/'nnunet_conditional_band_v2/analysis'


def main():
    from scripts.colab_hfh.residual_band_v2_archive import validate
    from scripts.colab_hfh.constraint_archive import paired_result
    rows=[];probes=[];traces=[]
    original=json.loads((ROOT/'bands_v1_payload.json').read_text())['references']
    for seed in range(3):
        with zipfile.ZipFile(OUT/f'nnunet_f0_s{seed}.zip') as z,TemporaryDirectory() as temp:
            read=lambda n:json.loads(z.read(n))
            p=read('protocol/protocol.json');ref=read('protocol/baseline_references.json')[f'nnunet_f0_s{seed}']
            assert ref==original[f'nnunet_f0_s{seed}']
            for name,digest in read('results/archive_manifest.json').items():
                assert hashlib.sha256(z.read('results/'+name)).hexdigest()==digest
            for key in ('residual_band_source_sha256','residual_band_v2_source_sha256'):
                for name,digest in p[key].items():assert hashlib.sha256(z.read('source/'+name)).hexdigest()==digest
            d=Path(temp)
            for name in z.namelist():
                if name.startswith('results/'):
                    assert len(Path(name).parts)==2
                    (d/Path(name).name).write_bytes(z.read(name))
            assert validate(d,p)==read('results/residual_band_v2_audit.json')
            pair=read('results/paired_comparison.json');assert paired_result(d,ref,p)==pair
            trace=[json.loads(l) for l in z.read('results/residual_band_trace.jsonl').decode().splitlines()]
            active=[r for r in trace if r['selected_errors']]
            with zipfile.ZipFile(ROOT/'nnunet_conditional_band_v1/analysis'/f'nnunet_f0_s{seed}.zip') as old:
                v1=json.loads(old.read('results/paired_comparison.json'))
                v1final=json.loads(old.read('results/validation_latest.json'))
                same_predictions=read('results/validation_latest.json')==v1final
                v1cal=json.loads(old.read('results/calibration.json'))
                current_cal=read('results/calibration.json')
                assert {k:v for k,v in current_cal.items() if k!='config_sha256'}=={k:v for k,v in v1cal.items() if k!='config_sha256'},'Calibration changed'
            rows.append(dict(seed=seed,baseline_pct=100*pair['baseline_dice'],v1_pct=100*v1['constraint_dice'],
                v2_pct=100*pair['constraint_dice'],delta_pp=pair['delta_dice_pp'],
                delta_vs_v1_pp=100*(pair['constraint_dice']-v1['constraint_dice']),
                fp_change=pair['constraint_foreground_fp']-pair['baseline_foreground_fp'],
                fn_change=pair['constraint_foreground_fn']-pair['baseline_foreground_fn'],
                first_active=active[0]['charged_attempt'],first_scale=active[0]['applied_scale'],active_updates=len(active),
                last_ramp_update=next(r['charged_attempt'] for r in active if r['applied_scale']==1.),
                median_actual_logit_gradient_ratio=st.median(r['logit_gradients']['auxiliary_over_native'] for r in active),
                maximum_actual_logit_gradient_ratio=max(r['logit_gradients']['auxiliary_over_native'] for r in active),
                median_fp_logit_gradient_squared_share=st.median(r['logit_gradients']['fp_gradient_squared_share'] for r in active),
                v1_v2_validation_metrics_identical=same_predictions))
            for r in trace:
                traces.append(dict(seed=seed,update=r['charged_attempt'],scale=r['applied_scale'],
                    active_count=r['active_update_count'],fn_selected=r['sides']['inner']['selected_errors'],
                    fp_selected=r['sides']['outer']['selected_errors'],**{k:v for k,v in r['logit_gradients'].items() if k!='l2'}))
                if 'parameter_gradients' not in r:continue
                q=r['parameter_gradients'];s=r['same_crop_post_update'];step=r['optimizer_step_directional_loss_change']
                probes.append(dict(seed=seed,update=r['charged_attempt'],scale=r['applied_scale'],
                    actual_logit_gradient_ratio=r['logit_gradients']['auxiliary_over_native'],
                    parameter_gradient_ratio=q['auxiliary_over_native'],cosine_fn_fp=q['cosine_fn_fp'],
                    cosine_aux_native=q['cosine_auxiliary_native'],
                    weighted_fn_parameter_norm=q['weighted_fn_l2'],weighted_fp_parameter_norm=q['weighted_fp_l2'],
                    auxiliary_direction_fn=q['auxiliary_directional_fn_loss_change'],auxiliary_direction_fp=q['auxiliary_directional_fp_loss_change'],
                    actual_optimizer_direction_fn=step['fn'],actual_optimizer_direction_fp=step['fp'],
                    fp_corrected=s['fp_corrected']['total'],fp_introduced=s['fp_introduced']['total'],
                    fn_corrected=s['fn_corrected']['total'],fn_introduced=s['fn_introduced']['total'],
                    fp_change=s['fp_after']-s['fp_before'],fn_change=s['fn_after']-s['fn_before']))
    report=dict(status='three_seeds_complete',checks='Archive hashes, both source manifests, frozen baseline references, matched native settings, ramp/probe audits and unchanged calibration verified',
        seeds=rows,mean_delta_pp=st.mean(r['delta_pp'] for r in rows),
        mean_delta_vs_v1_pp=st.mean(r['delta_vs_v1_pp'] for r in rows),
        fp_change=sum(r['fp_change'] for r in rows),fn_change=sum(r['fn_change'] for r in rows),
        probes=probes,
        parameter_summary=dict(probes=len(probes),median_auxiliary_native_ratio=st.median(p['parameter_gradient_ratio'] for p in probes),
            opposing_fn_fp=sum(p['cosine_fn_fp']<0 for p in probes),
            auxiliary_opposes_native=sum(p['cosine_aux_native']<0 for p in probes),
            auxiliary_first_order_increases_fn_loss=sum(p['auxiliary_direction_fn']>0 for p in probes),
            actual_optimizer_first_order_increases_fn_loss=sum(p['actual_optimizer_direction_fn']>0 for p in probes),
            probes_with_net_new_fn=sum(p['fn_change']>0 for p in probes)),
        caveats=['Parameter probes cover four selected training updates per run, not all cases or held-out outcomes.',
            'Directional derivatives are first-order proxies; FP/FN transitions use the actual total optimizer update, including native loss, clipping and momentum.',
            'Seeds1/2 had unchanged ramp schedules but validation shifted by +0.012/+0.040pp; these small differences cannot be attributed to the ramp. Calibration anchors were identical; numerical execution/instrumentation effects remain unresolved.',
            'Full paired v1/v2 prediction maps were not retained; compare archived case metrics without claiming voxel identity.',
            'Optimization cap and 90 updates unchanged; read-only diagnostics have a separately declared allowance.'])
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,values in [('seed_comparison.csv',rows),('parameter_probes.csv',probes),('actual_gradient_trace.csv',traces)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
