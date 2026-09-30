"""Paired summaries of the frozen translation intervention."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
KEYS=['macro_dice','union_dice','boundary_union_errors','all_fp','all_fn','ap_swaps','cross_correct_transition','assd_voxels','surface_dice_1voxel','bands_bce']


def comparison(rows):
    result={}
    for k in KEYS:
        a=np.array([r['identity'][k] for r in rows]);b=np.array([r['tta13'][k] for r in rows]);d=b-a
        rng=np.random.default_rng(20260929)
        sample=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
        result[k]=dict(identity_mean=float(a.mean()),tta_mean=float(b.mean()),delta_mean=float(d.mean()),
            descriptive_case_interval95=np.quantile(sample,[.025,.975]).tolist(),
            decreased=int((d<0).sum()),unchanged=int((d==0).sum()),increased=int((d>0).sum()))
    result['case_count']=len(rows)
    result['shell_corrected']=sum(r['transitions']['shell_errors_corrected'] for r in rows)
    result['shell_introduced']=sum(r['transitions']['shell_errors_introduced'] for r in rows)
    result['mean_view_boundary_disagreement']=float(np.mean([np.mean([v['boundary_union_disagreement_voxels'] for v in r['views']]) for r in rows]))
    return result


def main():
    result=dict(arms={},geometry={},scope='Fixed intervention; descriptive paired-case intervals on the same reused cohort. No new training. No validation-tuned views or thresholds.')
    reference_hashes={}
    for f in sorted((ROOT/'results').glob('*.json')):
        x=json.loads(f.read_text())
        if 'cases' not in x:continue
        arm=x['arm'];result['arms'][arm]={}
        for r in x['cases']:
            key=(r['split'],r['case_name']);h=r['gt_geometry']['label_sha256']
            if key in reference_hashes:assert reference_hashes[key]==h
            reference_hashes[key]=h
        for split in ('train','validation'):
            rr=[r for r in x['cases'] if r['split']==split]
            if not rr:continue
            assert len(rr)==(50 if split=='train' else 52)
            s=comparison(rr)
            s['max_identity_drift']={k:max(abs(r['identity_drift'][k]) for r in rr) for k in rr[0]['identity_drift']}
            clean=[r for r in rr if all(v['lost_input_nonzero_voxels']==0 for v in r['views'])]
            s['all_views_unclipped_input_cases']=len(clean)
            s['any_gt_clipping_cases']=sum(any(v['lost_gt_foreground_voxels'] for v in r['views']) for r in rr)
            if clean:s['unclipped_input_subset']=comparison(clean)
            result['arms'][arm][split]=s
            if split not in result['geometry']:
                g={k:sum(r['gt_geometry'][k] for r in rr) for k in ('ap_faces','ap_endpoints','outer_endpoints')}
                g.update(outer_faces=sum(r['identity']['cross_face_count'] for r in rr),case_count=len(rr))
                g['pooled_outer_to_ap_face_ratio']=g['outer_faces']/g['ap_faces']
                result['geometry'][split]=g
    (ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Frozen translation intervention','',result['scope'],'',
        '| Model | Identity val Dice% | TTA val Dice% | Change pp | Net shell errors removed | Corrected / introduced | Input unclipped cases |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,data in result['arms'].items():
        if 'validation' not in data:continue
        r=data['validation'];d=r['macro_dice'];net=r['shell_corrected']-r['shell_introduced']
        lines.append(f"| {arm} | {100*d['identity_mean']:.4f} | {100*d['tta_mean']:.4f} | {100*d['delta_mean']:+.4f} | {net} | {r['shell_corrected']} / {r['shell_introduced']} | {r['all_views_unclipped_input_cases']}/52 |")
    lines+=['','The geometry ratio counts GT faces, whereas the error totals count unique voxels; it provides surface-size context, not a directly comparable error rate.','']
    for split,g in result['geometry'].items():
        lines.append(f"{split}: {g['outer_faces']} outer faces and {g['ap_faces']} A/P faces, ratio {g['pooled_outer_to_ap_face_ratio']:.2f}; {g['outer_endpoints']} outer endpoint voxels and {g['ap_endpoints']} A/P endpoints.")
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
