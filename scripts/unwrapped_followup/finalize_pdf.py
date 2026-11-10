"""Build the final, reproducible PDF from verified JSON/CSV evidence."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.graphics.shapes import Drawing, Line, Circle, String, Rect

OUT = Path('docs/experiments/unwrapped_followup_20261007/final_results')
PDF = OUT / 'hippocampus_unfolding_final_report.pdf'
NAVY = colors.HexColor('#18334B')
TEAL = colors.HexColor('#087F8C')
ORANGE = colors.HexColor('#C87B1A')
GRAY = colors.HexColor('#5A6877')
LIGHT = colors.HexColor('#EDF3F6')
WIDTH = A4[0] - 72
STYLES = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=25, leading=29, textColor=NAVY, spaceAfter=12),
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=19, leading=23, textColor=NAVY, spaceAfter=12),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=TEAL, spaceBefore=10, spaceAfter=7),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=14, textColor=NAVY, spaceAfter=9),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=GRAY, spaceAfter=7),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.6, leading=11.4, textColor=NAVY),
    'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=8.5, leading=11.4, textColor=colors.white),
}


def read(name):
    return json.loads((OUT / name).read_text())


def para(text, style='body'):
    return Paragraph(text, STYLES[style])


def tab(headers, rows, widths=None, size=8.6):
    style = ParagraphStyle('c', parent=STYLES['cell'], fontSize=size, leading=size+2.8)
    data = [[Paragraph(escape(str(v)), STYLES['head']) for v in headers]]
    data += [[Paragraph(str(v), style) for v in row] for row in rows]
    result = Table(data, colWidths=widths or [WIDTH/len(headers)]*len(headers), repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY), ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
        ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#C4D2DD'))]))
    return result


def page(title):
    return [PageBreak(), para(title,'h1')]


def number(v, digits=2):
    return f'{v:.{digits}f}'


def signed(v, digits=2):
    return f'{v:+.{digits}f}'


def count(v):
    return f'{v:,.0f}'


def main():
    runs = read('all_results.json')
    records = {r['id']:r for r in runs}
    seeds = read('seed_summary.json')
    comparisons = read('matched_comparisons.json')
    audits = read('checkpoint_audits.json')
    collection = read('collection_receipt.json')
    pairs = {'5pct': ['plane_position','unfold_position'], 'full':['plane_attention','unfold_attention']}
    names = {'5pct':'5% data', 'full':'Full data'}

    def key(f, a, s=0):
        return f'prior/{f}/{a}' if s==0 else f'{f}_replication_s{s}_{a}_none'

    def comparison(group, candidate):
        return next(c for c in comparisons if c['group']==group and c['candidate']==candidate)

    questions = [
        ['Does unfolding itself help?',
         '<b>No clear benefit over the matched plane branch.</b> Three-seed means: unwrapped 79.73%, plane 80.03% (-0.30 pp).',
         '<b>No consistent advantage.</b> Unwrapped 88.21%, plane 88.19% (+0.02 pp); the direction reverses in seed 2. This is not an equivalence test.'],
        ['Is depth balancing still useful?',
         'Small, architecture-dependent seed-0 effect: +0.21 pp for unwrapped, -0.19 pp for plane. The branch gain remains with Dice alone.',
         'Small seed-0 effect: +0.15 pp for unwrapped, -0.10 pp for plane. Both full-data case-bootstrap intervals include zero.'],
        ['Are improvements repeatable?',
         'Both branches beat native + depth in all three seeds. Mean gains: plane +5.15 pp; unwrapped +4.85 pp. Seed variability remains large.',
         'Both branches beat native + depth in all three seeds. Mean gains: plane +0.32 pp; unwrapped +0.34 pp.'],
        ['Does uncertainty locate useful corrections?',
         'Spatial gating beats uniform gating, but <b>neither beats ordinary fusion</b>. Unwrapped: 81.47% spatial, 79.72% uniform, 81.89% ordinary.',
         'Same ranking for unwrapped: 88.21% spatial, 87.88% uniform, 88.38% ordinary. Plane spatial and uniform gates are nearly tied, both below ordinary.'],
        ['Are anatomical errors improving?',
         'Compared with native + depth, unwrapped has far fewer extra voxels and better surfaces, but <b>more missed tissue in every seed</b>. A/P swaps are not consistently improved.',
         'Extra voxels and surface distances improve; missed tissue increases in every seed versus native + depth. Swaps improve in two seeds and worsen in one.']]
    (OUT/'five_question_answers.json').write_text(json.dumps([
        {'question':q[0],'5pct':q[1],'full':q[2]} for q in questions],indent=2)+'\n')
    story = [para('Hippocampus unfolding<br/>Final experimental report','title'),
             para('MSD Hippocampus / SwinUNETR / fold 0 / 7 October 2026','small'),
             para('<b>The auxiliary branch helps, but this study does not establish an advantage specific to unwrapping.</b> Its main benefit is fewer false positives. Consistent recovery of missing tissue remains unresolved.'),
             tab(['Question','5% data: 10 training cases','Full data: 208 training cases'], questions,
                 [109,207,WIDTH-316],size=9), Spacer(1,12),
             para('Completion: all 24 new runs and four checkpoint audits succeeded. The appendix includes all 20 original and 24 new conditions, with reused runs counted once. All scores use the same 52 validation cases. Dice is reported as a percentage; differences are percentage points (pp).','small'),
             para('The strongest inference is repeatable benefit from the overall branch setup. Matched plane controls prevent attributing that benefit to unfolding alone.','small')]

    story += page('What was trained and held fixed')
    story += [para('One end-to-end segmentation model','h2'),
        para('The native SwinUNETR and auxiliary branch were trained together. MRI samples in the auxiliary chart are encoded, mapped back through saved source voxel indices, and fused with native decoder features. Ordinary fusion predicts from <b>F + R</b>. No native encoder was frozen.'),
        tab(['Approach','Meaning'],[
            ['Native Dice','SwinUNETR with soft multiclass Dice including background; no auxiliary branch.'],
            ['Native + depth','Same model plus depth-balanced cross-entropy. Signed 2 mm bands have balanced influence within foreground and an 8 mm background margin. Sparse bands are downweighted.'],
            ['Matched plane branch','Same branch capacity, chart membership and sampled MRI information as unfolding, arranged in plane coordinates. It is the geometric control.'],
            ['Unwrapped branch','Cut-shell harmonic UV charts with distance from the longitudinal cutting plane. Separate signed sides retain depth; features are mapped into the original segmentation grid.'],
            ['Local / context / position / attention','Cumulative additions: fine encoder; coarse UV and chart context; longitudinal and signed-depth coordinates; then channel and spatial attention.'],
            ['Spatial / uniform gates','Multiply the residual R by detached native-path softmax entropy / log(3), either per voxel or averaged over the volume. No added head, parameters or loss.']],[115,WIDTH-115]),
        para('Geometry and optimization','h2'),
        para('The plane passes through the foreground centroid and contains the longest physical PCA axis. Its normal is the RAS right axis projected orthogonally to that long axis. Branch inputs use fixed guide predictions, not annotation masks. Annotations define the loss geometry and evaluation strata.'),
        para('Two out-of-fold 25-epoch guides generate training geometry; a fixed 25-epoch guide supplies validation geometry. Segmentor seeds 0, 1 and 2 share the seed-0 guide atlas and depth coefficients. Replication therefore varies the segmentor conditional on fixed geometry.'),
        para('64 x 64 x 64 input; feature size 24; batch size 1; AdamW learning rate 0.0001, weight decay 0.00001; StepLR halves the rate every 20 epochs; bfloat16; no augmentation. Maximum 75 epochs; minimum 25; patience 8; min_delta 0.0005. Select the highest mean patient A/P Dice checkpoint.'),
        para('Depth CE: five Dice-only warm-up epochs, then a five-epoch ramp. Frozen coefficients: 0.0029145569633 at 5%, 0.0803403871895 at full data. No SDF, topology term, or additional anatomical loss was introduced.','small')]

    story += page('Repeatability across three seeds')
    rows=[]
    for f, pair in pairs.items():
        for arm,label in [('depth','Native + depth'),(pair[0],'Plane'),(pair[1],'Unwrapped')]:
            v=seeds[f][arm]
            rows.append([names[f],label,*[number(s) for s in v['dice_pct']],number(v['mean_dice_pct']),number(v['sample_sd_dice_pp'])])
    story += [tab(['Data','Model','Seed 0','Seed 1','Seed 2','Mean','SD'],rows,[64,109,68,68,68,72,WIDTH-449]),
              Spacer(1,12), para('Both branches improve on native + depth in all six matched regime/seed comparisons. The 5% seed-2 runs are lower across all architectures. Unwrapping does not remove this variability.'),
              para('Unwrapped minus matched plane','h2')]
    rows=[]
    for f in pairs:
        for seed in (0,1,2):
            c=comparison('selected_unfold_vs_plane',key(f,pairs[f][1],seed))
            rows.append([names[f],seed,signed(c['dice_delta_pp']),f"[{c['case_bootstrap_low_pp']:+.3f}, {c['case_bootstrap_high_pp']:+.3f}]",f"{c['improved_cases']} / {c['worsened_cases']}"])
    story += [tab(['Data','Seed','Dice delta (pp)','95% case-bootstrap interval','Cases better / worse'],rows,[70,40,100,180,WIDTH-390]),
              Spacer(1,12),para('At 5%, the mean unfolding-specific difference is -0.302 pp. At full data it is +0.023 pp. Full-data seed differences change sign, and every per-seed interval includes zero. The small numerical mean does not establish equivalence.'),
              para('Interpretation of intervals','h2'),
              para('Intervals resample 52 paired validation cases 10,000 times with RNG seed 0. They condition on already-selected checkpoints, the reused fold and fixed guides. They do not quantify full-pipeline training uncertainty or independent-test performance. SD describes only three segmentor seeds, and no multiple-comparison correction is applied.'),
              para('The selected 5% pair uses position features; the full-data pair uses attention. The data-regime contrast is not an isolated experiment on training-set size. The original seed-0 plane-attention result (82.06%) also exceeds the selected 5% plane-position result (81.81%).','small')]

    story += page('Depth loss and uncertainty gating')
    rows=[]
    for f, pair in pairs.items():
        for arm in pair:
            d=records[f'{f}_dice_ablation_s0_{arm}_none']; ordinary=records[key(f,arm)]
            spatial=records[f'{f}_uncertainty_s0_{arm}_entropy'];uniform=records[f'{f}_uncertainty_s0_{arm}_mean_entropy']
            rows.append([names[f], 'Plane' if arm.startswith('plane') else 'Unwrapped',
                         *[number(r['macro_dice']*100) for r in [d,ordinary,spatial,uniform]]])
    story += [para('All entries here use seed 0. Gated models retain the depth objective; Dice-only ablations change only the objective.'),
              tab(['Data','Branch','Dice only','Depth + ordinary','Depth + spatial','Depth + uniform'],rows,[63,84,82,98,98,WIDTH-425]),
              para('Adding depth loss to the same architecture','h2')]
    rows=[]
    for f,pair in pairs.items():
        for arm in pair:
            c=comparison('add_depth',key(f,arm))
            rows.append([names[f],arm.replace('_',' '),signed(c['dice_delta_pp']),f"[{c['case_bootstrap_low_pp']:+.3f}, {c['case_bootstrap_high_pp']:+.3f}]"])
    story += [tab(['Data','Branch','Delta (pp)','95% case-bootstrap interval'],rows,[68,151,85,WIDTH-304]),
              Spacer(1,10),para('Depth balancing slightly benefits the unwrapped branch and slightly reduces plane-branch Dice in these runs. Its added value is small relative to the 5% branch gain. Removing depth still leaves Dice-only branches far above the native Dice baseline: plane +7.83 pp and unwrapped +7.52 pp at 5%; +0.43 and +0.38 pp at full data.'),
              para('Spatial gating versus the uniform control','h2')]
    rows=[]
    for f,pair in pairs.items():
        for arm in pair:
            c=comparison('spatial_vs_uniform',f'{f}_uncertainty_s0_{arm}_entropy')
            rows.append([names[f],arm.replace('_',' '),signed(c['dice_delta_pp']),f"[{c['case_bootstrap_low_pp']:+.3f}, {c['case_bootstrap_high_pp']:+.3f}]"])
    story += [tab(['Data','Branch','Delta (pp)','95% case-bootstrap interval'],rows,[68,151,85,WIDTH-304]),
              Spacer(1,10),para('<b>Ordinary fusion remains the best gated/ungated choice for every selected branch.</b> Spatial selection is better than uniform attenuation at 5% and for full-data unwrapped, but it does not improve on unattenuated fusion.'),
              para('Entropy is an uncalibrated proxy from a shared classifier. The uniform control matches mean gate strength only at a given model state; independently trained models can develop different entropy and residual magnitudes. These seed-0 tests cannot establish repeatability.','small')]

    story += page('What changed in the errors?')
    rows=[]
    for f,pair in pairs.items():
        for arm,label in [('depth','Native + depth'),(pair[0],'Plane'),(pair[1],'Unwrapped')]:
            v=seeds[f][arm]['mean_metrics']
            rows.append([names[f],label,count(v['missed']),count(v['extra']),count(v['swaps']),number(v['surface_distance_mean_mm'],3),number(v['hd95_mm'],3)])
    story += [para('Three-seed averages. FN, FP and A/P swaps are totals over 52 cases, averaged across seeds. Surface measures are case means; smaller is better.'),
              tab(['Data','Model','Missed (FN)','Extra (FP)','A/P swaps','MSD mm','HD95 mm'],rows,[57,99,78,78,71,69,WIDTH-452]),
              para('Fewer extra voxels, with a sensitivity cost','h2'),
              para('At 5%, unwrapped versus native + depth changes the mean counts by <b>+9,159 missed voxels, -39,218 extra voxels and +353 A/P swaps</b>. Mean surface distance falls from 0.999 to 0.739 mm. Compared with plane, unwrapped has more A/P swaps in all three seeds.'),
              para('At full data, the corresponding changes are <b>+1,435 missed voxels, -2,922 extra voxels and -102 A/P swaps</b>. Mean surface distance falls from 0.432 to 0.414 mm. The swap reduction is not uniform: two seeds improve, one worsens.'),
              para('Unwrapped spatial gate minus ordinary fusion (seed 0)','h2')]
    rows=[]
    for f,pair in pairs.items():
        c=comparison('gate_vs_ordinary',f'{f}_uncertainty_s0_{pair[1]}_entropy')
        rows.append([names[f],signed(c['dice_delta_pp']),f"{c['missed_delta']:+,}",f"{c['extra_delta']:+,}",f"{c['swaps_delta']:+,}",signed(c['surface_distance_mean_mm_delta'],3)])
    story += [tab(['Data','Dice pp','FN delta','FP delta','Swap delta','MSD delta mm'],rows,[65,77,95,95,95,WIDTH-427]),
              Spacer(1,12),para('Spatial gating reduces the missed-tissue count in the unwrapped model, but adds enough extra voxels to lower Dice and worsen mean surface distance in both regimes. This is a trade-off, not an overall improvement.'),
              para('What these numbers can and cannot say','h2'),
              para('FN is annotated foreground predicted as background; FP is predicted foreground outside annotation; swaps are anterior/posterior errors where both are foreground. A reduction in aggregate FN does not identify exactly which voxels were recovered, because some voxels may improve while others regress. The checkpoint audits on page 7 use paired voxel identities for that narrower comparison.'),
              para('Surface metrics use the combined foreground boundary, not the anterior/posterior interface. Mean surface distance pools both directional boundary distances within a patient, then averages patients. HD95 is the analogous patient-level 95th percentile, averaged across patients.','small')]

    story += page('Error distribution by depth')
    raw=list(csv.DictReader((OUT/'depth_band_metrics.csv').open()))
    buckets=['0-2','2-4','4-6','6-8','8-10','10+']
    aggregates={}
    for f,pair in pairs.items():
        aggregates[f]={}
        for arm in ['depth',*pair]:
            ids={key(f,arm,s) for s in (0,1,2)}
            bands=[dict(union=0,missed=0,extra=0,swaps=0) for _ in buckets]
            for row in raw:
                if row['run_id'] not in ids:continue
                b=int(row['signed_bin']);bucket=min(b if b>=0 else -b-1,5)
                for field in bands[bucket]:bands[bucket][field]+=int(row[field])
            for b in bands:b['error_pct']=100*sum(b[k] for k in ['missed','extra','swaps'])/b['union']
            aggregates[f][arm]=bands
    (OUT/'absolute_depth_summary.json').write_text(json.dumps(aggregates,indent=2)+'\n')
    rows=[]
    for f,pair in pairs.items():
        for i,b in enumerate(buckets):
            n,p,u=[aggregates[f][a][i] for a in ['depth',*pair]]
            rows.append([names[f],b,*[number(v['error_pct'],1) for v in (n,p,u)],
                         count(u['missed']/3),count(u['extra']/3),count(u['swaps']/3)])
    story += [para('Absolute distance from the cutting plane, combining the two signed sides for readability. Error rate = (FN + FP + A/P swaps) / foreground union. Rates pool counts across cases and three seeds; counts in the final columns are per-seed averages.'),
              tab(['Data','Depth mm','Native %','Plane %','Unwrap %','Unwrap FN','Unwrap FP','Unwrap swap'],rows,[52,55,64,61,66,77,77,WIDTH-452],size=8.0),
              para('How to read the depth bands','h2'),
              para('The table localizes where errors accumulate; it does not establish that each band has equal anatomical difficulty. The union denominator changes with the model prediction, and sparse outer bands can have unstable percentages. Signed per-case bands and exact totals are retained in the supporting CSV/JSON.'),
              para('Depth balancing remains a useful way to give sparse layers training influence. However, these results do not support the stronger claim that the added branch consistently restores missing tissue at difficult depths. The overall Dice improvement can coexist with increased FN, because FP decreases much more.'),
              para('The reference plane for these evaluation bands comes from the annotated foreground. It is an evaluation/loss coordinate system. The auxiliary MRI inputs instead use the frozen guide-derived charts. A cutting-plane distance is not the distance to the outer surface and is not an SDF.'),
              para('No new layer-specific or 3D signed-distance loss was tested in this follow-up. These tables should not be interpreted as evidence for such a term.','small')]

    story += page('What the checkpoint audits show')
    rows=[]
    for a in audits:
        s=a['summaries']
        rows.append([names[a['fraction']],a['arm'].replace('_',' '),*[number(s[k]['macro_dice']*100) for k in ('native_only','zero_mapped','normal')]])
    story += [para('These are ablations of the same jointly trained seed-0 checkpoints. Native-only here is <b>not</b> a separately trained baseline. Zero-mapped removes auxiliary features while retaining support and the learned fusion block.'),
              tab(['Data','Branch','Native path only','Zero mapped features','Normal fusion'],rows,[60,145,108,112,WIDTH-425]),
              Spacer(1,10),para('Mapped information contributes: removing it reduces Dice, especially for full-data unwrapped (88.38% to 85.28%). But a native-only ablation also changes the operating point of a jointly trained system, so its difference cannot isolate a standalone branch benefit.'),
              para('Exact foreground changes: normal fusion versus native path','h2')]
    rows=[]
    for a in audits:
        rows.append([names[a['fraction']],a['arm'].replace('_',' '),count(a['introduced_missed']),count(a['recovered_missed']),
                     count(a['introduced_missed_beyond_2mm']),number(a['introduced_missed_native_entropy_voxel_mean'],3)])
    story += [tab(['Data','Branch','Newly missed','Recovered FG','New FN >2 mm','Native entropy'],rows,[57,139,85,83,83,WIDTH-447],size=8.1),
              Spacer(1,12),para('<b>Fusion primarily prunes foreground in these audits.</b> Relative to each checkpoint\'s native path, it introduces thousands of missed foreground voxels and recovers zero in three conditions, only eight in full-data plane. Nearly all newly missed voxels lie within 2 mm of the annotation boundary.'),
              para('This supports the concern that current fusion is an effective false-positive suppressor rather than a reliable missing-tissue repair mechanism. It does not prove that a separately trained native baseline would make exactly these same errors.'),
              para('The entropy column is voxel-weighted mean normalized native entropy at newly missed foreground. It is descriptive: there is no matched correct-voxel entropy control here, so these means do not establish calibrated error detection.'),
              para('Foreground recovery only means changing background to some foreground class; the A/P label may still be wrong. Surface-distance bins here use unsigned distance to the annotation surface, which differs from the cutting-plane depth bands. All normal reloads reproduced saved Dice exactly.','small')]

    for f in ('5pct','full'):
        story += page('All results: '+names[f])
        story += [para('22 unique conditions in this regime: 10 original seed-0 models and 12 follow-up runs. Best/stop gives the selected epoch and total epochs completed. MSD and HD95 are in millimetres. FN/FP/swaps are totals over 52 validation cases.','small')]
        rows=[]
        for r in runs:
            if r['fraction']!=f:continue
            arm=r['arm'].replace('unfold','unwrap').replace('_',' ')
            if r['arm']=='dice':label='Native Dice / s0'
            elif r['arm']=='depth':label=f"Native + depth / s{r['seed']}"
            else:
                obj='Dice' if r['objective']=='dice' else 'depth'
                gate={'none':'','entropy':' / spatial','mean_entropy':' / uniform'}[r['gate']]
                label=f"{arm} / s{r['seed']} / {obj}{gate}"
            rows.append([label,number(r['macro_dice']*100),f"{r['best_epoch']}/{r['completed_epochs']}",
                         count(r['missed']),count(r['extra']),count(r['swaps']),number(r['surface_distance_mean_mm'],3),number(r['hd95_mm'],3)])
        story += [tab(['Model / seed / loss / gate','Dice %','Best / stop','FN','FP','Swaps','MSD','HD95'],rows,[167,45,49,56,56,52,49,WIDTH-474],size=7.6),
                  Spacer(1,8),para('Local -> context -> position -> attention are cumulative architectural variants. All original auxiliary models use depth loss. Gate variants retain depth. Full run IDs, class-specific Dice, foreground Dice, per-case metrics, epoch histories and checkpoint hashes are in all_results.csv and the evidence archive.','small')]

    story += page('Limits, preservation and cleanup')
    story += [para('What this experiment establishes','h2'),
              para('For this fold, schedule and fixed guide geometry, auxiliary branches improve validation Dice versus the native depth model across three segmentor seeds. The matched plane branch explains most or all of that gain. Ordinary fusion outperforms the tested entropy gates at seed 0. The anatomical trade-off is fewer extra voxels with more missed tissue.'),
              para('What remains unresolved','h2'),
              para('This is one reused validation fold with validation-selected architecture choices and checkpoints. It is not an independent test and does not establish clinical utility. Three seeds share the same training subsets and fixed guide pipeline. Dice-only and gating comparisons have one seed. The 75-epoch limit and early stopping describe this budget; they do not establish convergence or asymptotic rankings.'),
              para('The plane control matches capacity and corresponding MRI information within a branch pair. Comparisons against the native model also change architecture, guide-derived support and contextual information, so those gains cannot be assigned to a single mechanism. Absence of a consistent unfolding gain here does not prove that every possible unfolding architecture is ineffective.'),
              para('Verified evidence','h2'),
              para(f"Job 687700 completed successfully on 7 October 2026 at 18:38:59 Europe/Rome (exit 0), after 5 h 9 min 7 s. Verified: 24 requested follow-up completion records, 20 original conditions, four checkpoint audits, two CUDA preflights, and {collection['verified_artifact_references']:,} recorded artifact hashes including checkpoints and geometry references."),
              para(f"Transferred and SHA-256 verified {collection['preserved_files']} files; {collection['deduplicated_files']} duplicate original files map to canonical copies. Small prediction arrays, per-case results, histories, configurations, source snapshots, guide/atlas provenance, environment, logs and scheduler records are preserved. The analysis independently recomputed every run summary, best epoch and depth-error total."),
              para('Cleanup scope','h2')]
    plan_path=OUT/'cleanup_plan.json'
    if plan_path.exists():
        p=read('cleanup_plan.json')
        story += [para(f"The allowlisted inventory selects {len(p['delete']):,} unshared generated files/links ({p['bytes_to_delete']/2**30:.3f} GiB logical size) for deletion after this PDF and its evidence are verified. The exact paths are recorded in cleanup_plan.json.")]
    story += [para('The active tissue-profile campaign uses the original atlas and baseline result files. Those dependencies are retained, together with source code and runtime dependencies. Raw MRI/labels, unrelated jobs (including capped-depth), and local report/evidence are outside deletion scope. No active job is stopped or changed.'),
              para('The final cleanup_receipt.json records deletions, errors if any, retained-reference verification and freed bytes. final_quota.json records the refreshed BeeGFS quota. These receipts are produced after report verification; the PDF is the pre-cleanup scientific record.','small'),
              para('Evidence index','h2'),
              para('all_results.json/csv: 44 conditions. per_case_metrics.csv: 2,288 case rows. matched_comparisons.json/csv: paired differences and bootstrap intervals. seed_summary.json: seed variability. depth_band_metrics.csv and absolute_depth_summary.json: layer errors. checkpoint_audits.json: paired foreground changes. evidence/: source records. preservation_manifest.json and PDF QA receipt: local integrity and visual verification.','small')]

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#CDD9E0'))
        canvas.line(36,36,A4[0]-36,36)
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(GRAY)
        canvas.drawString(36,23,'HIPPOCAMPUS UNFOLDING  |  Fold 0  |  7 October 2026')
        canvas.drawRightString(A4[0]-36,23,str(doc.page))

    doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=36,leftMargin=36,topMargin=34,bottomMargin=49,
                          title='Hippocampus unfolding - final experimental report',author='Research results compiled with Codex')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(PDF)


if __name__=='__main__':
    main()
