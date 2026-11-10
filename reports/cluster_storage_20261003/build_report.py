"""Build a read-only cluster storage report from the saved scan and evidence."""
import collections
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCAN = [json.loads(line) for line in (ROOT / 'inventory.jsonl').open()]
ENTRIES = [r for r in SCAN if r['record'] == 'entry']
SUMMARY = json.loads((ROOT / 'summary.json').read_text())
for folder in SUMMARY['folders']:
    folder['checkpoints'] = [e for e in folder['checkpoints'] if not (e['path'].endswith('.pth') and (Path(e['path']).name == 'distutils-precedence.pth' or Path(e['path']).name.startswith('__editable__.')))]
EVIDENCE = collections.defaultdict(list)
for line in (ROOT / 'evidence.jsonl').open():
    e = json.loads(line)
    EVIDENCE[e['top']].append(e)
QUOTA = json.loads((ROOT / 'quota.json').read_text())[0]
USED, LIMIT = map(int, QUOTA['space'].split('/'))
BY_PATH = {e['path']: e for e in ENTRIES}

def quota_bytes(es):
    # File data, two Buddy Mirror replicas, estimated 4-KiB storage allocation.
    return 2 * sum(((e['allocated'] + 4095) // 4096) * 4096 for e in es if e['kind'] == 'file')

def gb(n):
    return f'{n / 1e9:.6f}'

def short(n):
    return f'{n / 1e9:.3f} GB' if n >= 1e8 else f'{n / 1e6:.3f} MB' if n >= 1e5 else f'{n / 1024:.2f} KiB'

DESCRIPTIONS = {
 'adamw_preservation_20260923_01a0cd':'Reset-AdamW one-step preservation diagnostic: scripts, locked protocol, gradients and update audit results; no saved updated backbone.',
 'adamw_preservation_20260923_v2_01a0cd':'Second AdamW diagnostic isolating the head update with a matched backbone; supplementary protocol and audit outputs.',
 'ap_interface_weighted_full_20260920_01':'Staged source and preflight for the full-data, augmented A/P interface-distance-weighted experiment; no retained trained weights.',
 'ap_interface_weighted_full_20260920_02':'Production full-data comparison: augmented Dice, uniform A/P cross-entropy and interface-weighted A/P cross-entropy; runs, checkpoints and diagnostics.',
 'ap_swap_mechanism_20260919_01':'Frozen-prediction anterior/posterior mechanism audit; spatial probability exports in cache/, export manifest, source and logs.',
 'augmentation_replication_20260916_01':'Augmentation replication for seeds 1 and 2; frozen source, retained run weights and generalization audits.',
 'bands_augmented_20260916_04':'Augmented Dice+bands seed-0 run, calibration source/checkpoints, training outputs and generalization audit.',
 'baseline_replication_20260916_01':'Dice baseline replication for seeds 1 and 2; training runs, retained weights and seed-level generalization audits.',
 'boundary_auxiliary_20260923_34d3':'Initial endpoint auxiliary-supervision attempt; source, config and job records; no retained checkpoints.',
 'boundary_auxiliary_20260923_34d3_v2':'Second endpoint auxiliary-supervision attempt; source, config and error/job records; no retained checkpoints.',
 'boundary_auxiliary_20260923_34d3_v3':'Endpoint auxiliary smoke and superseded comparison records; discard manifest, small prediction arrays and results; checkpoints removed.',
 'boundary_auxiliary_20260923_34d3_v4':'Later endpoint auxiliary comparison and quota/resume cleanup records; retained prediction arrays, metrics, source and logs; no checkpoints remain.',
 'boundary_causal_20260929_01':'50-case seed-0 causal comparison: summed objectives and Dice, each with/without augmentation; four runs, selected/export weights and training/evaluation records.',
 'boundary_diversity_20260929_01':'10 distinct cases repeated to match the 50-case optimizer budget; summed-objective run, retained weights and completion metrics.',
 'boundary_pool_overfit_20260923_01a0cd':'Fitting-only whole-ray versus boundary-local pooling diagnostic; small head weights and JSON results.',
 'boundary_preservation_20260923_01a0cd':'Read-only A/E checkpoint audit of corrections and regressions in fixed surface strata; metrics and provenance, no checkpoints.',
 'boundary_reconstruction_20260923_34d3':'Contextual boundary reconstruction diagnostic: independent/context heads, smoke and full screen, small weights and aggregate results.',
 'boundary_replication_20260929_01':'Seed-1 replication of Dice versus augmented Dice on 50 training cases; retained run weights, logs and audits.',
 'boundary_translation_20260929_01':'Frozen seed-0 translation-view inference audit; identity plus 12 integer shifts, paired correction metrics; no new weights.',
 'boundary_translation_replication_20260929_01':'Seed-1 repeat of the fixed translation-view inference policy; JSON metrics and execution source/logs.',
 'correction_control_20260923_01a0cd':'Frozen-backbone residual correction-head comparison against scalar bias and existence BCE; head weights and fitting/inner/outer results.',
 'correction_discrimination_20260923_01a0cd':'Saved correction-head discrimination audit on fitting and inner cases; source and JSON results only.',
 'degree_bands_ab_20260924_01':'Initial staged degree-weighted bands A/B source/config package; no checkpoints.',
 'degree_bands_ab_20260924_02_3g':'Completed degree-weighted bands variants A/B plus selected-checkpoint boundary audits against Dice/ordinary-bands controls; retained model weights and metrics.',
 'degree_edge_lowdata_A_20260924_cleanup':'Cleanup receipt for the degree/edge low-data A experiment; one JSON record, no models.',
 'direct_contour_20260923_34d3':'Initial direct annotated-contour supervision pilot package, smoke/error records and archive; no retained model.',
 'direct_contour_20260923_34d3_v2':'Second direct contour pilot comparing Dice/CE with signed-distance boundary supervision; source, result records and one retained weight file.',
 'edge_coherence_20260928_01':'Inference-only spatial coherence audit of 27 low-data models; JSON metrics, summary plots and scripts.',
 'edge_consistency_20260924_01':'Signed face-contrast edge-consistency protocol, source and diagnostic/preflight records; no weights.',
 'edge_consistency_training_20260924_01':'Full-data edge-consistency training run with retained selected/export weights and comparison metrics.',
 'edge_generalization_audit_20260924_01':'Inference-only training/validation error comparison of edge, Dice and ordinary bands; per-case/aggregate metrics, no dense image exports.',
 'edge_lowdata_20260924_01':'5%-data edge training on fold 0, seeds 0/1/2; three run directories, selected/export weights and matched-control audits.',
 'edge_lowdata_folds12_20260924_01':'Low-data edge extension across folds 1 and 2; per-fold/per-seed run weights, frozen source and audits.',
 'function_head_20260922':'Boundary function-head code, launchers, logs and configuration; model outputs are stored separately in hippopotamus_runs.',
 'function_head_robustness_20260922':'Function-head robustness and follow-up analysis source, metrics and CSVs; no retained checkpoints in this folder.',
 'hippo':'Shared research project: datasets/preprocessing files, baselines, source, thesis artifacts, W&B logs and older SwinUNETR model directories. models/ dominates.',
 'hippo_constraint_eval_20260728.py':'Standalone historical constraint evaluator source file.',
 'hippo_equivariance_eval_20260728.py':'Standalone historical equivariance evaluator source file.',
 'hippopotamus_runs':'Shared run collection (51 immediate run directories): SwinUNETR overfit/early-stopping, boundary function robustness, nnU-Net and other baseline outputs/caches. Expand the breakdown for every run.',
 'low_data_5pct_augmented_seed0_20260917_01':'Staged source package for an initial 5%-data augmented seed-0 experiment; no retained trained weights.',
 'low_data_aug_5pct_200ep_fold0_20260919_01':'5%-data augmented Dice experiment, fold 0, seeds 0/1/2; checkpoint snapshots/export weights and dense train/validation prediction audits.',
 'low_data_aug_bands_5pct_200ep_fold0_20260919_01':'Matched 5%-data augmented Dice+bands experiment, seeds 0/1/2; run weights, three calibration-source snapshots and dense prediction audits.',
 'low_data_bands75_seed0_20260917_01':'75-epoch 5%-data bands seed-0 run with selected/export weights and generalization audit.',
 'low_data_baseline75_seed0_20260917_01':'75-epoch 5%-data Dice baseline seed-0 run with selected/export weights and generalization audit.',
 'low_data_bands_seed0_20260917_01':'Earlier 5%-data bands seed-0 experiment records and retained calibration-source checkpoints; no trained run directory remains.',
 'low_data_noaug_10pct_fold0_seed0_20260918_01':'Staged/source-only 10%-data no-augmentation fold-0 seed-0 package; no retained trained weights.',
 'low_data_noaug_12p5pct_fold0_seed0_20260918_01':'12.5%-data no-augmentation fold-0 seed-0 Dice/bands comparison; run weights, calibration checkpoint and audit exports.',
 'low_data_noaug_25pct_fold0_seed0_20260918_01':'25%-data no-augmentation fold-0 seed-0 Dice/bands comparison; run weights, calibration checkpoint and audit exports.',
 'low_data_noaug_5pct_200ep_es_fold0_seed0_20260918_01':'5%-data, 200-epoch-cap early-stopped no-augmentation Dice/bands seed-0 comparison; run weights, calibration and audits.',
 'low_data_noaug_5pct_200ep_es_fold0_seeds12_20260918_01':'Same early-stopped 5%-data experiment for seeds 1 and 2; paired runs, calibration sources and audits.',
 'low_data_noaug_5pct_200ep_fold0_seed0_20260918_01':'Staged/source-only non-early-stopped 200-epoch 5%-data no-augmentation package; no retained trained weights.',
 'low_data_noaug_5pct_400ep_es_fold0_seed2_20260918_01':'5%-data no-augmentation seed-2 comparison with a 400-epoch cap and early stopping; weights, calibration source and audits.',
 'matched_control_20260902':'Matched 50-epoch seed-0 Dice control, retained weights and dense baseline/bands evaluation exports.',
 'medsam3_binary_retirement_20260930':'Preserved results and hashed cleanup receipts for retired binary MedSAM3 studies; five small Markdown/JSON files, no weights/predictions.',
 'medsam3_discarded_pilots.md':'Written record of discarded MedSAM3 pilots.',
 'medsam3_fold3_extension_20260929':'Retired binary fold-3 extension: source, protocol, status and retirement records; no adapters or dense predictions remain.',
 'medsam3_multiclass_fold0_seed46_20260930':'Three-class BG/A/P MedSAM3 pilot, fold 0 seed 46: three fine-tuned adapter checkpoint files, dense NIfTI predictions, metrics and source.',
 'medsam3_multiclass_replication_20260930':'BG/A/P replication: folds 0/1/3 × seeds 17/83/191 × baseline/bands/bands+edge; 27 adapter checkpoint files, dense NIfTI predictions, aggregate analysis and source.',
 'medsam3_pilot_cleanup.json':'Small MedSAM3 pilot cleanup receipt.',
 'medsam3_replication_20260928':'Retired binary MedSAM3 replication: source and study/status/retirement records; no adapters or predictions remain.',
 'medsam3_runtime_20260928':'Shared MedSAM3 checkout, pretrained LoRA weights and auxiliary virtual environment; required runtime assets, not expendable experiment output.',
 'medsam3_supervised_5shot_20260928':'Retired supervised five-shot binary MedSAM3 pilot: source, protocol and retirement records; no adapters or predictions remain.',
 'negative_cleanup_20260928':'Two JSON receipts for previous negative-experiment cleanup.',
 'nnunet_recovery_20260920_01':'Recovered nnU-Net run with retained .pth weights, verified preprocessed cache, continuity probabilities and stopped-checkpoint audit.',
 'onecut_protocol_20260902':'Historical one-cut protocol/calibration metadata and tables; six files, no checkpoints.',
 'overfit_loss_flow_20260923_01a0cd':'Frozen-head loss/activation/gradient diagnostic source, protocol and original job records; no checkpoints.',
 'overfit_loss_flow_v2_20260923_01a0cd':'Second loss-flow diagnostic with decomposed fitting-only gradient results; no checkpoints.',
 'pca_refiner_eight_20260930_01':'Eight-case fitting diagnostic for a PCA anatomical coefficient refiner; small refiner weights, PCA/diagnostic arrays, figures and source.',
 'pca_refiner_fold0_208_20260930_01':'32-mode PCA correction regressor trained on 208 fold-0 cases and evaluated on 52 validation cases; refiner weights, arrays, diagnostics and figures.',
 'pcgrad_50cases_audit_20260929_01':'Inference/analysis audit of the 50-case PCGrad experiment; JSON metrics and scripts, no weights.',
 'pcgrad_50cases_seed0_20260929_01':'50-training-case seed-0 PCGrad run; selected/latest checkpoints, validation metrics, frozen source and job records.',
 'pcgrad_edge_20260929_01':'Three-seed fold-0 summed-objective versus PCGrad pilot; retained selected weights, gradient-conflict diagnostics, per-run metrics and source.',
 'presence_decisive_20260923_v2':'Presence mechanism revision 2: component/oracle and capacity/pooling diagnostic outputs with small presence-head checkpoint files.',
 'presence_decisive_20260923_v3':'Presence mechanism revision 3: capacity/pooling and scalar-presence feasibility code/results with small head checkpoint files.',
 'presence_decisive_20260923_v4':'Four-fold presence/capacity/pooling studies and aggregate outputs; many small head checkpoint files, metrics and protocol/source.',
 'presence_decisive_audit_20260923':'Component-oracle/surface-attribution presence audit; source, provenance and results, no checkpoints.',
 'presence_decisive_capacity_20260923':'Capacity/pooling mechanism package and job/config records; no retained checkpoints here.',
 'presence_feasibility_20260923':'Scalar-presence feasibility diagnostic: attainable thin-ray corrections, tables and JSON results; no checkpoints.',
 'presence_signed_gradients_20260923':'Signed fitting-batch gradient diagnostic of saved E models; source and JSON/CSV outputs.',
 'presence_signed_gradients_20260923_v2':'Second signed-gradient diagnostic with revised execution and result records; no checkpoints.',
 'project':'Small pre-existing project/configuration directory; seven YAML/JSON/shell files.',
 'python_build.sh':'Historical Python 3.9.16 installation shell script.',
 'regularization_fast_20260916_01':'Training-speed/overhead audit, baseline/dropout comparison records and source; no checkpoints remain.',
 'regularization_round1_20260916_01':'Full-data regularization/augmentation round records, frozen source and generalization audits; no checkpoints remain.',
 'renderer_presence_one_20260923_01a0cd':'Frozen-E intervention setting presence to one only in rendering; metrics and scripts, no checkpoints.',
 'renderer_state_overfit_20260923_01a0cd':'Fitting-only test of access to renderer state; small head weights, objective histories and JSON results.',
 'seed1_replicate_20260903':'Historical seed-1 matched Dice/bands 50-epoch replication, retained checkpoints/export weights and calibration records.',
 'seed2_four_arm_600ep_20260919_01':'600-epoch seed-2 comparison: noaug Dice, noaug bands, augmented Dice and augmented bands; epoch-200/400/best/latest checkpoints and dense prediction audits.',
 'separated_edge_20260929_01':'Fold-0 three-seed pooled versus separated inner/outer/crossing edge pilot; retained selected weights, reports, source and prior cleanup manifests.',
 'separated_edge_generalization_20260929_01':'Inference-only generalization follow-up to separated-edge training; recomputed bands and error metrics, no weights.',
 'spectral_shape_fold0_20260930_01':'Fold-0 spectral anatomical shape analysis; mode/oracle/prediction arrays, small saved model state, results and plots.',
 'transfer.log':'Tiny log of a failed historical SSH/rsync transfer.',
 'translation_followup_matched_20260906_02':'Historical matched augmentation/translation follow-up, retained seed-0 checkpoints/export weights, source and completion records.',
 'wandb':'One historical local W&B run and debug logs; four regular files.',
 '.conda':'One live hippocampus conda environment (54,495 files), mostly PyTorch/CUDA/native libraries. No large package-download cache is present.',
 '.codex':'Cluster Codex installation package and small state/log databases.',
 '.cache':'Small cache directories; negligible reclaimable space.',
 '.local':'Small user-local data/configuration directory.',
 '.anaconda':'Small Anaconda metadata directory.',
 '.codex-backups':'Small Codex backup metadata.',
 '.config':'Small application configuration directories.',
 '.mozilla':'Small browser profile/configuration directories.',
 '.nv':'Small NVIDIA runtime cache directory.',
 '.ssh':'SSH configuration/key material: metadata measured; file contents were not read.',
 '.netrc':'Credential configuration: metadata measured; file contents were not read.',
}

def description(f):
    name=f['name']
    if name in DESCRIPTIONS:return DESCRIPTIONS[name]
    if name.endswith('.tar.gz'):
        members=next((e['archive_members'] for e in EVIDENCE[name] if 'archive_members' in e),[])
        return f'Compressed upload/source payload archive; {sum(m["kind"]=="file" for m in members)} member files. Member names and sizes are listed below.'
    if name.startswith('low_data75_seed'):
        return '75-epoch 5%-data Dice/bands paired replication for the named seed; retained selected/export weights and baseline/bands audits.'
    if name.startswith('low_data_seed'):
        return 'Earlier 5%-data paired experiment records and retained calibration-source checkpoints; trained run directory no longer present.'
    if name.startswith('low_data_noaug_seed'):
        return '5%-data no-augmentation Dice/bands comparison for the named seed; selected/export weights, calibration snapshots and generalization audits.'
    if name.startswith('low_data_noaug_fold'):
        return '5%-data no-augmentation Dice/bands comparison across seeds 0/1/2 on the named fold; run weights, calibration sources and audit outputs.'
    if name.startswith('.'):
        return 'Small hidden user configuration/history file or directory; metadata measured, private contents not read.'
    return 'See the measured child-directory breakdown, exact file metadata and collected experiment evidence.'

all_checkpoints=[e for f in SUMMARY['folders'] for e in f['checkpoints']]
hash_groups=collections.defaultdict(list)
for line in (ROOT/'calibration_hashes.jsonl').open():
    e=json.loads(line)
    assert e['unchanged_during_hash']
    hash_groups[e['sha256']].append(e)
duplicates=[]
for digest,es in hash_groups.items():
    if len(es)>1:
        keep=next((e for e in es if 'checkpoint_best.pt' in e['path']),es[0])
        duplicates.append({'sha256':digest,'keep':keep['path'],'redundant_paths':[e['path'] for e in es if e['path']!=keep['path']]})
tier1=[BY_PATH[p] for d in duplicates for p in d['redundant_paths']]
tier1_paths={e['path'] for e in tier1}
tier2=[e for e in all_checkpoints if Path(e['path']).name=='checkpoint_latest.pt' and e['path'] not in tier1_paths and str(Path(e['path']).with_name('checkpoint_best.pt')) in BY_PATH]
cache_roots={'ap_swap_mechanism_20260919_01','low_data_aug_5pct_200ep_fold0_20260919_01','low_data_aug_bands_5pct_200ep_fold0_20260919_01','seed2_four_arm_600ep_20260919_01'}
tier3=[e for e in ENTRIES if e['kind']=='file' and e['path'].split('/')[0] in cache_roots and Path(e['path']).suffix in {'.npz','.npy'} and ('/audits/' in e['path'] or '/cache/' in e['path'])]
tiers=[('Verified duplicate calibration snapshots',tier1,'Seven SHA-256-identical latest/best pairs. Retain each listed best copy; preserve or update any code referencing the redundant path before deduplication.'),('Additional latest optimizer snapshots',tier2,'Each path has a retained best checkpoint beside it. Removing these loses exact training resumption; selected models and epoch-200/400 scientific comparisons remain.'),('Regenerable dense prediction caches',tier3,'Frozen-prediction A/P cache and old augmented/four-arm audit arrays. Preserve metrics, export manifests, source and required selected/epoch checkpoints; regeneration needs inference.')]
cleanup={'deleted_anything':False,'quota_method':'Two replicas, each file allocation rounded to 4096 bytes; estimates, not per-directory server quota readings.','verified_duplicate_groups':duplicates,'tiers':[{'name':name,'estimated_quota_bytes':quota_bytes(es),'files':len(es),'consequence':note,'paths':[e['path'] for e in es]} for name,es,note in tiers]}
(ROOT/'cleanup_candidates.json').write_text(json.dumps(cleanup,indent=2))

estimated_home=quota_bytes(ENTRIES)
esc=html.escape
head='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bocconi cluster storage — 3 October 2026</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#18302e;background:#f5f7f5;margin:0}main{max-width:1450px;margin:auto;padding:36px 24px}h1{font-size:30px;letter-spacing:-.6px;margin:0 0 8px}h2{margin-top:32px;font-size:21px}p{max-width:1050px}.meta{color:#576763}.cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.card{padding:18px 24px;border:1px solid #d8e0dc;border-radius:8px;background:white;min-width:180px}.card strong{font-size:27px;display:block}.bar{height:10px;background:#e6ece8;border-radius:8px;overflow:hidden}.bar span{display:block;height:100%;background:#be4931}table{border-collapse:collapse;width:100%;background:#fff}th,td{padding:11px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #e4e9e5}th{background:#eaf0ec;font-size:12px;text-transform:uppercase;letter-spacing:.5px;position:sticky;top:0}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.name{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere;min-width:250px}.purpose{max-width:490px;font-size:13px}summary{cursor:pointer;color:#206b53}details{margin-top:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f2f5f3;padding:12px;border-radius:4px}details table{font-size:12px}input,select{font:inherit;border:1px solid #bbc9c0;border-radius:6px;padding:10px;background:white}input{width:min(500px,75%)}.tools{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.tiny{font-size:12px;color:#576763}.badge{font-size:11px;padding:2px 7px;background:#edf2ef;border-radius:5px}a{color:#206b53}.scroll{overflow:auto}.details-cell{padding:0}button{font:inherit;padding:8px 12px;border-radius:6px;border:1px solid #bbc9c0;background:#fff;cursor:pointer}@media(max-width:700px){main{padding:20px 12px}.card{min-width:125px}h1{font-size:24px}th,td{padding:8px}.name{min-width:230px}}
</style><main><h1>Bocconi cluster storage inventory</h1>'''
parts=[head,f'<p class="meta">Read-only audit • 3 October 2026 • <code>/mnt/beegfsstudents/home/3160552</code> • {len(SUMMARY["folders"])} top-level entries • {len(ENTRIES):,} filesystem entries • zero scan errors</p>',f'<div class="cards"><div class="card">Server quota used<strong>{USED/1e9:.2f} / {LIMIT/1e9:.0f} GB</strong><div class="bar"><span style="width:{USED/LIMIT*100:.2f}%"></span></div></div><div class="card">Remaining quota<strong>{(LIMIT-USED)/1e9:.2f} GB</strong>{(LIMIT-USED)/2**30:.2f} GiB</div><div class="card">Checkpoint file data, mirrored<strong>{quota_bytes(all_checkpoints)/1e9:.2f} GB</strong>{len(all_checkpoints)} retained weight/checkpoint files</div><div class="card">Proposed recovery<strong>{sum(quota_bytes(es) for _,es,_ in tiers)/1e9:.2f} GB</strong>Nothing deleted</div></div>', '<p>Each row describes what is currently present, rather than assuming that historical completion manifests imply checkpoints still exist. Expand a row for child-directory sizes, file types, exact retained checkpoint names, largest files and available protocol evidence. The full metadata inventory lists every file, directory and symlink.</p>',f'<p class="tiny"><b>Accounting:</b> the exact user quota is {USED:,} / {LIMIT:,} bytes ({USED/2**30:.3f} / {LIMIT/2**30:.3f} GiB). BeeGFS Buddy Mirror keeps two data replicas; 113 home/top-level layouts and 821 checkpoint/largest-file layouts were checked. Per-folder quota values are <b>estimates</b>: regular-file client allocation rounded to 4 KiB × 2, excluding metadata-only directories/symlinks. Full-home estimate = {estimated_home:,} bytes; server counter exceeds it by {USED-estimated_home:,} bytes ({(USED-estimated_home)/1e6:.3f} MB, {(USED-estimated_home)/USED*100:.4f}% of used quota). The client cannot attribute this remainder exactly to folders. No hard-linked regular files were found. Symlinks were recorded without following targets. Quota refresh can lag by 30 seconds.</p>', '<h2>Cleanup candidates, ordered by consequence</h2><div class="scroll"><table><thead><tr><th>Candidate</th><th class="num">Files</th><th class="num">Quota recovered, estimate</th><th>What retaining/removing it means</th></tr></thead><tbody>']
for name,es,note in tiers:
    parts.append(f'<tr><td>{esc(name)}</td><td class="num">{len(es)}</td><td class="num">{short(quota_bytes(es))}</td><td>{esc(note)}</td></tr>')
parts += ['</tbody></table></div><p>The three candidate lists are disjoint. They preserve the live conda environment, shared MedSAM3 runtime, datasets, selected best models, multiclass MedSAM3 adapters, experiment protocols and result tables. Large folders containing current models are not blanket deletion candidates. A copy elsewhere on this same BeeGFS quota does not free quota. <a href="cleanup_candidates.json">Exact candidate paths and verified duplicate hashes</a>.</p>', '<h2>Every folder and top-level file</h2><div class="tools"><input id="query" type="search" placeholder="Search folder name or contents" aria-label="Search folder name or contents"><select id="sort" aria-label="Sort"><option value="size">Largest quota footprint first</option><option value="name">Alphabetical</option></select><label><input id="hidden" type="checkbox" checked style="width:auto"> Include hidden entries</label><button id="reset">Reset</button></div><p class="tiny" id="count"></p><div class="scroll"><table id="inventory"><thead><tr><th>Folder / file and current contents</th><th class="num">Regular files</th><th class="num">Client allocated GB<br>one copy, includes directory blocks</th><th class="num">Quota GB<br>estimate, two replicas</th><th class="num">% of 100 GB quota</th></tr></thead><tbody>']
text_lines=[f'Bocconi cluster storage inventory — 3 October 2026\nHome: /mnt/beegfsstudents/home/3160552\nServer quota: {USED:,}/{LIMIT:,} bytes; {(LIMIT-USED):,} bytes free.\nPer-folder quota is estimated using two replicas and 4-KiB file allocation rounding.\nFull-home estimate {estimated_home:,}; unattributed server difference {USED-estimated_home:,} bytes.\nNo remote files were modified or deleted.\n']
folder_entries=collections.defaultdict(list)
for e in ENTRIES:folder_entries[e['path'].split('/')[0]].append(e)
for f in sorted(SUMMARY['folders'],key=lambda f:-quota_bytes(folder_entries[f['name']])):
    name=f['name'];es=folder_entries[name];q=quota_bytes(es);desc=description(f);f['quota_estimate_4k_bytes']=q;f['description']=desc
    artifacts=f"{f['counts'].get('file',0):,} regular files; {f['counts'].get('dir',0):,} directories; {f['counts'].get('symlink',0):,} symlinks; {len(f['checkpoints'])} checkpoint/weight files."
    types=', '.join(f'{k}: {v["files"]}' for k,v in list(f['extensions'].items())[:8])
    detail=[f'<p class="tiny">{esc(artifacts)} Exact client blocks: {f["allocated"]:,} bytes; logical regular-file size: {f["logical_bytes"]:,} bytes; estimated quota: {q:,} bytes.</p>','<b>Immediate children and their complete subtrees</b><table><tr><th>Name</th><th>Files</th><th>Client GB</th><th>Quota GB, estimate</th></tr>']
    for c in f['children']:
        ces=[e for e in es if (e['path'].split('/')[1] if len(e['path'].split('/'))>1 else '(entry itself)')==c['name']]
        cq=quota_bytes(ces)
        detail.append(f'<tr><td class="name">{esc(c["name"])}</td><td>{c["counts"].get("file",0)}</td><td>{gb(c["allocated"])}</td><td>{gb(cq)}</td></tr>')
    detail.append('</table><details><summary>All file types</summary><pre>'+esc('\n'.join(f'{k}: {v["files"]} files; client allocated {v["allocated"]:,} bytes' for k,v in f['extensions'].items()))+'</pre></details>')
    if f['checkpoints']:
        detail.append('<details><summary>Every retained checkpoint / weight file</summary><pre>'+esc('\n'.join(f'{e["path"]} | {e["bytes"]:,} logical bytes | quota estimate {quota_bytes([e]):,} bytes' for e in sorted(f['checkpoints'],key=lambda e:e['path'])))+'</pre></details>')
    detail.append('<details><summary>Largest 25 files</summary><pre>'+esc('\n'.join(f'{e["path"]} | {e["bytes"]:,} logical bytes | {e["allocated"]:,} client allocated bytes' for e in f['largest_files']))+'</pre></details>')
    for e in EVIDENCE[name]:
        if 'archive_members' in e:
            detail.append('<details><summary>Every archive member</summary><pre>'+esc('\n'.join(f'{m["path"]} | {m["bytes"]:,} bytes | {m["kind"]}' for m in e['archive_members']))+'</pre></details>')
        elif e.get('path','').endswith('.md'):
            detail.append('<details><summary>Protocol evidence: '+esc(e['path'])+'</summary><pre>'+esc(e['text'])+'</pre></details>')
    parts.append(f'<tr data-name="{esc(name,quote=True)}" data-size="{q}" data-search="{esc((name+" "+desc+" "+types).lower(),quote=True)}"><td><div class="name">{esc(name)} <span class="badge">{f["kind"]}</span></div><div class="purpose">{esc(desc)}</div><details><summary>Measured contents and exact paths</summary>{"".join(detail)}</details></td><td class="num">{f["counts"].get("file",0):,}</td><td class="num">{gb(f["allocated"])}</td><td class="num">{gb(q)}</td><td class="num">{q/LIMIT*100:.4f}%</td></tr>')
    text_lines += [f'\n{name}\n  {desc}\n  {artifacts}\n  Client allocation: {f["allocated"]:,} bytes ({f["allocated"]/1e9:.6f} GB); quota estimate: {q:,} bytes ({q/1e9:.6f} GB; {q/LIMIT*100:.4f}% of quota).\n  File types: {types}\n  Immediate child subtrees: '+ '; '.join(c['name']+' ('+str(c['counts'].get('file',0))+' files, '+gb(c['allocated'])+' client GB)' for c in f['children'])+'\n']
parts += ['''</tbody></table></div><h2>Audit files</h2><p><a href="inventory.jsonl">Complete filesystem metadata inventory</a> · <a href="summary.json">Machine-readable per-folder breakdown</a> · <a href="folder_inventory.txt">Plain-text inventory</a> · <a href="evidence.jsonl">Read experiment metadata/protocol evidence</a> · <a href="quota.json">Exact BeeGFS user quota</a> · <a href="cleanup_candidates.json">Cleanup candidate paths</a></p><p class="tiny">Archive listings inspect member names and sizes only; nothing was extracted. Private credentials and user histories were not read. This is a point-in-time audit. No Slurm jobs were queued/running for this user at the audit check.</p></main><script>
const rows=[...document.querySelectorAll('#inventory > tbody > tr')],body=document.querySelector('#inventory > tbody');
function update(){const q=document.querySelector('#query').value.toLowerCase(),show=document.querySelector('#hidden').checked,sort=document.querySelector('#sort').value;rows.sort((a,b)=>sort==='size'?Number(b.dataset.size)-Number(a.dataset.size):a.dataset.name.localeCompare(b.dataset.name));let n=0,total=0;rows.forEach(r=>{body.append(r);const ok=r.dataset.search.includes(q)&&(show||!r.dataset.name.startsWith('.'));r.hidden=!ok;if(ok){n++;total+=Number(r.dataset.size)}});document.querySelector('#count').textContent=n+' entries shown • estimated quota footprint '+(total/1e9).toFixed(3)+' GB';}
document.querySelector('#query').addEventListener('input',update);document.querySelector('#sort').addEventListener('change',update);document.querySelector('#hidden').addEventListener('change',update);document.querySelector('#reset').addEventListener('click',()=>{document.querySelector('#query').value='';document.querySelector('#sort').value='size';document.querySelector('#hidden').checked=true;update()});update();
</script></html>''']
(ROOT/'report.html').write_text(''.join(parts))
(ROOT/'folder_inventory.txt').write_text(''.join(text_lines))
SUMMARY['accounting']={'server_used_bytes':USED,'server_limit_bytes':LIMIT,'home_estimate_bytes':estimated_home,'unattributed_difference_bytes':USED-estimated_home,'method':'Two replicas of regular-file allocated blocks rounded to 4096 bytes. Client allocations and logical sizes are measured; per-folder quota contributions are estimates.'}
(ROOT/'summary.json').write_text(json.dumps(SUMMARY,indent=2))
print(json.dumps({'folders':len(SUMMARY['folders']),'quota_used':USED,'home_estimate':estimated_home,'difference':USED-estimated_home,'cleanup':[{'name':n,'files':len(es),'quota_bytes':quota_bytes(es)} for n,es,_ in tiers],'html_bytes':(ROOT/'report.html').stat().st_size},indent=2))
