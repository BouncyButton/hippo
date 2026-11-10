"""Write the human-readable final retirement record from verified receipts."""
from pathlib import Path
import json,re,stat
BASE=Path(__file__).resolve().parent
verification=json.loads((BASE/'results_verification.json').read_text())
receipt=json.loads((BASE/'deletion_receipt.json').read_text())
evaluation=json.loads((BASE/'evaluation_gate.json').read_text())
supplement=json.loads((BASE/'evaluation_retention/results_verification.json').read_text())
assert verification['status']=='verified' and receipt['all_targets_absent'] and len(receipt['deleted'])==16
assert evaluation['status']=='verified'
labels={'1':'GeoFWI3D experiment outputs','2':'Outer-combination experiments','3':'94,603-budget baseline and bands campaigns','4':'Depth-supervision and depth-weight experiments','5':'Band-focus, evolution, Chamfer and native-branch experiments','6':'Capped-depth / delayed-SDF campaign'}
scope=json.loads((BASE/'scope.json').read_text());groups=scope['annotation_groups']
byname={r['target']:r for r in verification['archives']}
lines=['Completed experiment retirement — 14 October 2026','',
'User-approved policy: preserve results and predictions; delete checkpoint weights.',
'All 16 selected experiment directories were removed only after local archive verification and a fresh comparison with the original file hashes.',
'Pretrained SAM3 and MedSAM3 inputs, datasets, source repositories, the active cross-dataset campaign, and unrelated experiments were retained.','',
f"Original non-checkpoint files preserved: {sum(r['verified_regular_files'] for r in byname.values()):,}",
f"Checkpoint files excluded and deleted: {sum(r['excluded_checkpoint_files'] for r in byname.values()):,}",
f"Additional audit files preserved: {sum(r['verified_regular_files'] for r in supplement['archives'])}",
'Additional evaluation: 100 hippocampus checkpoint endpoints with predictions, Dice, FN/FP counts and directional A/P swaps; 9 GeoFWI3D runs with exact FN/FP reconstruction from saved voxel counts.',
f"Largest per-case Dice difference versus original evaluation: {evaluation['maximum_case_dice_difference']:.10g}",
'','Selected groups:']
for key,names in groups.items():
 rows=[byname[n] for n in names]
 lines.append(f"{key}. {labels[key]}: {len(names)} directories; {sum(r['verified_regular_files'] for r in rows):,} preserved files; {sum(r['excluded_checkpoint_files'] for r in rows)} checkpoint files deleted.")
lines+=['','Archive guide:',
'results_archives/ contains portable tar.gz archives with original paths and ARCHIVE_MANIFEST.json checksums.',
'results_catalog.json indexes original result paths; results_manifests/ lists every retained and excluded file.',
'evaluation_retention/results_archives/ contains supplemental predictions, voxel metrics, evaluation source and logs.',
'evaluation_gate.json provides searchable summaries of the 100 supplemental checkpoint evaluations.',
'geofwi3d_voxel_metrics.json contains the nine GeoFWI3D error-count audits.',
'deletion_receipt.json and evaluation_retention/deletion_receipt.json record exact deletions.',
'checkpoint_locations.txt documents the separate pretrained SAM3 and MedSAM3 files.',
'',
'To inspect an archive: tar -tzf results_archives/EXPERIMENT.tar.gz',
'To restore its preserved files into an existing destination: tar -xzf results_archives/EXPERIMENT.tar.gz -C DESTINATION',
'Checkpoint weights are deliberately not recoverable from these result archives.',
'The native-branch archive preserves one external atlas symlink; its BeeGFS target was retained.',
'', 'Final personal quota snapshot:', (BASE/'quota_after.txt').read_text().strip()]
issues=[x for r in byname.values() for x in r['readability']['preexisting_json_issues']]
if issues:lines+=['','Pre-existing JSON readability issues (original bytes preserved):',json.dumps(issues,indent=2)]
(BASE/'SUMMARY.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='complete',preserved_files=sum(r['verified_regular_files'] for r in byname.values()),deleted_checkpoints=sum(r['excluded_checkpoint_files'] for r in byname.values()),deleted_directories=len(receipt['deleted']),archive_gib=sum(r['archive_bytes'] for r in byname.values())/2**30,preexisting_json_issues=len(issues))),flush=True)
