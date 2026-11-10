# SAM3 thesis update — 23 October 2026

The 126-run SAM3 GeoFWI3D campaign is incomplete: 110 completed, 16 outstanding. `inventory.json` records exact missing tasks and verification scope; `completed_runs.csv` preserves the reconciled completion records. No partial headline result is presented as a balanced three-fold, three-seed comparison.

The thesis now documents the verified protocol and a visible result placeholder. Final quantitative integration requires the remaining runs, full per-case archive checks and paired fold/seed aggregation. The existing completed studies and their totals are unchanged. The saved thesis compiled successfully with the desktop editor compiler; all references resolve and no duplicate labels were found.

`recovery.json` records recovery and verified cleanup. Job 714363 resumed the unchanged campaign, with one dependent successor (714364); the existing local archive collector was restored. No new monitoring automation was created. Two already completed runs were retired after original metadata was restored and verified; no optimizer updates were replayed by this recovery. The separate earlier Colab restart allowance remains documented and must enter resource accounting.

## Subsequent status check

`status_112.json` records 112 completed runs and fourteen outstanding tasks. The two additional fold-3/seed-2 runs (bands and signed distance) completed 25 updates and 101 evaluations each; their frozen-parameter audits and archive hashes passed. The local check covers 852 cluster metadata files.

The archive connection failed again during job 714363. The existing collector now has a campaign-specific continuation wrapper in `split_cluster_20261022/recovery_20261023/collect_and_resume.py`. It accepts only the known archive-outage failure, rechecks durable archives, closes completed runs on the CPU without overwriting original GPU provenance, and resumes the unchanged campaign. It refuses recovery for a user pause, an active account job, an occupied worker lock, or an unrelated failure. All four refusal checks passed. No scheduled notification or monitoring automation was added.
