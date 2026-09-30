# Fold-0 separated edge pilot: job 675451

Completed successfully (exit 0) on 29 September 2026 at 15:02 Europe/Rome,
after 35m14s. All six runs finished. MedSAM3 job 675308 also completed successfully
in 3h57m04s. No other folds were launched. Six selected-checkpoint hashes and all
562 downloaded report/metadata files were verified.

Selected validation macro Dice: pooled 77.7751%, separated 77.8598% (+0.0847
percentage points; all three seeds improve). The common epoch-60 comparison is
negative (-0.1302 points), and probability coherence worsens in both bands at the
selected checkpoints. Inner-versus-crossing parameter-gradient conflict occurs
in all 36 post-warmup probes per arm; combined edge-versus-Dice+bands conflict
occurs in 8/36 pooled probes and 0/36 separated probes. No PCGrad run was started.

Interpretation: ../../../reports/separated_edge_20260929/INTERPRETATION.md.
Full downloaded results: ../../../reports/separated_edge_20260929/results/.

Scope is strictly fold 0, seeds 0/1/2, pooled versus separated edge rules: six
new models. No further folds are submitted or expanded automatically. Preserve
the original 75-epoch cap and stopping policy, data subset, seeds, optimizer,
bands weights and warmup. Preserve original pooled edge coefficients; match the
separated rule's calibration-time median gradient budget on training cases only.
See PROTOCOL.md for exact controls, gradient measurements and endpoints.

Local and cluster checks passed: 15 tests in total, including loss partition
identity/gradients, independent spatial-audit agreement, coefficient matching,
RNG-preserving gradient probes, checkpoint recovery and report aggregation.
The cluster test runner was isolated; its training environment was not modified.

BeeGFS quota immediately before submission was 87.75/93.13 GiB used: 5.38 GiB
free, above the 1.6 GiB start requirement. Per user request, six latest/intermediate
checkpoints from completed older 200-epoch augmentation baseline/bands runs were
archived locally, individually SHA-256 verified, and removed remotely. This
cleanup released 1,177,127,424 allocated bytes (~1.10 GiB). Other concurrent storage
changes mean the full quota change should not be attributed to this cleanup.
All their selected best models remain on the cluster; calibration sources,
original 75-epoch references and MedSAM3 artifacts were preserved.

Recovery archive:
`reports/separated_edge_20260929/archive/200_epoch_intermediates.tar`.
ARCHIVE_VERIFIED.json records its absolute local path, archive checksum, all six
original paths and checksums. ARCHIVE_CLEANUP.json records the completed removal.
Archive members use paths relative to /mnt/beegfsstudents/home/3160552; restoring
one should first verify its recorded checksum and ensure its target is absent.

Cluster root:
`/mnt/beegfsstudents/home/3160552/separated_edge_20260929_01`.
Logs: `logs/675451.out`, `logs/675451.err`.
Final outputs: REPORT.md, SUMMARY.json, fold0/seed{0,1,2}/{pooled,separated}/,
including selected models, case audits, epoch-60 audits, learning curves and
parameter/logit gradient measurements. The launcher writes LAUNCHER_EXIT.json.
The final report includes the original baseline, bands and bands+edge comparisons.

The report and interpretation are complete. This pilot does not establish a
consistent coherence benefit. Further folds or a PCGrad experiment require a new
decision; neither was launched automatically. No scheduled monitoring was created.
