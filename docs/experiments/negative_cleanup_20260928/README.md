# Verified unsuccessful-artifact cleanup

User explicitly requested deleting artifacts of experiments without progress or
improvement to free BeeGFS space for the MedSAM3 pilot.

Deleted exactly **14 weight files**, totaling **1,491,287,241 logical bytes
(1.389 GiB)**. The complete paths, sizes, SHA256 hashes, and reasons are in
`deleted.json`; its status is `complete`.

- Auxiliary-boundary run `boundary_auxiliary_20260923_34d3_v4/results_666967`:
  all 12 arm/fold combinations selected epoch zero. Eight treatment exports
  were verified byte-identical to retained selected controls, then removed.
  Four initialization/optimizer snapshots were also removed. The four selected
  control exports remain and their hashes were verified after deletion.
- Direct-contour run `direct_contour_20260923_34d3_v2`: removed its final
  treatment weight and smoke-test serialization. Its inner Dice was 0.882259
  versus matched control 0.882723, and ASSD 0.493022 versus 0.490045 mm.
  The matched control weight remains and passed post-deletion hash verification.

All experiment source, protocols, metrics, reports, and logs remain on the
cluster. No datasets, successful low-data/bands/edge weights, baseline audit
caches, or environments were deleted. The queue was empty immediately before
cleanup. No broad recursive directory deletion was used.

Initial quota was 91.13 / 93.13 GiB. Quota usage refreshes asynchronously and
new MedSAM3 source/adapter staging overlaps the subsequent accounting; do not
equate logical bytes removed with an exact quota delta.
