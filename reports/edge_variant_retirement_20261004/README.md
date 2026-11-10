# Active edge variant retirement

On 2026-10-04 the user selected the original foreground edge-consistency loss and requested removal of the separated edge variant from active code.

Removed the variant's implementation, tests, training launcher and summary scripts. Retired the PCGrad edge experiment scripts that imported that implementation and its trainer; leaving those launchers would have created broken imports. The generic PCGrad utility and its independent tests remain available.

The original edge implementation and its workflows remain in `thesis/new_constraints/edge_consistency.py` and `scripts/train_edge_lowdata.py`. No separated-edge imports or settings remain in the active `scripts/`, `thesis/`, `semantic_constraints/`, or `evaluation/` Python/shell sources.

`manifest.json` records the 14 removed paths and their hashes. Before removal, their exact contents were archived in `retired_source.tar.gz`, then read back and checked against those hashes. Existing dated experiment snapshots, reports and deployment records remain historical artifacts and were not rewritten. Cluster deployments and past jobs were not changed.

Validation: 12 original-edge tests passed; remaining active source references were searched for the retired module, class, launcher names and `edge_arm` settings.
