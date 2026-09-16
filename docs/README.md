# Hippo documentation

Project guides, research notes, audits, and experiment protocols live here.
Subfolders mirror the source tree so each document retains its project context.
Commands and plain code/data paths are relative to the repository root unless a
document explicitly says otherwise. Linked source files and images remain in their original locations.

The repository root keeps its introductory README and AGENTS.md. Vendored library
documentation and ignored generated reports remain alongside their owning code or outputs.

## Start here

- [Repository guide](REPO_GUIDE.md)
- [Project handoff — 2026-09-08](PROJECT_HANDOFF_20260908.md)
- [Training run log](allenamenti.md)

## Documents

### Project notes

- [Hippocampus segmentation with learnable constraints — project handoff, 2026-09-08](PROJECT_HANDOFF_20260908.md) — `PROJECT_HANDOFF_20260908.md`
- [Repository audit — 2026-09-05](REPO_AUDIT_20260905.md) — `REPO_AUDIT_20260905.md`
- [Hippo — Repository Guide](REPO_GUIDE.md) — `REPO_GUIDE.md`
- [Research critique and falsifiers — 2026-09-05](RESEARCH_CRITIQUE_20260905.md) — `RESEARCH_CRITIQUE_20260905.md`
- [Thesis audit — hippocampus segmentation with loss constraints](THESIS_AUDIT_20260906.md) — `THESIS_AUDIT_20260906.md`
- [Training runs](allenamenti.md) — `allenamenti.md`

### Claude outputs

- [Thesis audit — hippocampus segmentation with loss constraints](Claude%20outputs/THESIS_AUDIT_20260906.md) — `Claude outputs/THESIS_AUDIT_20260906.md`

### baselines

- [Baselines](baselines/README.md) — `baselines/README.md`

### cluster protocol 20260830

- [Protocol amendment: inner focal-1 / outer BCE, 30 epochs](cluster_protocol_20260830/protocol_amendment_inner_focal1_30epoch.md) — `cluster_protocol_20260830/protocol_amendment_inner_focal1_30epoch.md`

### cluster protocol 20260902

- [Outer one-cut pilot protocol — 2026-09-02](cluster_protocol_20260902/README.md) — `cluster_protocol_20260902/README.md`
- [Band-dominant cut-location residual sweep](cluster_protocol_20260902/band_location_sweep_train32_20260902/REPORT.md) — `cluster_protocol_20260902/band_location_sweep_train32_20260902/REPORT.md`
- [BCE-band + one-cut frozen-logit feasibility audit](cluster_protocol_20260902/hybrid_feasibility_20260902/REPORT.md) — `cluster_protocol_20260902/hybrid_feasibility_20260902/REPORT.md`

### datasets

- [Datasets](datasets/README.md) — `datasets/README.md`

### evaluation

- [Evaluation](evaluation/README.md) — `evaluation/README.md`

### experiments

- [Translation experiments, 2026-09-05](experiments/equivariance_family_b_20260905/NEXT_PROTOCOL.md) — `experiments/equivariance_family_b_20260905/NEXT_PROTOCOL.md`
- [Verification, 2026-09-05](experiments/equivariance_family_b_20260905/VERIFICATION.md) — `experiments/equivariance_family_b_20260905/VERIFICATION.md`
- [Official baseline GPU audit and project direction](experiments/loss_constraint_followup_20260905/GPU_AUDIT_RESULTS_20260905.md) — `experiments/loss_constraint_followup_20260905/GPU_AUDIT_RESULTS_20260905.md`
- [Loss-constraint follow-up, 2026-09-05](experiments/loss_constraint_followup_20260905/PROTOCOL.md) — `experiments/loss_constraint_followup_20260905/PROTOCOL.md`
- [Local verification, 2026-09-05](experiments/loss_constraint_followup_20260905/VERIFICATION.md) — `experiments/loss_constraint_followup_20260905/VERIFICATION.md`

### scripts

- [Scripts](scripts/README.md) — `scripts/README.md`

### semantic constraints

- [Semantic Constraints: Architecture and Design Decisions](semantic_constraints/ARCHITECTURE.md) — `semantic_constraints/ARCHITECTURE.md`
- [Semantic Constraints](semantic_constraints/README.md) — `semantic_constraints/README.md`

### thesis

- [Exploratory fold-0 result from the existing saved probabilities](thesis/Surface-normal%20ordinal%20LogLTN/EXPLORATORY_FOLD0_RESULT.md) — `thesis/Surface-normal ordinal LogLTN/EXPLORATORY_FOLD0_RESULT.md`
- [Surface-normal one-cut and ordinal LogLTN: pre-training audit](thesis/Surface-normal%20ordinal%20LogLTN/README.md) — `thesis/Surface-normal ordinal LogLTN/README.md`
- [SwinUNETR constraint experiments](thesis/new_constraints/README.md) — `thesis/new_constraints/README.md`
- [Supervised A/P cut posterior](thesis/new_constraints/ap_cut/README.md) — `thesis/new_constraints/ap_cut/README.md`
- [Protocol-derived A/P plane](thesis/new_constraints/ap_plane/README.md) — `thesis/new_constraints/ap_plane/README.md`
- [Focal-logLTN interpretation of the two-band boundary constraint](thesis/new_constraints/bands/Focal%20Log-LTN.md) — `thesis/new_constraints/bands/Focal Log-LTN.md`
- [Two-step outer-boundary bands](thesis/new_constraints/bands/README.md) — `thesis/new_constraints/bands/README.md`
- [Outer-boundary band constraint: rationale and mathematical formulation](thesis/new_constraints/bands/SCIENTIFIC_RATIONALE_AND_FORMULATION.md) — `thesis/new_constraints/bands/SCIENTIFIC_RATIONALE_AND_FORMULATION.md`
- [Motivazione scientifica dell’equivarianza alle traslazioni](thesis/new_constraints/equivariance/MOTIVAZIONE_EQUIVARIANZA.md) — `thesis/new_constraints/equivariance/MOTIVAZIONE_EQUIVARIANZA.md`
- [Translation equivariance](thesis/new_constraints/equivariance/README.md) — `thesis/new_constraints/equivariance/README.md`
- [Translation equivariance only — MSD fold 0](thesis/new_constraints/equivariance/results/equivariance_only.md) — `thesis/new_constraints/equivariance/results/equivariance_only.md`
- [Outer-surface one-cut LogLTN pilot](thesis/new_constraints/onecut/README.md) — `thesis/new_constraints/onecut/README.md`
- [Stop-gradient translation teacher](thesis/new_constraints/teacher/README.md) — `thesis/new_constraints/teacher/README.md`
- [Paper reproduction: SwinUNETR + LTN](thesis/paper_reproduction/README.md) — `thesis/paper_reproduction/README.md`

### utils

- [Offline 3D validation viewer](utils/VIEW_FOLD0_3D.md) — `utils/VIEW_FOLD0_3D.md`
