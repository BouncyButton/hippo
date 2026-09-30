#!/usr/bin/env bash
#SBATCH --job-name=boundary-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=boundary-audit-%j.out
#SBATCH --error=boundary-audit-%j.err
set -Eeuo pipefail
AUDIT_DIR="${SLURM_SUBMIT_DIR:?Submit from the frozen audit package}"
SOURCE="${HOME}/function_head_robustness_20260922"
PYTHON_BIN="${HOME}/.conda/envs/hippocampus/bin/python"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/boundary-audit-mpl-${SLURM_JOB_ID}"
export PYTHONPATH="${SOURCE}:${PYTHONPATH:-}"
cd "${AUDIT_DIR}"
sha256sum --check MANIFEST.sha256
# Build the reference in place from results already stored on this cluster.
# No patient masks or per-case medical evaluation records are uploaded.
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
source = Path.home() / "presence_decisive_20260923_v2/results_audit_666651"
provenance = json.loads((source / "provenance.json").read_text())
cases = []
for path in sorted(source.glob("fold*_cases.json")):
    for row in json.loads(path.read_text()):
        if row["arm"] in ("A", "E", "E_raw"):
            cases.append({key: row[key] for key in ("fold", "case", "arm", "assd_mm", "union_dice")})
assert len(cases) == 624
reference = {key: provenance[key] for key in ("pkl_sha256", "splits_sha256", "source_sha256")}
reference.update(reference_source=str(source), cases=cases)
Path("reference.json").write_text(json.dumps(reference, indent=2) + "\n")
PY
"${PYTHON_BIN}" run_function_boundary_preservation.py \
  --source "${SOURCE}" \
  --root "${HOME}/hippopotamus_runs/boundary_function_robustness_20260922_666174" \
  --pkl "${HOME}/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
  --splits "${SOURCE}/datasets/Dataset101_MSD/splits_final.json" \
  --reference "${AUDIT_DIR}/reference.json" \
  --output "${AUDIT_DIR}/results_${SLURM_JOB_ID}"
