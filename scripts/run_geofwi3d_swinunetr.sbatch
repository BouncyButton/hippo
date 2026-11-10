#!/usr/bin/env bash
#SBATCH --job-name=geofwi3d-swin-lowdata
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail
CAMPAIGN="$1"
GEOFWI3D_PYTHON="$2"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${CAMPAIGN}/matplotlib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${CAMPAIGN}"
"${GEOFWI3D_PYTHON}" source/submit_geofwi3d.py --campaign "${CAMPAIGN}"
"${GEOFWI3D_PYTHON}" - "${CAMPAIGN}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

campaign = Path(sys.argv[1])
config = json.loads((campaign / 'campaign.json').read_text())
subprocess.run([sys.executable, str(campaign / 'source/train_geofwi3d_swinunetr.py'),
               '--root', config['dataset_root'], '--protocol', str(campaign / 'protocol.json'),
               '--output', str(campaign / 'results'), '--device', 'cuda',
               '--fractions', *config['fractions'], '--methods', *config['methods']], check=True)
PY
