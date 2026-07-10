#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="nnunetrpp_py38"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNETRPP_ROOT="${REPO_ROOT}/baselines/unetr_plus_plus"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda environment '${ENV_NAME}' already exists."
else
  echo "[INFO] Creating conda environment '${ENV_NAME}' (python=3.8)..."
  conda create -y -n "${ENV_NAME}" python=3.8
fi

echo "[INFO] Activating '${ENV_NAME}'"
conda activate "${ENV_NAME}"

echo "[INFO] Installing UNETR++ dependencies"
python -m pip install --upgrade pip
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r "${UNETRPP_ROOT}/requirements.txt"

echo "[DONE] Environment is ready."
echo "[INFO] Activate later with: conda activate ${ENV_NAME}"
