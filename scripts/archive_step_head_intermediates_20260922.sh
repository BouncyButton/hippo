#!/usr/bin/env bash
# Move only redundant checkpoints from the new pilot to local temporary storage
# while BeeGFS user quota is constrained. Selected and recent resume files stay.
set -Eeuo pipefail

RUN_ROOT="${1:?pilot output root required}"
ARCHIVE_ROOT="${2:?local archive root required}"
test -f "${RUN_ROOT}/config.json"
mkdir -p "${ARCHIVE_ROOT}"
shopt -s nullglob

archive_once() {
  local arm arm_dir archive_dir count keep_from index
  local -a versions
  for arm in A B C D; do
    arm_dir="${RUN_ROOT}/${arm}"
    [[ -d "${arm_dir}" ]] || continue
    archive_dir="${ARCHIVE_ROOT}/${arm}"
    mkdir -p "${archive_dir}"
    versions=("${arm_dir}"/best_epoch_*.pt)
    count="${#versions[@]}"
    if [[ -f "${arm_dir}/done.json" ]]; then
      keep_from=$((count > 0 ? count - 1 : 0))
    else
      keep_from=$((count > 1 ? count - 2 : 0))
    fi
    for ((index=0; index<keep_from; index++)); do
      mv -n "${versions[index]}" "${archive_dir}/"
    done
    if [[ -f "${arm_dir}/done.json" ]]; then
      [[ ! -f "${arm_dir}/latest.pt" ]] || mv -n "${arm_dir}/latest.pt" "${archive_dir}/latest.pt"
      [[ ! -f "${arm_dir}/best.pt" ]] || mv -n "${arm_dir}/best.pt" "${archive_dir}/best.pt"
    fi
  done
}

if [[ "${3:-}" == "--once" ]]; then
  archive_once
  exit
fi
for ((cycle=0; cycle<960; cycle++)); do
  archive_once
  [[ ! -f "${RUN_ROOT}/summary.json" ]] || exit 0
  sleep 15
done
