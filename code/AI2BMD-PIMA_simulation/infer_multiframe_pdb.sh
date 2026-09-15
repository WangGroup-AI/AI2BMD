#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER="${SCRIPT_DIR}/apptainer/ai2bmd.sif"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "Apptainer image not found: ${CONTAINER}" >&2
    exit 1
fi

exec apptainer exec --nv \
    --bind "${SCRIPT_DIR}:${SCRIPT_DIR}" \
    "${CONTAINER}" \
    bash -c '
        script="$1"
        shift
        user_args=("$@")
        set --
        source /opt/env
        exec python "$script" "${user_args[@]}"
    ' _ "${SCRIPT_DIR}/src/infer_multiframe_pdb.py --output-dir ${REPO_ROOT}/results/log_AI2BMD_PIMA_calculations" "$@"
