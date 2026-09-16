#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

python ${SCRIPT_DIR}/src/infer_multiframe_pdb.py --output-dir ${REPO_ROOT}/results/log_AI2BMD_PIMA_calculations
