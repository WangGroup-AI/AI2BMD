#!/usr/bin/env bash

nvidia-smi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="."

python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/ViSNet-Chignolin.yml" --load-model "${REPO_ROOT}/data/checkpoint/Chignolin.ckpt" --dataset-root "${REPO_ROOT}/data/AIMD-Chig" --task inference --log-dir "${REPO_ROOT}/results/${LOG_DIR}/log_Chig" --splits "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/Chignolin_splits.npz"
