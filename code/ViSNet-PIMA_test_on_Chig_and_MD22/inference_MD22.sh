#!/usr/bin/env bash

nvidia-smi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="."

# Ac-Ala3-NHMe
python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/ViSNet-MD22-Ac_Ala3_NHMe.yml" --load-model "${REPO_ROOT}/data/checkpoint/Ac-Ala3-NHMe.ckpt" --dataset-root "${REPO_ROOT}/data/md22-dataset" --task inference --log-dir "${REPO_ROOT}/results/${LOG_DIR}/log_md22_Ac_Ala3_NHMe" --splits "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/MD22_Ac-Ala3-NHMe_splits.npz"

# DHA
python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/ViSNet-MD22-DHA.yml" --load-model "${REPO_ROOT}/data/checkpoint/DHA.ckpt" --dataset-root "${REPO_ROOT}/data/md22-dataset" --task inference --log-dir "${REPO_ROOT}/results/${LOG_DIR}/log_md22_DHA" --splits "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/MD22_DHA_splits.npz"

# Stachyose
python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/ViSNet-MD22-stachyose.yml" --load-model "${REPO_ROOT}/data/checkpoint/stachyose.ckpt" --dataset-root "${REPO_ROOT}/data/md22-dataset" --task inference --log-dir "${REPO_ROOT}/results/${LOG_DIR}/log_md22_Stachyose" --splits "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/MD22_Stachyose_splits.npz"

# AT-AT
python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/ViSNet-MD22-AT_AT.yml" --load-model "${REPO_ROOT}/data/checkpoint/AT-AT.ckpt" --dataset-root "${REPO_ROOT}/data/md22-dataset" --task inference --log-dir "${REPO_ROOT}/results/${LOG_DIR}/log_md22_AT_AT" --splits "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/MD22_AT-AT_splits.npz"

# AT-AT-CG-CG
python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/ViSNet-MD22-AT_AT_CG_CG.yml" --load-model "${REPO_ROOT}/data/checkpoint/AT-AT-CG-CG.ckpt" --dataset-root "${REPO_ROOT}/data/md22-dataset" --task inference --log-dir "${REPO_ROOT}/results/${LOG_DIR}/log_md22_AT_AT_CG_CG" --splits "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/MD22_AT-AT-CG-CG_splits.npz"

# Buckyball catcher
python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/ViSNet-MD22-buckyball_catcher.yml" --load-model "${REPO_ROOT}/data/checkpoint/buckyball-catcher.ckpt" --dataset-root "${REPO_ROOT}/data/md22-dataset" --task inference --log-dir "${REPO_ROOT}/results/${LOG_DIR}/log_md22_Buckyball_Catcher" --inference-batch-size 2 --splits "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/MD22_Buckyball-catcher_splits.npz"

# Double-walled nanotube
python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/ViSNet-MD22-double_walled_nanotube.yml" --load-model "${REPO_ROOT}/data/checkpoint/double-walled-nanotube.ckpt" --dataset-root "${REPO_ROOT}/data/md22-dataset" --task inference --log-dir "${REPO_ROOT}/results/${LOG_DIR}/log_md22_Double_Walled_Nanotube" --splits "${SCRIPT_DIR}/examples_MD22_AIMD-Chig/MD22_Double-walled-nanotube_splits.npz"
