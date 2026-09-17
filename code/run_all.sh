#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
OUTPUT_FILE="${RESULTS_DIR}/output.log"
export PYTHONWARNINGS="ignore"

mkdir -p "${RESULTS_DIR}"

echo "Running ViSNet-PIMA inference on MD22, Chignolin, Trp-cage, and NCIA datasets..."
cd "${SCRIPT_DIR}/ViSNet-PIMA_test_on_Chig_and_MD22"
bash inference_Chig.sh "$@"
bash inference_MD22.sh "$@"

cd "${SCRIPT_DIR}/AI2BMD-PIMA_train"
bash inference_Trp-cage.sh "$@"

cd "${SCRIPT_DIR}/ViSNet-PIMA_test_on_NCIA"
bash inference_NCIA.sh "$@"

exec > >(tee "${OUTPUT_FILE}") 2>&1
python "${SCRIPT_DIR}/show_results.py"

echo "Running ViSNet-PIMA for simulation..."
cd "${SCRIPT_DIR}/AI2BMD-PIMA_simulation"
bash simulation.sh "$@"
