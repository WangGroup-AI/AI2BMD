export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python \"${SCRIPT_DIR}/src/main.py\" \
    --prot-file \"${SCRIPT_DIR}/testcases/1_rep_chig.c0.pdb\" \
    --base-dir \"${SCRIPT_DIR}\" \
    --ckpt-path \"${SCRIPT_DIR}/src/ViSNet/checkpoints\" \
    --ckpt-type new \
    --preeq-steps 1200 \
    --sim-steps 1000 \
    --temp-k 300 \
    --timestep 1 \
    --record-per-steps 100 \
    --write-solvent \
    --mm-method tinker-GPU \
    --solvent \
    --solvent-method AMOEBA \
    --frag-nonbonded-calc pima \
    --fragcalc fragment \
    --device-strategy large-molecule

echo 'Successfully finished running the simulation.'
