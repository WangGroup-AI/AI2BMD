#!/bin/bash
#SBATCH --job-name=AI2BMD_simulation_ala15_500ps
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.err

UNIQUE_ID="run_${SLURM_JOB_ID}"
BASE_OUTPUT_DIR="/home/wudianwei/AI2BMD/PUDP_run/PUDP_100pct_V8P1/ALA15_nowat/${UNIQUE_ID}_500ps"

# 定义容器内的执行命令
# 我们使用 apptainer exec 并在其内部运行一个 shell 来执行多条指令
apptainer exec --nv \
    --bind /home/wudianwei:/home/wudianwei \
    ai2bmd.sif /bin/bash -c "
        source /opt/env && \
        cd /home/wudianwei/AI2BMD && \
        python /home/wudianwei/AI2BMD/src/main.py --prot-file /home/wudianwei/AI2BMD/PUDP_run/ALA15_test/L-ALA15_1.pdb \
            --task simulation \
            --base-dir $BASE_OUTPUT_DIR \
            --ckpt-path ./src/ViSNet/checkpoints/visnet_pima_V8P1_trained_on_40m_PUDP_29epoch.zip \
            --sim-steps 500000 \
            --temp-k 300 \
            --timestep 1 \
            --preeq-steps 0 \
            --preprocess-method NONE \
            --record-per-steps 100 \
            --device-strategy large-molecule \
            --pima-mode
    "