#!/bin/bash
#SBATCH --job-name=AI2BMD_optimization
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.err

UNIQUE_ID="run_${SLURM_JOB_ID}"
BASE_OUTPUT_DIR="/home/wudianwei/AI2BMD/PUDP_run/PUDP_100pct_V8P1/optimization/${UNIQUE_ID}"

# 定义容器内的执行命令
# 我们使用 apptainer exec 并在其内部运行一个 shell 来执行多条指令
apptainer exec --nv \
    --bind /home/wudianwei:/home/wudianwei \
    ai2bmd.sif /bin/bash -c "
        source /opt/env && \
        cd /home/wudianwei/AI2BMD && \
        python /home/wudianwei/AI2BMD/src/main.py --prot-file /home/wudianwei/AI2BMD/PUDP_run/PUDP_100pct_V8P1/optimization/WW/ww_1_0_0.pdb \
            --task optimization \
            --base-dir $BASE_OUTPUT_DIR \
            --ckpt-path ./src/ViSNet/checkpoints/visnet_pima_V8P1_trained_on_40m_PUDP_29epoch.zip \
            --preprocess-method NONE \
            --device-strategy large-molecule \
            --pima-mode \
            --fmax 0.015 \
            --max-steps 1000 \
            --max-step-size 0.1
    "