nvidia-smi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

echo 'Starting learning curve and ablation study inference tasks, unit of "Scalar MAE" is eV, unit of "Forces MAE" is eV/Å'

echo 'Inference 20% Trp-cage finetuned model with pretraining...'

python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_Trp-cage/pima_ft_20pct.yml" --load-model "${REPO_ROOT}/data/learning_curve_and_ablation_study_checkpoint/pima_ft_20pct.ckpt" --dataset-root "${REPO_ROOT}/data/Trp-cage/reserved_test_set" --task inference --log-dir "${REPO_ROOT}/results/log_learning_curve_and_ablation_studys/pima_ft_20pct"

echo 'Inference 40% Trp-cage finetuned model with pretraining...'

python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_Trp-cage/pima_ft_40pct.yml" --load-model "${REPO_ROOT}/data/learning_curve_and_ablation_study_checkpoint/pima_ft_40pct.ckpt" --dataset-root "${REPO_ROOT}/data/Trp-cage/reserved_test_set" --task inference --log-dir "${REPO_ROOT}/results/log_learning_curve_and_ablation_studys/pima_ft_40pct"

echo 'Inference 60% Trp-cage finetuned model with pretraining...'

python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_Trp-cage/pima_ft_60pct.yml" --load-model "${REPO_ROOT}/data/learning_curve_and_ablation_study_checkpoint/pima_ft_60pct.ckpt" --dataset-root "${REPO_ROOT}/data/Trp-cage/reserved_test_set" --task inference --log-dir "${REPO_ROOT}/results/log_learning_curve_and_ablation_studys/pima_ft_60pct"

echo 'Inference 80% Trp-cage finetuned model with pretraining...'

python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_Trp-cage/pima_ft_80pct.yml" --load-model "${REPO_ROOT}/data/learning_curve_and_ablation_study_checkpoint/pima_ft_80pct.ckpt" --dataset-root "${REPO_ROOT}/data/Trp-cage/reserved_test_set" --task inference --log-dir "${REPO_ROOT}/results/log_learning_curve_and_ablation_studys/pima_ft_80pct"

echo 'Inference 100% Trp-cage finetuned model with pretraining...'

python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_Trp-cage/pima_ft_100pct.yml" --load-model "${REPO_ROOT}/data/learning_curve_and_ablation_study_checkpoint/pima_ft_100pct.ckpt" --dataset-root "${REPO_ROOT}/data/Trp-cage/reserved_test_set" --task inference --log-dir "${REPO_ROOT}/results/log_learning_curve_and_ablation_studys/pima_ft_100pct"

echo 'Inference 100% Trp-cage finetuned model without pretraining...'

python "${SCRIPT_DIR}/train.py" --conf "${SCRIPT_DIR}/examples_Trp-cage/pima_ft_ablation.yml" --load-model "${REPO_ROOT}/data/learning_curve_and_ablation_study_checkpoint/pima_ft_ablation.ckpt" --dataset-root "${REPO_ROOT}/data/Trp-cage/reserved_test_set" --task inference --log-dir "${REPO_ROOT}/results/log_learning_curve_and_ablation_studys/pima_ft_ablation"

echo 'All learning curve and ablation study inference tasks finished!'
